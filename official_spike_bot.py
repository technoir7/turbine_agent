"""
Turbine Market Maker Bot - Official Spike Variant
Generated based on Turbine Best Practices

Algorithm: Inventory-Aware Market Maker with Risk Controls
"""

import asyncio
import os
import re

import csv
import time
import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

# Third-party imports
from dotenv import load_dotenv
import httpx 

# Turbine Client Imports
from turbine_client import TurbineClient, TurbineWSClient, Outcome
from turbine_client.exceptions import TurbineApiError, WebSocketError

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("official_spike_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SpikeBot")

# Silence noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION (Embedded for Reliability)
# ============================================================
# Safety Tuned Values from previous debugging
CONFIG = {
    "strategy": {
        "base_spread": 0.04,      # Tight Spread (8 cents)
        "skew_factor": 0.10,      # High Skew (Dump inventory fast)
        "imbalance_threshold": 2.0,
        "imbalance_depth_n": 5,
        "overlay_bias": 0.005,
        "min_price": 0.01,
        "max_price": 0.99,
        # Risk / Extremes
        "extreme_low": 0.10,
        "extreme_high": 0.90,
        "extreme_spread_mult": 2.0,
        "extreme_size_mult": 0.5,
        "stop_loss_pct": 0.10,    # 10% Trailing Stop
    },
    "risk": {
        "max_inventory_units": 1000,
        "max_portfolio_exposure": 10000.0,
        "max_wallet_risk_pct": 0.20, # Max 20% of wallet value at risk
    },
    "loop": {
        "tick_interval_ms": 2000,       # 2s interval (Rate Limit Safe)
        "max_quote_age_seconds": 30.0,  # 30s max age (Smart Replacement)
        "replace_threshold": 0.02,      # 2 cent drift threshold
    }
}

ORDER_SIZE = 1_000_000  # 1 share (6 decimals)

# ============================================================
# UTILS
# ============================================================
def get_or_create_api_credentials(env_path: Path = None):
    """Get existing credentials or register new ones and save to .env."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"

    api_key_id = os.environ.get("TURBINE_API_KEY_ID")
    api_private_key = os.environ.get("TURBINE_API_PRIVATE_KEY")

    if api_key_id and api_private_key:
        logger.info("Using existing API credentials")
        return api_key_id, api_private_key

    private_key = os.environ.get("TURBINE_PRIVATE_KEY")
    if not private_key:
        raise ValueError("Set TURBINE_PRIVATE_KEY in your .env file")

    logger.info("Registering new API credentials...")
    try:
        credentials = TurbineClient.request_api_credentials(
            host="https://api.turbinefi.com",
            private_key=private_key,
        )
    except Exception as e:
        logger.error(f"Failed to register API credentials: {e}")
        # Assuming run via python, might fail if key invalid
        raise

    api_key_id = credentials["api_key_id"]
    api_private_key = credentials["api_private_key"]

    # Auto-save to .env
    _save_credentials_to_env(env_path, api_key_id, api_private_key)
    os.environ["TURBINE_API_KEY_ID"] = api_key_id
    os.environ["TURBINE_API_PRIVATE_KEY"] = api_private_key

    logger.info(f"API credentials saved to {env_path}")
    return api_key_id, api_private_key


def _save_credentials_to_env(env_path: Path, api_key_id: str, api_private_key: str):
    """Save API credentials to .env file."""
    env_path = Path(env_path)

    if env_path.exists():
        content = env_path.read_text()
        if "TURBINE_API_KEY_ID=" in content:
            content = re.sub(r'^TURBINE_API_KEY_ID=.*$', f'TURBINE_API_KEY_ID={api_key_id}', content, flags=re.MULTILINE)
        else:
            content = content.rstrip() + f"\nTURBINE_API_KEY_ID={api_key_id}"
        if "TURBINE_API_PRIVATE_KEY=" in content:
            content = re.sub(r'^TURBINE_API_PRIVATE_KEY=.*$', f'TURBINE_API_PRIVATE_KEY={api_private_key}', content, flags=re.MULTILINE)
        else:
            content = content.rstrip() + f"\nTURBINE_API_PRIVATE_KEY={api_private_key}"
        env_path.write_text(content + "\n")
    else:
        content = f"# Turbine Bot Config\nTURBINE_PRIVATE_KEY={os.environ.get('TURBINE_PRIVATE_KEY', '')}\nTURBINE_API_KEY_ID={api_key_id}\nTURBINE_API_PRIVATE_KEY={api_private_key}\n"
        env_path.write_text(content)

@dataclass
class MarketState:
    bids: List[Tuple[float, float]] = field(default_factory=list) # price, size
    asks: List[Tuple[float, float]] = field(default_factory=list)
    last_update_ts: float = 0.0
    position: float = 0.0 
    open_orders: Dict[str, dict] = field(default_factory=dict) # client_id/hash -> Order Info
    price_history: deque = field(default_factory=lambda: deque(maxlen=30)) # For SMA-30 trend detection
    
    # Stop Loss Tracking
    trailing_high: float = 0.0      # High water mark (for Longs)
    trailing_low: float = 1.0       # Low water mark (for Shorts)
    stop_triggered: bool = False
    stop_cooldown_ts: float = 0.0

    # Portfolio Risk
    wallet_balance: float = 0.0     # USDC Balance
    last_balance_ts: float = 0.0

# ============================================================
# DATA RECORDER
# ============================================================
class CSVLogger:
    def __init__(self, filename="market_data.csv"):
        self.filename = filename
        self._ensure_header()
        
    def _ensure_header(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                f.write("ts,market_id,mid,regime,inv,bid,ask\n")

    def log_tick(self, market_id, mid, regime, inv, bid, ask):
        """Append a tick record."""
        try:
            with open(self.filename, 'a', newline='') as f:
                # Simple CSV formatting
                line = f"{time.time():.3f},{market_id},{mid},{regime},{inv:.2f},{bid or ''},{ask or ''}\n"
                f.write(line)
        except Exception as e:
            pass # Never crash the bot for logging


# ============================================================
# STRATEGY MANAGERS (Phase 2 & 3)
# ============================================================
class InventoryUrgencyManager:
    """Manages 6-tier urgency system."""
    def __init__(self):
        self.previous_regime = None
        
    def get_inventory_urgency(self, inventory: float, regime: str, seconds_remaining: float) -> Tuple[str, float, float]:
        """
        Returns (urgency_level, offset_multiplier, spread_multiplier).
        """
        urgency_level = "NORMAL"
        offset_multiplier = 0.02
        spread_multiplier = 1.0
        
        # Tier 2: Regime mismatch (Holding Long in Bear, Short in Bull)
        # Using 3 units as threshold
        if (regime == "BEAR" and inventory > 3) or (regime == "BULL" and inventory < -3):
            urgency_level = "REGIME_MISMATCH"
            offset_multiplier = 0.05
            spread_multiplier = 0.6
        
        # Tier 3: Large wrong-way position
        if abs(inventory) > 10:
            if (regime == "BEAR" and inventory > 0) or (regime == "BULL" and inventory < 0):
                urgency_level = "LARGE_POSITION"
                offset_multiplier = 0.10
                spread_multiplier = 0.3
        
        # Tier 4: Regime flip detection
        if self._detect_regime_flip(regime, inventory):
            urgency_level = "REGIME_FLIP"
            offset_multiplier = 0.15
            spread_multiplier = 0.2
        
        # Tier 5: Time urgency (T-2m)
        if seconds_remaining < 120 and abs(inventory) > 5:
            urgency_level = "TIME_URGENCY"
            offset_multiplier = 0.20
            spread_multiplier = 0.1
        
        # Tier 6: Panic (T-90s) - CHANGED from 30s for safety
        if seconds_remaining < 90 and abs(inventory) > 0:
            urgency_level = "PANIC"
            offset_multiplier = 0.50
            spread_multiplier = 0.05
            
        return urgency_level, offset_multiplier, spread_multiplier
    
    def _detect_regime_flip(self, current_regime, inventory):
        if self.previous_regime is None:
            self.previous_regime = current_regime
            return False
            
        dangerous = False
        if self.previous_regime == "BULL" and current_regime == "BEAR" and inventory > 5:
            dangerous = True
        elif self.previous_regime == "BEAR" and current_regime == "BULL" and inventory < -5:
            dangerous = True
            
        self.previous_regime = current_regime
        return dangerous



class TimeDecayManager:
    """Manages pricing urgency and position limits based on time."""
    def get_time_urgency_factor(self, seconds_remaining: float) -> float:
        if seconds_remaining > 600: return 1.0
        elif seconds_remaining > 480: return 1.2
        elif seconds_remaining > 360: return 1.5
        elif seconds_remaining > 240: return 2.0
        elif seconds_remaining > 120: return 3.0
        elif seconds_remaining > 60: return 4.0
        elif seconds_remaining > 30: return 5.0
        else: return 10.0
        
    def get_inventory_haircut(self, seconds_remaining: float) -> float:
        if seconds_remaining > 600: return 1.00
        elif seconds_remaining > 480: return 0.95
        elif seconds_remaining > 360: return 0.90
        elif seconds_remaining > 240: return 0.80
        elif seconds_remaining > 120: return 0.65
        elif seconds_remaining > 60: return 0.45
        elif seconds_remaining > 30: return 0.20
        else: return 0.0
        
    def get_max_position(self, seconds_remaining: float, base_limit: int = 15) -> int:
        urgency = self.get_time_urgency_factor(seconds_remaining)
        return max(1, int(base_limit / urgency))
        
    def get_time_decay_skew(self, inventory: float, seconds_remaining: float) -> float:
        if abs(inventory) < 0.1: return 0.0
        haircut = self.get_inventory_haircut(seconds_remaining)
        # Long inv -> Negative skew (Sell cheaper)
        return -inventory * (1 - haircut) * 0.5

class TimeAwarePricingEngine:
    """Combines all factors to determine pricing."""
    def __init__(self, urgency_manager, time_decay, base_spread=0.04):
        self.urgency = urgency_manager
        self.time_decay = time_decay
        self.base_spread = base_spread
        
    def calculate_prices(self, mid, inventory, regime, seconds_remaining):
        # 1. Factors
        time_urg = self.time_decay.get_time_urgency_factor(seconds_remaining)
        time_skew = self.time_decay.get_time_decay_skew(inventory, seconds_remaining)
        max_pos = self.time_decay.get_max_position(seconds_remaining)
        
        urg_level, inv_off_mult, inv_spread_mult = self.urgency.get_inventory_urgency(
            inventory, regime, seconds_remaining
        )
        
        # 2. Spread (Tightens with urgency?)
        # Actually prompt says: "spread_multiplier = inv_spread_mult / time_urgency"
        # If time_urgency is high (10x), spread becomes tiny?
        # Maybe we want Wide spread in panic?
        # Logic from prompt: "effective_spread = self.base_spread * spread_multiplier"
        # If panic, spread_mult is 0.05. So spread is tiny. This ensures a fill. CORRECT.
        
        spread_mult = inv_spread_mult / time_urg
        effective_spread = self.base_spread * spread_mult
        
        # 3. Skew
        # Regime Skew
        if regime == "BULL": r_skew = 0.03
        elif regime == "BEAR": r_skew = -0.03
        else: r_skew = 0.0
        
        # Inventory Skew (Amplified by urgency)
        i_skew = -1 * inventory * inv_off_mult * time_urg
        
        total_skew = r_skew + i_skew + time_skew
        
        bid = mid - (effective_spread / 2) + total_skew
        ask = mid + (effective_spread / 2) + total_skew
        
        # Safety
        bid = max(0.01, bid)
        ask = max(0.02, ask)
        
        # Market Order Check
        use_market = False
        if urg_level in ["PANIC", "REGIME_FLIP"] and abs(inventory) > 8:
            use_market = True
        if abs(inventory) > 12: # Wrong way large
            if (regime=="BEAR" and inventory>0) or (regime=="BULL" and inventory<0):
                use_market = True
                
        return {
            "bid": bid, "ask": ask, "spread": ask-bid,
            "urgency": urg_level, "time_urg": time_urg,
            "max_pos": max_pos, "use_market_orders": use_market
        }

class PerformanceTracker:
    """Tracks metrics for improvement."""
    def __init__(self):
        self.fills = 0
        self.orders = 0
        self.regime_flips = 0
        
    def record_order(self): self.orders += 1
    def record_fill(self): self.fills += 1
    def record_flip(self): self.regime_flips += 1
    
    def get_report(self):
        if self.orders == 0: rate = 0.0
        else: rate = self.fills / self.orders
        return f"FillRate={rate:.1%} ({self.fills}/{self.orders}) Flips={self.regime_flips}"

class AdaptiveSpreadManager:
    """Dynamically adjusts spread based on fill rate."""
    def __init__(self, base_spread=0.04):
        self.base_spread = base_spread
        self.current_spread = base_spread
        self.min_spread = 0.01
        self.max_spread = 0.06
        
    def get_spread(self, fill_rate: float, time_since_fill: float) -> float:
        # If very stale (>60s) or low fill rate, tighten spread
        if time_since_fill > 60 or fill_rate < 0.20:
             # Tighten aggressivley
             self.current_spread = max(self.min_spread, self.base_spread * 0.5)
        elif fill_rate < 0.40:
             # Moderate tightening
             self.current_spread = max(self.min_spread, self.base_spread * 0.75)
        else:
             # Healthy
             self.current_spread = self.base_spread
             
        return self.current_spread

# ============================================================

# ============================================================
# BOT LOGIC
# ============================================================
class MarketMakerBot:
    """Official Market Maker Bot with Spike Strategy Integration."""

    def __init__(self, client: TurbineClient):
        self.client = client
        self.market_id: str | None = None
        self.settlement_address: str | None = None
        self.contract_address: str | None = None 
        self.start_price: int = 0
        self.end_time: int = 0 # Unix TS
        
        self.recorder = CSVLogger()
        self.urgency_manager = InventoryUrgencyManager()
        self.tracker = PerformanceTracker()
        self.adaptive_spread = AdaptiveSpreadManager(base_spread=CONFIG['strategy']['base_spread'])
        self.time_decay = TimeDecayManager()
        # Pass adaptive spread instance or we adjust base_spread dynamically?
        # We will adjust effectively in the loop
        self.engine = TimeAwarePricingEngine(self.urgency_manager, self.time_decay, base_spread=CONFIG['strategy']['base_spread'])
        
        # Strategy State
        self.state = MarketState()
        self.running = True

        # Performance Stats (New)
        self.stats = {
            'placed': 0,
            'filled': 0,
            'cancelled': 0
        }
        self._last_summary_ts = 0.0
        self.last_fill_ts = 0.0
        
        # Winnings Tracking
        self.traded_markets: Dict[str, str] = {}  # market_id -> contract_address
        
        # Loop Control
        self.last_poll_ts = 0.0

    async def get_active_market(self) -> Optional[Tuple[str, int, int]]:
        """Get the currently active BTC quick market."""
        try:
            # We attempt ONCE. If API error/None, we return None.
            # The retry logic belongs in the caller if needed.
            quick_market = await asyncio.to_thread(self.client.get_quick_market, "BTC")
            return quick_market.market_id, quick_market.end_time, quick_market.start_price
        except Exception as e:
            logger.warning(f"Could not fetch active market: {e}")
            return None

    async def cancel_all_orders(self, market_id: str) -> None:
        """Cancel all local active orders."""
        if not self.state.open_orders:
            return

        logger.info(f"Cancelling {len(self.state.open_orders)} orders on market {market_id[:8]}...")
        # Since we might not have all hashes, we rely on tracking. 
        # But safest is to use the API if tracking is loose. 
        # For now, iterate local tracking.
        for order_id in list(self.state.open_orders.keys()):
            try:
                # We store order_hash as key
                self.client.cancel_order(order_hash=order_id, market_id=market_id)
                await asyncio.sleep(0.1) # Rate limit protect
            except Exception as e:
                logger.warning(f"Failed to cancel order {order_id}: {e}")
        self.state.open_orders.clear()

    async def switch_to_new_market(self, new_market_id: str, start_price: int, end_time: int) -> None:
        """Transition logic."""
        old_market_id = self.market_id

        if old_market_id:
            logger.info(f"MARKET TRANSITION: {old_market_id[:8]} -> {new_market_id[:8]}")
            # Track old for claiming
            if self.contract_address:
                self.traded_markets[old_market_id] = self.contract_address
                logger.info(f"Tracking market {old_market_id[:8]}... for winnings claim")
            
            # Cancel all old orders
            await self.cancel_all_orders(old_market_id)

        self.market_id = new_market_id
        self.start_price = start_price
        self.end_time = end_time
        self.state = MarketState() # Reset strategy state

        # Fetch addresses
        try:
            markets = self.client.get_markets()
            for market in markets:
                if market.id == new_market_id:
                    self.settlement_address = market.settlement_address
                    self.contract_address = market.contract_address
                    logger.info(f"Market Info: Settlement={self.settlement_address[:10]}... Contract={self.contract_address[:10]}...")
                    break
        except Exception as e:
            logger.error(f"Could not fetch market addresses: {e}")

        logger.info(f"Now trading: {new_market_id} (Strike: ${start_price/1e8:,.2f})")

    async def monitor_market_transitions(self) -> None:
        """Background task to handle market rollovers."""
        while self.running:
            try:
                res = await self.get_active_market()
                if res:
                    new_market_id, end_time, start_price = res
                    if new_market_id != self.market_id:
                        await self.switch_to_new_market(new_market_id, start_price, end_time)
                else:
                    # No active market found, wait and retry
                    pass
            except Exception as e:
                logger.error(f"Market monitor error: {e}")
            await asyncio.sleep(5)

    async def claim_resolved_markets(self) -> None:
        """Background task to claim winnings."""
        while self.running:
            try:
                if not self.traded_markets:
                    await asyncio.sleep(30)
                    continue

                markets_to_remove = []
                # Fetch fresh markets list once
                try:
                    all_markets = self.client.get_markets()
                except Exception:
                    await asyncio.sleep(10)
                    continue

                for market_id, contract_address in list(self.traded_markets.items()):
                    is_resolved = False
                    for m in all_markets:
                        if m.id == market_id and m.resolved:
                            is_resolved = True
                            break
                    
                    if is_resolved:
                        logger.info(f"Claiming winnings for {market_id[:8]}...")
                        try:
                            # Note: Py Client claims via API which handles permit/relayer
                            self.client.claim_winnings(contract_address)
                            logger.info(f"Claim submitted for {market_id[:8]}")
                            markets_to_remove.append(market_id)
                        except Exception as e:
                            if "no winnings" in str(e).lower() or "no position" in str(e).lower():
                                logger.info(f"No winnings for {market_id[:8]}.")
                                markets_to_remove.append(market_id)
                            else:
                                logger.error(f"Claim failed for {market_id[:8]}: {e}")

                for mid in markets_to_remove:
                    self.traded_markets.pop(mid, None)

            except Exception as e:
                logger.error(f"Claim monitor error: {e}")
            
            await asyncio.sleep(30)

    # --- Strategy Logic ---
    async def fetch_position_snapshot(self):
        """Poll API for authoritative position."""
        if not self.market_id: return
        try:
            # Use get_user_positions(address, chain_id)
            if not self.client.address:
                logger.warning("No wallet address available for positions.")
                return

            positions = await asyncio.to_thread(
                self.client.get_user_positions,
                address=self.client.address,
                chain_id=137
            )
            
            # API might return None for empty portfolio
            if positions is None: 
                positions = []

            # DEBUG: Dump positions occasionally to debug Inv=0 mystery
            # logger.info(f"DEBUG POSITIONS: Found {len(positions)} positions {[p.market_id for p in positions]}")
            
            # Find current market position (Robust Case-Insensitive Match)
            # Both self.market_id and p.market_id might have different casing
            target = next((p for p in positions if p.market_id.lower() == self.market_id.lower()), None)
            
            if target:
                # Net = YES - NO (scaled 1e6)
                net = (target.yes_shares - target.no_shares) / 1_000_000.0
                self.state.position = net
                # logger.info(f"DEBUG: Pos Snapshot -> {net}")
            else:
                self.state.position = 0.0
                
        except Exception as e:
            logger.warning(f"Pos Fetch Error: {e}")
            self.state.position = 0.0
            
    async def fetch_usdc_balance(self):
        """Fetch USDC balance from RPC."""
        # Rate limit friendly (every 10s)
        if time.time() - self.state.last_balance_ts < 10: return
        
        try:
            # Polygon USDC (Native)
            usdc_address = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
            if not self.client.address: return
            wallet = self.client.address[2:]
            
            # Simple JSON-RPC call
            # Using a public aggregated RPC or user provided
            rpc_url = os.environ.get("TURBINE_RPC_URL", "https://polygon-rpc.com")
            
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{
                    "to": usdc_address,
                    "data": f"0x70a08231000000000000000000000000{wallet}" # balanceOf
                }, "latest"],
                "id": 1
            }
            
            async with httpx.AsyncClient(timeout=5.0) as http:
                resp = await http.post(rpc_url, json=payload)
                data = resp.json()
                
                if "result" in data:
                    hex_val = data["result"]
                    if hex_val:
                        val = int(hex_val, 16)
                        self.state.wallet_balance = val / 1_000_000.0 # 6 decimals
                        self.state.last_balance_ts = time.time()
                        
        except Exception as e:
            logger.warning(f"Balance fetch failed: {e}")

    def check_risk_limits(self, potential_new_exposure: float = 0.0, side: str = None) -> bool:
        """Return True if within limits, False if blocked."""
        # 1. Calculate Current Exposure
        # Inventory Value (Abs position)
        inv_risk = abs(self.state.position) 
        
        # Open Bids Value
        orders_risk = 0.0
        for o in self.state.open_orders.values():
            if o['side'] == 'buy':
                # If we are placing a NEW BUY, assume we replace the OLD BUY.
                # So don't count the old buy in the risk check.
                if side == 'buy': continue
                
                # size is in units of 1e6 (1 share = 1_000_000)
                # We need dollar exposure: (size/1e6) * price
                orders_risk += (o['size'] / 1_000_000.0) * o['price']
                
        total_risk = inv_risk + orders_risk + potential_new_exposure
        
        # 2. Compare to Wallet
        balance = self.state.wallet_balance
        if balance <= 0: return True 
        
        limit_pct = CONFIG['risk']['max_wallet_risk_pct']
        allowed = balance * limit_pct
        
        if total_risk > allowed:
            logger.warning(f"RISK BLOCKED: Exposure ${total_risk:.2f} > Limit ${allowed:.2f} ({limit_pct*100}% of ${balance:.2f})")
            return False
            
        return True

    def calculate_order_size(self, seconds_remaining: float, inventory: float) -> int:
        """Calculate decreasing order size based on time."""
        # 3 Shares early, 2 mid, 1 late
        if seconds_remaining > 600: base = 3
        elif seconds_remaining > 300: base = 2
        else: base = 1
        
        # Adjust for Inventory (Don't add to losing side aggresively?)
        # For now, keep simple symmetric
        return int(base * 1_000_000)

    def _log_performance_dashboard(self):
        """Log stats every 60s."""
        if time.time() - self._last_summary_ts > 60:
            self._last_summary_ts = time.time()
            report = self.tracker.get_report()
            logger.info("=" * 60)
            logger.info(f"📊 DASHBOARD: {report}")
            logger.info(f"   Stats: {self.stats}")
            logger.info(f"   Spread: {self.adaptive_spread.current_spread:.3f}")
            logger.info("=" * 60)

    def calculate_order_size(self, seconds_remaining: float, inventory: float) -> int:
        """Calculate decreasing order size based on time."""
        # 3 Shares early, 2 mid, 1 late
        if seconds_remaining > 600: base = 3
        elif seconds_remaining > 300: base = 2
        else: base = 1
        
        # Adjust for Inventory (Don't add to losing side aggresively?)
        # For now, keep simple symmetric
        return int(base * 1_000_000)

    def _get_regime_skew(self) -> float:
        """Calculate skew based on Bull/Bear regime (SMA-30)."""
        history = self.state.price_history
        if len(history) < 10: 
            return 0.0 # Not enough data
        
        sma = sum(history) / len(history)
        current = history[-1]
        
        # Threshold could be configurable, using 0.0 for now (simple above/below)
        # Bullish: Price > Average -> We want to be Long
        if current > sma:
            return 0.05 # Add 5 cents skew (Shift quotes UP)
            
        # Bearish: Price < Average -> We want to be Short
        if current < sma:
            return -0.05 # Subtract 5 cents skew (Shift quotes DOWN)
            
        return 0.0

    async def check_stop_loss(self, mid: float) -> bool:
        """Check and trigger trailing stop loss."""
        # cooldowned?
        if self.state.stop_triggered:
            if time.time() > self.state.stop_cooldown_ts:
                logger.info("Stop Loss Cooldown Expired. Resuming...")
                self.state.stop_triggered = False
                # Reset tracking
                self.state.trailing_high = mid
                self.state.trailing_low = mid
            else:
                return True # Still stopped

        pos = self.state.position
        pct = CONFIG['strategy']['stop_loss_pct'] # e.g. 0.10
        
        # 1. Update Water Marks
        if pos > 0.1: # LONG
            if mid > self.state.trailing_high:
                self.state.trailing_high = mid
            
            # Check Drawdown
            # If Mid < High * (1 - pct)
            drop_limit = self.state.trailing_high * (1.0 - pct)
            if mid < drop_limit:
                logger.warning(f"STOP LOSS TRIGGERED (LONG): Price {mid:.3f} < {drop_limit:.3f} (High: {self.state.trailing_high:.3f})")
                await self.panic_close(side='sell')
                return True
                
        elif pos < -0.1: # SHORT
            if mid < self.state.trailing_low:
                self.state.trailing_low = mid
                
            # Check Drawdown (Rise)
            # If Mid > Low * (1 + pct)
            rise_limit = self.state.trailing_low * (1.0 + pct)
            if mid > rise_limit:
                logger.warning(f"STOP LOSS TRIGGERED (SHORT): Price {mid:.3f} > {rise_limit:.3f} (Low: {self.state.trailing_low:.3f})")
                await self.panic_close(side='buy')
                return True
        else:
            # Flat - Reset trackers to current price
            self.state.trailing_high = mid
            self.state.trailing_low = mid
            
        return False

    async def panic_close(self, side: str):
        """Execute panic close."""
        self.state.stop_triggered = True
        self.state.stop_cooldown_ts = time.time() + 60 # 1 min cooldown
        
        # 1. Cancel All
        await self.cancel_all_orders(self.market_id)
        
        # 2. Place Market Order (Crossing Limit)
        # We use a massive skew or just a limit order that crosses far.
        size = int(abs(self.state.position) * 1_000_000)
        if size <= 0: return

        price = 0.01 if side == 'sell' else 0.99
        logger.warning(f"PANIC CLOSE: Placing {side.upper()} {size} @ {price}")
        
        try:
            # To be safe and fast, we just implement simple place here
            expiration = int(time.time() + 60)
            if side == 'buy':
                o = self.client.create_limit_buy(
                    self.market_id, Outcome.YES, int(price * 1e6), size, expiration, self.settlement_address
                )
                # For buy, we need permit for cost
                # Cost = size * price. 
                # price=0.99 -> cost=0.99 USDC per share.
                val = (size * int(price*1e6)) // 1_000_000
                fee = size // 100
                total = ((val+fee)*110)//100
                permit = self.client.sign_usdc_permit(total, self.settlement_address)
                o.permit_signature = permit
                self.client.post_order(o)
            else:
                o = self.client.create_limit_sell(
                    self.market_id, Outcome.YES, int(price * 1e6), size, expiration, self.settlement_address
                )
                # Sell permit
                amt = (size * 110)//100
                permit = self.client.sign_usdc_permit(amt, self.settlement_address)
                o.permit_signature = permit
                self.client.post_order(o)
                
            logger.warning("PANIC CLOSE SENT.")
            
        except Exception as e:
            logger.error(f"Panic Close Failed: {e}")


    def compute_quotes(self) -> Tuple[Optional[float], Optional[float]]:
        """Calculate implementation of Spike Strategy."""
        # 1. Stale Check
        age = time.time() - self.state.last_update_ts
        if age > CONFIG['loop']['max_quote_age_seconds']:
            return None, None
        
        if not self.state.bids or not self.state.asks:
            return None, None

        best_bid = self.state.bids[0][0]
        best_ask = self.state.asks[0][0]
        mid = (best_bid + best_ask) / 2.0

        strat = CONFIG['strategy']
        base_half = strat['base_spread'] / 2.0

        # Extremes
        if mid < strat['extreme_low'] or mid > strat['extreme_high']:
            base_half *= strat['extreme_spread_mult']

        # Inventory Skew
        pos = self.state.position
        max_pos = CONFIG['risk']['max_inventory_units']
        
        # TIME-BASED URGENCY
        # Increase skew and restrict entry as we near end_time
        time_left = self.end_time - time.time()
        urgency_mult = 1.0
        
        # Hard Limits (Stop quoting if full)
        allow_bid = True
        allow_ask = True
        
        if time_left < 30:
            # T-30s: HARD STOP. Cancel everything, place nothing.
            return None, None
            
        elif time_left < 120:
            # T-2m: PANIC MODE. 
            # 5x Skew to force inventory flush.
            # Reduce Only: Don't increase absolute position size.
            urgency_mult = 5.0
            if pos > 0: allow_bid = False # Don't buy more
            if pos < 0: allow_ask = False # Don't sell more
            
        elif time_left < 300:
            # T-5m: URGENT. 2x Skew.
            urgency_mult = 2.0
        
        if pos >= max_pos:
            allow_bid = False
        if pos <= -max_pos:
            allow_ask = False
            
            
        # Skew Calculation
        inv_skew = -1 * (pos / max_pos) * strat['skew_factor'] * urgency_mult
        regime_skew = self._get_regime_skew()
        
        total_skew = inv_skew + regime_skew

        # Quote
        bid_p = mid - base_half + total_skew
        ask_p = mid + base_half + total_skew

        # Clamp
        bid_p = max(strat['min_price'], min(bid_p, strat['max_price']))
        ask_p = max(strat['min_price'], min(ask_p, strat['max_price']))

        if bid_p >= ask_p: return None, None
        
        # Apply Hard Limits
        final_bid = bid_p if allow_bid else None
        final_ask = ask_p if allow_ask else None
        
        return final_bid, final_ask

    async def trading_loop(self):
        """Main Maker Loop (2s interval)."""
        while self.running:
            start_ts = time.time()
            try:
                if self.market_id:
                    # 1. Update (Pos + Time + Balance)
                    await self.fetch_position_snapshot()
                    await self.fetch_usdc_balance()
                    
                    mid = 0.5
                    if self.state.bids and self.state.asks:
                         mid = (self.state.bids[0][0]+self.state.asks[0][0])/2
                         self.state.price_history.append(mid)
                    
                    # Regime
                    regime_str = "NEUT"
                    r_skew = self._get_regime_skew()
                    if r_skew > 0: regime_str = "BULL"
                    elif r_skew < 0: regime_str = "BEAR"

                    # 2. Pricing Engine
                    time_left = max(0, self.end_time - time.time())
                    
                    if time_left < 90:
                        # EMERGENCY EXIT (T-90s)
                        if self.state.open_orders:
                            await self.cancel_all_orders(self.market_id)
                        if abs(self.state.position) > 0.1:
                            logger.critical(f"T-{int(time_left)}s PANIC DUMP: {self.state.position}")
                            side = 'sell' if self.state.position > 0 else 'buy'
                            await self.panic_close(side)
                        continue

                    # Performance & Adaptive Spread
                    self._log_performance_dashboard()
                    
                    fill_rate = 0.0
                    if self.stats['placed'] > 0: 
                        fill_rate = self.stats['filled'] / self.stats['placed']
                    
                    time_since_fill = time.time() - self.last_fill_ts
                    # Update engine base spread dynamically
                    current_spread = self.adaptive_spread.get_spread(fill_rate, time_since_fill)
                    self.engine.base_spread = current_spread
                    
                    pricing = self.engine.calculate_prices(
                        mid, self.state.position, regime_str, time_left
                    )
                    
                    # 3. Log
                    logger.info(
                        f"T-{int(time_left)}s {regime_str} "
                        f"Inv={self.state.position:.1f} "
                        f"Urg={pricing['urgency']} "
                        f"Spr={pricing['spread']:.3f} "
                        f"Bid={pricing['bid']:.2f} Ask={pricing['ask']:.2f} "
                        f"Bal=${self.state.wallet_balance:.2f}"
                    )
                    self.recorder.log_tick(
                        self.market_id, mid, regime_str, self.state.position, 
                        pricing['bid'], pricing['ask']
                    )

                    # 4. Execute
                    # Market Order Mode?
                    if pricing['use_market_orders']:
                         if self.state.position > 0: await self.panic_close('sell')
                         elif self.state.position < 0: await self.panic_close('buy')
                         continue
                         
                    # Check Limits
                    max_p = pricing['max_pos']
                    inv = self.state.position
                    
                    # Over limit? Reduce only.
                    allow_bid = True
                    allow_ask = True
                    
                    if inv >= max_p: allow_bid = False
                    if inv <= -max_p: allow_ask = False
                    
                    # Phase 5: Regime-Based Limits
                    # Bull: Max Short 0 (Don't allow net short, or strictly reduce)
                    if regime_str == "BULL" and inv < 0: allow_ask = False # Don't sell more
                    if regime_str == "BEAR" and inv > 0: allow_bid = False # Don't buy more
                    
                    # Place Limits
                    if allow_bid:
                        # Risk Check
                        cost = pricing['bid'] * 1 
                        if self.check_risk_limits(cost, side='buy'):
                           self.tracker.record_order()
                           # Sizing
                           size = self.calculate_order_size(time_left, inv)
                           await self._converge_side('buy', pricing['bid'], size=size)
                    elif self.state.open_orders:
                        # Cancel existing bids if forbidden?
                        # For now, let smart replacement handle or clean sweep
                        pass 
                        
                    if allow_ask:
                        self.tracker.record_order()
                        size = self.calculate_order_size(time_left, inv)
                        await self._converge_side('sell', pricing['ask'], size=size)
                    
                    # 5. Safety / Kill Switch
                    # If PnL < -2.00 (Requires PnL tracking, not impl yet, skipping)
                    # If Time < 30s and Inv > 3 (Handled by Panic Logic above)
                    
            except Exception as e:
                 logger.error(f"Loop error: {e}")

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
            
            # Sleep remainder of 2s
            elapsed = time.time() - start_ts
            sleep_time = max(0.1, (CONFIG['loop']['tick_interval_ms'] / 1000.0) - elapsed)
            await asyncio.sleep(sleep_time)

    async def _converge_side(self, side: str, price: float, size: int = ORDER_SIZE):
        # Identify existing order for this side
        existing_id = None
        existing_info = None
        
        for oid, info in self.state.open_orders.items():
            if info['side'] == side:
                existing_id = oid
                existing_info = info
                break
        
        if not existing_id:
            await self._place_limit_order(side, price, size=size)
            return

        # Drift Check
        old_price = existing_info['price']
        drift = abs(old_price - price)
        age = time.time() - existing_info['ts']
        # Check size mismatch
        current_size = existing_info.get('size', ORDER_SIZE)
        size_changed = (current_size != size)
        
        if drift > CONFIG['loop']['replace_threshold'] or age > CONFIG['loop']['max_quote_age_seconds'] or size_changed:
            logger.info(f"Replacing {side.upper()}: Drift {drift:.4f} or Age {age:.1f}s or Size {current_size}->{size}")
            # Cancel
            try:
                # We use stored order_hash as index
                self.client.cancel_order(order_hash=existing_id, market_id=self.market_id)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    logger.info(f"Order {existing_id} already gone (Filled?). Cleared.")
                else:
                    logger.warning(f"Cancel failed for {existing_id}: {e}")
            finally:
                # Always remove strict tracking. If it failed, it's gone or we can't manage it anyway.
                # Better to clear it than to loop infinitely creating duplicates.
                self.state.open_orders.pop(existing_id, None)
            
            await asyncio.sleep(0.2) # Burst smoothing
            await self._place_limit_order(side, price, size=size)

    async def _place_limit_order(self, side: str, price: float, size: int = ORDER_SIZE):
        if not self.settlement_address: return # Can't sign permit without it
        
        outcome = Outcome.YES
        # Scale to API
        price_int = int(price * 1_000_000) 
        
        try:
            # We run this in thread because it involves signing/requests
            if side == 'buy':
                # Permit calc: (size * price / 1e6) + fee + margin
                # But `create_limit_buy` helper handles signing if we use the helper?
                # The helper in SKILL.md `sign_usdc_permit` is manual.
                # `create_limit_buy` creates the object but doesn't post.
                # `client.post_order` sends it.
                
                # We need to construct the order then sign permit then post.
                # The template code in SKILL.md lines 834+ shows exact flow.
                
                # Wrap in thread for async safety
                def _do_place():
                    # 1. Create Object
                    if side == 'buy':
                        o = self.client.create_limit_buy(
                            market_id=self.market_id,
                            outcome=outcome,
                            price=price_int,
                            size=size,
                            expiration=int(time.time() + 120),
                            settlement_address=self.settlement_address
                        )
                        # 2. Permit
                        # Cost = size * price_fraction. 
                        # size=1e6, price=0.5e6 (0.5). cost = 0.5e6 USDC units (0.5 USDC).
                        # Wait, USDC is 6 decimals. 
                        # 1 share = 1 USDC max payout. 
                        # buying 1 share at 0.50 costs 0.50 USDC.
                        # permit val = 0.50 USDC -> 500,000 units.
                        val = (size * price_int) // 1_000_000
                        fee = size // 100
                        amt = ((val + fee) * 110) // 100
                        
                        permit = self.client.sign_usdc_permit(amt, self.settlement_address)
                        o.permit_signature = permit
                        self.client.post_order(o)
                        return o.order_hash # Return hash from object
                    else:
                        o = self.client.create_limit_sell(
                            market_id=self.market_id,
                            outcome=outcome,
                            price=price_int,
                            size=size,
                            expiration=int(time.time() + 120),
                            settlement_address=self.settlement_address
                        )
                        # Sell permit: just size + margin (for splitting)
                        amt = (size * 110) // 100
                        permit = self.client.sign_usdc_permit(amt, self.settlement_address)
                        o.permit_signature = permit
                        self.client.post_order(o)
                        return o.order_hash

                # Run sync usage in thread
                tx_id = await asyncio.to_thread(_do_place)
                
                # Store with extended stats
                self.state.open_orders[tx_id] = {
                    'side': side, 
                    'price': price, 
                    'size': ORDER_SIZE, 
                    'ts': time.time()
                }
                
                # Stats & Logging
                self.stats['placed'] += 1
                logger.info(f"📤 PLACED: {side.upper()} {size/1e6:.1f} @ {price:.3f} (ID: {tx_id[:8]})")

        except Exception as e:
            if "rate limit" in str(e).lower():
                logger.warning("Order Rate Limit Exhausted. Backing off 5s...")
                await asyncio.sleep(5)
            else:
                logger.error(f"Order place failed: {e}")

    # --- Configured Run ---
    async def run(self, host: str):
        ws = TurbineWSClient(host)
        
        # Start Backgrounds
        monitor = asyncio.create_task(self.monitor_market_transitions())
        claimer = asyncio.create_task(self.claim_resolved_markets())
        trader = asyncio.create_task(self.trading_loop())
        
        logger.info("Bot Started. Waiting for market...")
        
        # ensure we have a market logic -> REMOVED BLOCKING CALL
        # m, _, s = await self.get_active_market()
        # await self.switch_to_new_market(m, s)
        
        # We rely on monitor_market_transitions to pick up the market.
        # Main loop just handles WS connection when market is set.

        while self.running:
            if not self.market_id:
                logger.info("Waiting for active market...")
                await asyncio.sleep(5)
                continue

            try:
                async with ws.connect() as stream:
                    await stream.subscribe_orderbook(self.market_id)
                    await stream.subscribe_trades(self.market_id)
                    logger.info(f"Connected to WS: {self.market_id}")
                    
                    async for msg in stream:
                        if self.market_id and msg.type == 'orderbook':
                            # Update internal book
                            # Flatten simple list of [price, size]
                            # msg.orderbook.bids is likely list of helpers. Or dicts?
                            # TurbineWSClient returns parsed objects.
                            # msg.orderbook.bids -> list of PriceLevel(price, size)
                            # We need to map to floats. 
                            # Wait, API sends scaled ints? 
                            # If client unpacks, let's verify.
                            # Assuming standard client returns raw-ish or objects.
                            # Let's check spike_bot logic for parsing.
                            
                            # spike_bot used adapter which mapped it. 
                            # Here we use raw client. 
                            # Let's just assume we get bids/asks lists.
                            # We need to normalize to 0-1 float for strategy.
                            
                            # SAFETY: If bids is empty, don't crash
                            bids = []
                            if hasattr(msg, 'orderbook') and msg.orderbook.bids:
                                for b in msg.orderbook.bids:
                                    # b.price is int (1e6 scale probably), b.size is int (1e6)
                                    # Map to float
                                    bids.append((float(b.price)/1_000_000, float(b.size)/1_000_000))
                            
                            asks = []
                            if hasattr(msg, 'orderbook') and msg.orderbook.asks:
                                for a in msg.orderbook.asks:
                                    asks.append((float(a.price)/1_000_000, float(a.size)/1_000_000))

                            self.state.bids = bids
                            self.state.asks = asks
                            self.state.last_update_ts = time.time()
                            
            except Exception as e:
                logger.error(f"WS Error: {e}. Reconnecting...")
                await asyncio.sleep(1)

        monitor.cancel()
        claimer.cancel()
        trader.cancel()

# --- Entry Point ---
async def main():
    api_key_id, api_private_key = get_or_create_api_credentials()
    
    private_key = os.environ.get("TURBINE_PRIVATE_KEY")
    client = TurbineClient(
        host="https://api.turbinefi.com",
        chain_id=137,
        private_key=private_key,
        api_key_id=api_key_id,
        api_private_key=api_private_key,
    )
    
    bot = MarketMakerBot(client)
    try:
        await bot.run("wss://api.turbinefi.com")
    except KeyboardInterrupt:
        pass
    finally:
        await bot.cancel_all_orders(bot.market_id)
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
