"""
Turbine Market Maker Bot - Official Spike Variant
Generated based on Turbine Best Practices

Algorithm: Inventory-Aware Market Maker with Risk Controls
"""

import asyncio
import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
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
        "base_spread": 0.40,      # Wide spread for safety (Makers earn premium)
        "skew_factor": 0.01,      # Inventory skew
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
    },
    "risk": {
        "max_inventory_units": 1000,
        "max_portfolio_exposure": 10000.0,
    },
    "loop": {
        "tick_interval_ms": 2000,       # 2s interval (Rate Limit Safe)
        "max_quote_age_seconds": 10.0,  # 10s max age (Stale Data Protection)
        "replace_threshold": 0.001,
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
        
        # Strategy State
        self.state = MarketState()
        self.running = True
        
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
                # We store order_hash as key
                self.client.cancel_order(order_hash=order_id, market_id=market_id)
                await asyncio.sleep(0.1) # Rate limit protect
                await asyncio.sleep(0.1) # Rate limit protect
            except Exception as e:
                logger.warning(f"Failed to cancel order {order_id}: {e}")
        self.state.open_orders.clear()

    async def switch_to_new_market(self, new_market_id: str, start_price: int = 0) -> None:
        """Switch liquidity to a new market."""
        old_market_id = self.market_id

        # Track for winnings
        if old_market_id and self.contract_address:
            self.traded_markets[old_market_id] = self.contract_address
            logger.info(f"Tracking market {old_market_id[:8]}... for winnings claim")

        if old_market_id:
            logger.info(f"MARKET TRANSITION: {old_market_id[:8]} -> {new_market_id[:8]}")
            await self.cancel_all_orders(old_market_id)

        self.market_id = new_market_id
        self.start_price = start_price
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
                        await self.switch_to_new_market(new_market_id, start_price)
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
            
            # Find current market position
            target = next((p for p in positions if p.market_id == self.market_id), None)
            
            if target:
                # Net = YES - NO (scaled 1e6)
                net = (target.yes_shares - target.no_shares) / 1_000_000.0
                self.state.position = net
                # logger.info(f"DEBUG: Pos Snapshot -> {net}")
            else:
                self.state.position = 0.0
                
        except Exception as e:
            # Silence NoneType iteration error if it happens
            if "NoneType" in str(e): return
            logger.error(f"Pos polling error: {e}")

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
        skew = -1 * (pos / max_pos) * strat['skew_factor']

        # Quote
        bid_p = mid - base_half + skew
        ask_p = mid + base_half + skew

        # Clamp
        bid_p = max(strat['min_price'], min(bid_p, strat['max_price']))
        ask_p = max(strat['min_price'], min(ask_p, strat['max_price']))

        if bid_p >= ask_p: return None, None
        return bid_p, ask_p

    async def trading_loop(self):
        """Main Maker Loop (2s interval)."""
        while self.running:
            start_ts = time.time()
            try:
                if self.market_id:
                    # 1. Update Position
                    await self.fetch_position_snapshot()
                    
                    # 2. Compute
                    bid, ask = self.compute_quotes()
                    
                    # 3. Log Heartbeat
                    mid_str = "?"
                    if self.state.bids and self.state.asks:
                        mid_str = f"{(self.state.bids[0][0]+self.state.asks[0][0])/2:.3f}"
                    # logger.info(f"Tick: Mid={mid_str} Inv={self.state.position} Bid={bid} Ask={ask}")

                    # 4. Reconcile/Place
                    if bid and ask:
                        # Converge Bids
                        await self._converge_side('buy', bid)
                        await asyncio.sleep(0.5) # Burst smoothing
                        # Converge Asks
                        await self._converge_side('sell', ask)
                    else:
                        # Stale or invalid -> Cancel all
                        if self.state.open_orders:
                            await self.cancel_all_orders(self.market_id)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
            
            # Sleep remainder of 2s
            elapsed = time.time() - start_ts
            sleep_time = max(0.1, (CONFIG['loop']['tick_interval_ms'] / 1000.0) - elapsed)
            await asyncio.sleep(sleep_time)

    async def _converge_side(self, side: str, price: float):
        # Identify existing order for this side
        existing_id = None
        existing_info = None
        
        for oid, info in self.state.open_orders.items():
            if info['side'] == side:
                existing_id = oid
                existing_info = info
                break
        
        if not existing_id:
            await self._place_limit_order(side, price)
            return

        # Drift Check
        old_price = existing_info['price']
        drift = abs(old_price - price)
        age = time.time() - existing_info['ts']
        
        if drift > CONFIG['loop']['replace_threshold'] or age > CONFIG['loop']['max_quote_age_seconds']:
            logger.info(f"Replacing {side.upper()}: Drift {drift:.4f} or Age {age:.1f}s")
            # Cancel
            try:
                # We use stored order_hash as index
                self.client.cancel_order(order_hash=existing_id, market_id=self.market_id)
            except Exception as e:
                logger.warning(f"Cancel failed for {existing_id}: {e}")
            finally:
                # Always remove strict tracking. If it failed, it's gone or we can't manage it anyway.
                # Better to clear it than to loop infinitely creating duplicates.
                self.state.open_orders.pop(existing_id, None)
            
            await asyncio.sleep(0.2) # Burst smoothing
            await self._place_limit_order(side, price)

    async def _place_limit_order(self, side: str, price: float):
        if not self.settlement_address: return # Can't sign permit without it
        
        outcome = Outcome.YES
        # Scale to API
        price_int = int(price * 1_000_000) # Wait, API uses 1e6 for price too? 
        # SKILL.md says: "price: Price scaled by 1e6"
        # spike_bot used 10000? Let me check spike_bot logic.
        # TurbineAdapter used int(order.price * 10000). 
        # But create_limit_buy docs say 1e6. 
        # WAIT. 10000 * 100 = 1e6. 
        # If price is 0.50 (50 cents) -> 500,000.
        # spike_bot used 10k? 0.5 * 10k = 5000. That's 0.005?
        # Let's check `spike_bot` logic one more time.
        # Ah, Adapter said `int(order.price * 10000)`.
        # If order.price is 0.50, result 5000.
        # If API expects 1e6 for 1.0, then 0.50 should be 500,000.
        # 5000 is 0.5%.
        # Did spike_bot have a massive pricing bug?
        # Or does Turbine use 10k scale?
        # SKILL.md says: `price: Price scaled by 1e6`.
        # `quick_market.start_price` is 8 decimals (BTC).
        # Outcome tokens are binary 0-1.
        # Usually binary options use 0-1 scale.
        # If I use 1e6, 0.50 -> 500,000.
        # I will use 1e6 scaling to be safe per docs.

        size = ORDER_SIZE # 1e6 (1 unit)
        
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
                
                # Store
                self.state.open_orders[tx_id] = {'side': side, 'price': price, 'ts': time.time()}
                logger.info(f"Placed {side.upper()} @ {price:.3f} (ID: {tx_id})")

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
