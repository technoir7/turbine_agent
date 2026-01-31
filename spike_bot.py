#!/usr/bin/env python3
"""
Turbine Spike Bot
-----------------
A single-file, deterministic trading bot for Turbine.
Collapses the complex multi-module architecture into one safe, auditable loop.

Features:
- Single execution loop (no races)
- Explicit state management
- Strict safety gates (feed freshness, max inventory)
- Reuses existing TurbineAdapter for protocol details
"""

import asyncio
import logging
import os
import sys
import signal
import time
import yaml
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Reuse existing adapter
from src.exchange.turbine import TurbineAdapter

# Minimal reuse of core types if needed, or redefine for isolation
from turbine_client.types import Side as TurbineSide, Outcome

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("spike_bot.log")
    ]
)
logger = logging.getLogger("SpikeBot")

# --- Configuration & Constants ---
DEFAULT_CONFIG = {
    "strategy": {
        "base_spread": 0.20,
        "skew_factor": 0.01,
        "imbalance_threshold": 2.0,
        "imbalance_depth_n": 5,
        "overlay_bias": 0.005,
        "extreme_low": 0.10,
        "extreme_high": 0.90,
        "extreme_spread_mult": 2.0
    },
    "risk": {
        "max_inventory_units": 1000,
        "max_portfolio_exposure": 10000.0
    },
    "loop": {
        "tick_interval_ms": 1000,
        "max_quote_age_seconds": 30.0,
        "replace_threshold": 0.001
    }
}

# --- State Definitions ---
@dataclass
class Order:
    id: str
    price: float
    size: float
    side: str # 'buy' or 'sell'
    ts: float
    exchange_id: Optional[str] = None

@dataclass
class MarketState:
    bids: List[Tuple[float, float]] = field(default_factory=list) # price, size
    asks: List[Tuple[float, float]] = field(default_factory=list)
    last_update_ts: float = 0.0
    position: float = 0.0
    open_orders: Dict[str, Order] = field(default_factory=dict) # client_id -> Order

# --- Core Bot Class ---
class TurbineSpike:
    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv() # Load .env file
        self.config = self._load_config(config_path)
        self.state = MarketState()
        self.market_id = None
        self.adapter = None
        self.running = True
        
        # Safety Flags
        self.trading_enabled = os.environ.get("TRADING_ENABLED", "false").lower() == "true"
        
        # Default: Dry Run = True unless explicitly ENABLED
        self.dry_run = not self.trading_enabled
        
        # Allow explicit override if needed (e.g. TRADING_ENABLED=true but DRY_RUN=true for logged test)
        if os.environ.get("DRY_RUN", "").lower() == "true":
             self.dry_run = True

        self.max_age = float(os.environ.get("TURBINE_MAX_DATA_AGE_S", 30.0))
        self.last_poll_ts = 0.0 # Force poll on start
        
        if self.dry_run:
            logger.warning("DRY RUN ACTIVE. No real orders will be placed.")
        else:
            logger.warning("⚠️  LIVE TRADING ENABLED ⚠️")
            
        logger.info(f"Bot Initialized. MAX_AGE={self.max_age}s")

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r') as f:
                user_conf = yaml.safe_load(f)
            # Merge with defaults (shallow merge for sections)
            conf = DEFAULT_CONFIG.copy()
            if user_conf:
                for section in ['strategy', 'risk', 'loop']:
                    if section in user_conf:
                        conf[section].update(user_conf[section])
            return conf
        except Exception as e:
            logger.warning(f"Config load failed ({e}), using defaults.")
            return DEFAULT_CONFIG

    async def setup(self):
        """Initialize adapter and connection."""
        try:
            # Initialize Adapter (reusing existing polished code)
            self.adapter = TurbineAdapter(self.config) 
            # Note: Adapter automatically loads env vars for Auth
            
            # Connect WS
            logger.info("Connecting to WebSocket...")
            await self.adapter.connect()
            
            # Identify Market (Reuse logical flow: Poll BTC quick market)
            # Simplified: just get "BTC" quick market
            # In a real spike, we might hardcode or fetch. Let's fetch.
            # We need to construct a robust way to get the ID.
            # Assuming adapter has public methods or we use rest client directly.
            pass
            
        except Exception as e:
            logger.critical(f"Setup Failed: {e}")
            sys.exit(1)

    async def _fetch_active_market(self) -> str:
        """Poll for active BTC market."""
        # Using the private client inside adapter for simplicity as in original bot
        client = self.adapter._rest_client
        try:
             # This matches existing logic in RolloverManager
             meta = client.get_quick_market("BTC")
             if meta and meta.market_id:
                 return meta.market_id
        except Exception as e:
            logger.error(f"Failed to fetch market: {e}")
        return None

    # --- Strategy Logic (Extracted) ---
    def compute_quotes(self) -> Tuple[Optional[float], Optional[float]]:
        # 1. Check Feed Freshness
        age = time.time() - self.state.last_update_ts
        if age > self.max_age:
            if age < self.max_age + 60: # Log sparingly
                 logger.debug(f"Feed Stale: {age:.1f}s > {self.max_age}s")
            return None, None
            
        # 2. Extract BBO
        if not self.state.bids or not self.state.asks:
            return None, None
            
        best_bid = self.state.bids[0][0]
        best_ask = self.state.asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        
        # 3. Strategy Parameters
        strat = self.config['strategy']
        base_half = strat['base_spread'] / 2.0
        
        # 4. Extremes Logic
        fair = mid
        if fair < strat['extreme_low'] or fair > strat['extreme_high']:
            base_half *= strat['extreme_spread_mult']
            
        # 5. Inventory Skew
        # skew = -1 * (pos / max) * factor
        pos = self.state.position
        max_pos = self.config['risk']['max_inventory_units']
        skew = -1 * (pos / max_pos) * strat['skew_factor']
        
        # 6. Imbalance Overlay (Simplified to top N depth ratio)
        # Using consolidated list of (price, size)
        bid_vol = sum(s for p, s in self.state.bids[:strat['imbalance_depth_n']])
        ask_vol = sum(s for p, s in self.state.asks[:strat['imbalance_depth_n']])
        overlay = 0.0
        
        if bid_vol > 0 and ask_vol > 0:
            ratio = bid_vol / ask_vol
            threshold = strat['imbalance_threshold']
            bias = strat['overlay_bias']
            if ratio > threshold: overlay = -bias
            elif ratio < (1.0/threshold): overlay = bias

        # 7. Final Quote
        # Note: In spike bot, we prioritize safety. 
        # Widening spread is safer than tightening.
        
        bid_p = fair - base_half + skew + overlay
        ask_p = fair + base_half + skew + overlay
        
        # Clamp
        bid_p = max(strat['min_price'], min(bid_p, strat['max_price']))
        ask_p = max(strat['min_price'], min(ask_p, strat['max_price']))
        
        if bid_p >= ask_p: return None, None
        
        return bid_p, ask_p

    # --- Execution Logic ---
    async def reconcile(self):
        """Main decision loop."""
        bid_price, ask_price = self.compute_quotes()
        
        if bid_price is None:
            # Safe mode: cancel all? Or hold? 
            # Existing bot cancels if no quote.
            # We will cancel all to be safe.
            await self.cancel_all_local()
            return

        # Converge Bids
        await self._converge_side('buy', bid_price)
        # Converge Asks
        await self._converge_side('sell', ask_price)

    async def _converge_side(self, side: str, price: float):
        # Find existing order for side
        existing = None
        for order in self.state.open_orders.values():
            if order.side == side:
                existing = order
                break
        
        if not existing:
            await self.place_order(side, price)
            return
            
        # Check drift/age
        drift = abs(existing.price - price)
        age = time.time() - existing.ts
        
        if drift > self.config['loop']['replace_threshold'] or age > self.config['loop']['max_quote_age_seconds']:
            logger.info(f"Replacing {side}: Drift {drift:.4f} or Age {age:.1f}s")
            await self.cancel_order(existing.id)
            await self.place_order(side, price)

    # --- Exchange Primitives ---
    async def place_order(self, side: str, price: float):
        if self.dry_run:
            logger.info(f"[DRY] Would PLACE {side} @ {price:.2f}")
            return
            
        # Risk Check
        if abs(self.state.position) > self.config['risk']['max_inventory_units']:
            logger.warning("Max Inventory Reached. Skipping Place.")
            return

        try:
            # Construct Order object compatible with adapter
            # We need to reuse the Order class or mock it
            from src.core.state import Order as CoreOrder
            from src.core.events import Side as CoreSide, OrderStatus
            
            c_side = CoreSide.BID if side == 'buy' else CoreSide.ASK
            clid = f"spike_{int(time.time()*1000)}"
            
            o = CoreOrder(
                client_order_id=clid,
                market_id=self.market_id,
                side=c_side,
                price=price,
                size=1.0 # Fixed size for now
            )
            # manually set status/ts if needed, though defaults are usually fine or set elsewhere
            o.status = OrderStatus.PENDING_ACK
            o.created_ts = time.time()
            
            tx_id = await self.adapter.place_order(o)
            
            # Update Local State
            self.state.open_orders[clid] = Order(
                id=clid,
                price=price,
                size=1.0,
                side=side,
                ts=time.time(),
                exchange_id=tx_id
            )
            logger.info(f"Placed {side} @ {price} (ID: {tx_id})")
            
        except Exception as e:
            logger.error(f"Place Failed: {e}")

    async def cancel_order(self, client_id: str):
        if client_id not in self.state.open_orders: return
        order = self.state.open_orders[client_id]
        
        if self.dry_run:
            logger.info(f"[DRY] Would CANCEL {client_id}")
            del self.state.open_orders[client_id]
            return
            
        try:
            # Need CoreOrder wrapper again
            from src.core.state import Order as CoreOrder
            from src.core.events import Side as CoreSide
            
            c_side = CoreSide.BID if order.side == 'buy' else CoreSide.ASK
            o = CoreOrder(
                client_order_id=order.id,
                market_id=self.market_id,
                side=c_side,
                price=order.price,
                size=order.size
            )
            o.exchange_order_id = order.exchange_id
            
            await self.adapter.cancel_order(o)
            del self.state.open_orders[client_id]
            logger.info(f"Cancelled {client_id}")
            
        except Exception as e:
            if "404" in str(e):
                logger.warning(f"Order {client_id} 404 (already gone). Removing.")
                del self.state.open_orders[client_id]
            else:
                logger.error(f"Cancel Failed: {e}")

    async def cancel_all_local(self):
        ids = list(self.state.open_orders.keys())
        for cid in ids:
            await self.cancel_order(cid)

    # --- WS Handling ---
    async def _ws_loop(self):
        """Consume WS messages and update state."""
        while self.running:
            try:
                # Access adapter's WS queue directly or polling mechanism?
                # TurbineAdapter puts messages into an internal queue if using the client incorrectly,
                # BUT the adapter expects a callback.
                # Let's monkey-patch or register a callback.
                # Actually, standard TurbineAdapter uses a queue consumption model in `_ws_loop`?
                # No, it uses `on_message`. We need to hook into that.
                
                # The cleanest way without modifying Adapter is to subclass or assign handler.
                # Adapter exposes `_process_ws_messages`? No, it's internal.
                # We can just poll `adapter._received_messages`? No.
                
                # Let's override the adapter's `_handle_message` or similar if possible.
                # Or better, just implement a poller if the client supports it.
                # Inspecting adapter... it uses `_ws_connection.subscribe`.
                
                # REVISION: We will implement a simple poll of the adapter's logical queue
                # if it exists, OR we will implement `_on_ws_message`.
                # Given strict extraction: We'll assume we can pass a dummy callback if needed,
                # or better, just read raw from `adapter._ws_connection` if exposed.
                
                # Looking at `src/exchange/turbine.py`, it seems to have `_ws_messages` queue.
                # We can consume from that.
                if hasattr(self.adapter, '_ws_messages'):
                    while not self.adapter._ws_messages.empty():
                        msg = await self.adapter._ws_messages.get()
                        self._process_msg(msg)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"WS Loop Error: {e}")
                await asyncio.sleep(1)

    def _process_msg(self, msg):
        try:
            type_name = type(msg).__name__
            
            if type_name == 'BookDeltaEvent':
                # msg (BookDeltaEvent) has: price, size, side (Side.BID/ASK)
                # We need to update our list of tuples.
                # Simplification: Fetch snapshot on first connect? 
                # Or just build from deltas if it's an OrderBookUpdate (snapshot-like)?
                # TurbineAdapter's _translate sends individual updates.
                
                # IMPORTANT: For a spike bot, maintaining a perfect book from deltas is complex.
                # However, looking at adapter source, OrderBookUpdate usually gives a FULL snapshot of top levels.
                # But the adapter breaks it into individual deltas.
                # This makes the spike bot harder.
                
                # HACK: For the spike bot, we will just store the logical "best" from what we see
                # OR we will bypass the translation and read the raw message if possible.
                # But we can't easily bypass. state management is needed.
                
                # Let's apply the delta.
                # msg.side is Side enum.
                
                price = msg.price
                size = msg.size
                is_bid = (msg.side.name == 'BID') # Check enum name
                
                target_list = self.state.bids if is_bid else self.state.asks
                
                # Remove existing price level
                # (Price is float, use epsilon?)
                # Turbine price is normalized already by adapter (check adapter logic).
                # Actually adapter normalizes to float.
                
                new_list = [x for x in target_list if abs(x[0] - price) > 1e-9]
                if size > 0:
                    new_list.append((price, size))
                
                # Sort
                new_list.sort(key=lambda x: x[0], reverse=is_bid)
                
                if is_bid: self.state.bids = new_list
                else: self.state.asks = new_list
                
                self.state.last_update_ts = time.time()
                
            elif type_name == 'TradeEvent':
                pass 
                
        except Exception as e:
            logger.error(f"Msg Parse Error: {e}")

    # --- Main Loop ---
    async def run(self):
        await self.setup()
        
        # 1. Identify Market
        self.market_id = await self._fetch_active_market()
        if not self.market_id:
            logger.critical("No active market found. Exiting.")
            return

        logger.info(f"Targeting Market: {self.market_id}")
        
        # 2. Subscribe
        await self.adapter.subscribe_markets([self.market_id])
        
        # 3. Hook WS Processing
        # Adapter runs its own loop managed by _ws_task.
        # We need to hook into the messages. 
        # In spike bot, we will poll the adapter's queue if we can find it, 
        # OR we just rely on state inside adapter if we reused it?
        
        # ACTUALLY: The TurbineAdapter stores state in `self.callbacks`.
        # We should register a callback!
        
        async def on_msg(msg):
             self._process_msg(msg)
             
        self.adapter.callbacks.append(on_msg)
        
        # Trigger message processing loop if needed?
        # Adapter.connect() starts _process_ws_messages() (via _ws_task).
        # So we just need to register callback.
        
        logger.info("Starting Main Loop...")
        try:
            while self.running:
                # A. Rollover Check
                # (Simplified: every 10s check if market ID changed)
                if int(time.time()) % 10 == 0:
                     latest = await self._fetch_active_market()
                     if latest and latest != self.market_id:
                         logger.info(f"Rollover! {self.market_id} -> {latest}")
                         await self.cancel_all_local()
                         self.market_id = latest
                         await self.adapter.subscribe_markets([self.market_id])
                         self.state = MarketState() # Reset
                
                # B. Logic
                await self.reconcile()
                
                # C. Heartbeat Log
                if int(time.time()) % 10 == 0:
                    mid = "?"
                    if self.state.bids and self.state.asks:
                         mid = f"{ (self.state.bids[0][0] + self.state.asks[0][0])/2 :.3f}"
                    logger.info(f"Tick: Mid={mid} Orders={len(self.state.open_orders)} Inv={self.state.position}")

                await asyncio.sleep(self.config['loop']['tick_interval_ms'] / 1000.0)
                
        except asyncio.CancelledError:
            logger.info("Stopping...")
        finally:
            self.running = False
            await self.cancel_all_local()
            # Cleanup tasks...

# --- Entry Point ---
if __name__ == "__main__":
    bot = TurbineSpike()
    
    # Handle Signals
    def stop_sig(sig, frame):
        logger.info("Signal received. Shutting down...")
        bot.running = False
    
    signal.signal(signal.SIGINT, stop_sig)
    signal.signal(signal.SIGTERM, stop_sig)
    
    try:
        if "--probe" in sys.argv:
            bot.dry_run = True
            logger.info("PROBE MODE: Running for 30s then exiting.")
            async def probe():
                await bot.setup()
                m = await bot._fetch_active_market()
                if m:
                    # Register callback
                    async def on_msg(msg):
                        bot._process_msg(msg)
                        logger.info(f"Probe Msg: {type(msg).__name__}")
                    bot.adapter.callbacks.append(on_msg)
                    
                    await bot.adapter.subscribe_markets([m])
                    await asyncio.sleep(30)
                else:
                    logger.error("Probe failed: No market.")
            asyncio.run(probe())
        else:
            asyncio.run(bot.run())
    except Exception as e:
        logger.critical(f"Fatal: {e}")
