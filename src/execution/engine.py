import logging
import asyncio
import time
import uuid
import os
from typing import Dict, List
from ..core.state import StateStore, Order
from ..core.events import Side, OrderStatus
from ..exchange.interface import ExchangeAdapter
from ..risk.engine import RiskEngine
from ..strategy.engine import StrategyEngine

logger = logging.getLogger(__name__)

class ExecutionEngine:
    """
    Diff engine: Reconciles Desired vs Actual.
    """
    def __init__(self, config: Dict, adapter: ExchangeAdapter, 
                 state: StateStore, risk: RiskEngine, strategy: StrategyEngine):
        self.config = config['loop']
        self.adapter = adapter
        self.state = state
        self.risk = risk
        self.strategy = strategy
        self.strategy = strategy
        self.order_size = 1.0 # Default fixed size for this demo
        
        # Safety Config
        # Safety Config
        self.max_data_age = float(os.environ.get("TURBINE_MAX_DATA_AGE_S", 30.0))
        self.last_stale_log_ts = 0.0
        self.last_resync_ts = 0.0

    def is_trading_allowed(self) -> bool:
        """Centralized gate for trading actions."""
        try:
             is_fresh = self.adapter.is_feed_fresh(self.max_data_age)
        except Exception as e:
             # Fail closed if check errors
             now = time.time()
             if now - self.last_stale_log_ts > 10.0:
                  logger.error(f"Exec: Feed freshness check failed: {e}. Assuming stale.")
                  self.last_stale_log_ts = now
             return False

        if not is_fresh:
             now = time.time()
             if now - self.last_stale_log_ts > 10.0:
                  age = self.adapter.get_last_message_age()
                  logger.warning(f"Exec: Stale feed (age {age:.1f}s > {self.max_data_age}s). Skipping trading actions.")
                  self.last_stale_log_ts = now
             return False
        
        return True

    async def reconcile(self, market_id: str):
        # 0. Safety: Check Data Freshness
        if not self.is_trading_allowed():
            return

        # 1. Get Desired
        bid_p, ask_p = self.strategy.get_desired_quotes(market_id)
        
        # 2. Get Actual Open Orders
        # We look at local state which should be kept in sync via events
        open_orders = [o for o in self.state.orders.values() 
                      if o.market_id == market_id and o.status in (OrderStatus.OPEN, OrderStatus.PENDING_ACK)]
        
        existing_bid = next((o for o in open_orders if o.side == Side.BID), None)
        existing_ask = next((o for o in open_orders if o.side == Side.ASK), None)
        
        # 3. Converge Bid
        await self._converge_side(market_id, Side.BID, existing_bid, bid_p)
        
        # 4. Converge Ask
        await self._converge_side(market_id, Side.ASK, existing_ask, ask_p)

    async def _converge_side(self, market_id: str, side: Side, 
                             existing: Order, desired_price: float):
        
        # Gate: Do not cancel/replace if stale
        if not self.is_trading_allowed():
             return

        # Condition A: We want no order, but have one -> Cancel
        if desired_price is None:
            if existing:
                logger.info("Exec: Cancelling %s (No desired quote)", existing.client_order_id)
                await self.adapter.cancel_order(existing)
            return

        # Condition B: We have no order, want one -> Place
        if not existing:
            await self._place_new(market_id, side, desired_price)
            return

        # Condition C: Have order, check drift
        drift = abs(existing.price - desired_price)
        threshold = self.config['replace_threshold']
        
        age = time.time() - existing.created_ts
        max_age = self.config['max_quote_age_seconds']
        
        if drift > threshold or age > max_age:
            logger.info("Exec: Replacing %s (Drift %.4f or Age %.1f)", existing.client_order_id, drift, age)
            # Cancel then Place (simple reshape)
            # Ideally atomic replace if supported, but here separate
            try:
                await self.adapter.cancel_order(existing)
            except Exception as e:
                # Handle 404 gracefully (order already gone)
                if "404" in str(e):
                    logger.warning(f"Exec: Order {existing.client_order_id} 404, removing from state")
                    if existing.client_order_id in self.state.orders:
                        del self.state.orders[existing.client_order_id]
                    # Trigger resync to be safe
                    await self._trigger_resync()
                else:
                    logger.error(f"Exec: Cancel failed: {e}")
            
    async def _trigger_resync(self):
        """Trigger state resynchronization (rate limited)."""
        now = time.time()
        if now - self.last_resync_ts < 10.0:
            return
        self.last_resync_ts = now
        logger.info("Exec: Triggering state resync on 404 error...")
        await self._resync_state()

    async def _resync_state(self):
        """Fetch open orders and reconcile local state."""
        try:
            remote_orders = await self.adapter.get_open_orders()
            remote_map = {o.exchange_order_id: o for o in remote_orders if o.exchange_order_id}
            
            # 1. Remove locals that are gone from remote (and have exchange ID)
            to_remove = []
            for clid, order in self.state.orders.items():
                if order.status == OrderStatus.OPEN and order.exchange_order_id:
                     if order.exchange_order_id not in remote_map:
                         to_remove.append(clid)
            
            for clid in to_remove:
                logger.info(f"Exec: Resync removing stale order {clid}")
                del self.state.orders[clid]
                
            # 2. Add remotes that are missing locally
            # Map local orders by exchange ID for check
            local_exch_map = {o.exchange_order_id: o for o in self.state.orders.values() if o.exchange_order_id}
            
            for exch_id, r_order in remote_map.items():
                if exch_id not in local_exch_map:
                    # Adopt it
                    logger.info(f"Exec: Resync adopting order {exch_id}")
                    self.state.orders[r_order.client_order_id] = r_order
                    
        except Exception as e:
            logger.error(f"Exec: Resync failed: {e}")

    async def _place_new(self, market_id: str, side: Side, price: float):
        # Gate: Do not place if stale
        if not self.is_trading_allowed():
             return

        # Create Order Object
        clid = f"oid_{uuid.uuid4().hex[:8]}"
        order = Order(
            client_order_id=clid,
            market_id=market_id,
            side=side,
            price=price,
            size=self.order_size
        )
        
        # Risk Check
        if not self.risk.check_order_risk(order):
            return 
            
        # Place
        logger.info("Exec: Placing %s %s @ %.2f", side, market_id, price)
        # Update local state tentatively
        self.state.orders[clid] = order
        
        try:
             tx_id = await self.adapter.place_order(order)
             if tx_id:
                 order.exchange_order_id = tx_id
                 logger.info(f"Exec: Order placed with ID {tx_id}")
        except Exception as e:
             logger.error("Exec: Place failed: %s", e)
             # Remove from local state if failed immediate
             if clid in self.state.orders:
                 del self.state.orders[clid]
