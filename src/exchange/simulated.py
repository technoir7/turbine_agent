import asyncio
import logging
import random
import time
from typing import List, Optional, Dict, Any, Callable, Awaitable
from .interface import ExchangeAdapter
from ..core.events import Side, OrderStatus, BookDeltaEvent, OrderAckEvent, UserFillEvent
from ..core.state import Order
from ..core.events import HeartbeatEvent

logger = logging.getLogger(__name__)

class SimulatedExchange(ExchangeAdapter):
    """
    Simulates a CLOB exchange for integration testing.
    Generates random walk prices and matching.
    """
    
    def __init__(self, markets: List[str]):
        self.markets = markets
        self.callbacks: List[Callable] = []
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, float] = {m: 0.0 for m in markets}
        self.running = False
        
        # Simulation state
        self.prices = {m: 0.50 for m in markets}
        self.seq = 0
    
    async def connect(self):
        self.running = True
        asyncio.create_task(self._simulation_loop())
        logger.info("SimulatedExchange: Connected")

    async def close(self):
        self.running = False
        logger.info("SimulatedExchange: Closed")

    async def subscribe_markets(self, market_ids: List[str]):
        logger.info("SimulatedExchange: Subscribed to %s", market_ids)

    async def place_order(self, order: Order) -> str:
        ex_id = f"sim_oid_{int(time.time()*1000)}_{random.randint(0,999)}"
        order.exchange_order_id = ex_id
        order.status = OrderStatus.OPEN
        self.orders[ex_id] = order
        
        # Immediate ack
        ack = OrderAckEvent(
            ts=time.time(),
            client_order_id=order.client_order_id,
            status=OrderStatus.OPEN,
            exchange_order_id=ex_id
        )
        await self._emit(ack)
        
        # Check instant match (simplified)
        await self._check_match(order)
        return ex_id

    async def cancel_order(self, order: Order):
        if order.exchange_order_id in self.orders:
            del self.orders[order.exchange_order_id]
            ack = OrderAckEvent(
                ts=time.time(),
                client_order_id=order.client_order_id,
                status=OrderStatus.CANCELLED
            )
            await self._emit(ack)

    async def cancel_all(self, market_id: Optional[str] = None):
        ids = list(self.orders.keys())
        for oid in ids:
            o = self.orders[oid]
            if market_id and o.market_id != market_id:
                continue
            await self.cancel_order(o)

    async def get_positions(self) -> Dict[str, float]:
        return self.positions.copy()

    async def get_open_orders(self) -> List[Order]:
        return list(self.orders.values())

    def register_callback(self, callback: Callable[[Any], Awaitable[None]]):
        self.callbacks.append(callback)

    async def _emit(self, event):
        for cb in self.callbacks:
            await cb(event)

    async def _simulation_loop(self):
        while self.running:
            for m in self.markets:
                # Random walk price
                move = random.choice([-0.01, 0.01, 0.0])
                self.prices[m] += move
                self.prices[m] = max(0.02, min(0.98, self.prices[m])) # Clamp
                self.seq += 1
                
                # Emit book update (mid price move)
                mid = self.prices[m]
                
                # Update Best Bid (Spread 0.04 in sim)
                await self._emit(BookDeltaEvent(self.seq, m, Side.BID, mid-0.02, 100))
                
                self.seq += 1
                # Update Best Ask
                await self._emit(BookDeltaEvent(self.seq, m, Side.ASK, mid+0.02, 100))
                
                # Heartbeat every 10 updates
                if self.seq % 10 == 0:
                    await self._emit(HeartbeatEvent(time.time(), self.seq))
                
            await asyncio.sleep(0.5)

    async def _check_match(self, order: Order):
        # Simplified matching against current 'mid' +/- spread
        mid = self.prices[order.market_id]
        should_fill = False
        
        if order.side == Side.BID and order.price >= mid + 0.02: # Crossing ask
            should_fill = True
        elif order.side == Side.ASK and order.price <= mid - 0.02: # Crossing bid
            should_fill = True
            
        if should_fill:
            fill = UserFillEvent(
                ts=time.time(),
                market_id=order.market_id,
                client_order_id=order.client_order_id,
                exchange_order_id=order.exchange_order_id,
                side=order.side,
                price=order.price,
                size=order.size
            )
            # Update position
            if order.side == Side.BID:
                self.positions[order.market_id] += order.size
            else:
                self.positions[order.market_id] -= order.size
                
            # Remove from book
            if order.exchange_order_id in self.orders:
                del self.orders[order.exchange_order_id]
                
            await self._emit(fill)
