import asyncio
import logging
import signal
import time
from typing import Optional

from .config.loader import load_config
from .core.state import StateStore, OrderBook
from .core.events import BookDeltaEvent, TradeEvent, OrderAckEvent, UserFillEvent, OrderStatus, HeartbeatEvent
from .exchange.interface import ExchangeAdapter
from .exchange.turbine import TurbineAdapter
from .exchange.simulated import SimulatedExchange
from .strategy.engine import StrategyEngine
from .strategy.rollover import RolloverManager
from .risk.engine import RiskEngine
from .execution.engine import ExecutionEngine

logger = logging.getLogger(__name__)

class Supervisor:
    def __init__(self, config_path: str, simulated: bool = False):
        self.config = load_config(config_path)
        self.simulated = simulated
        self.running = False
        self.market_id = "MARKET_A" # Will be overridden by rollover in live mode
        
        # Components
        self.state = StateStore()
        self.adapter: Optional[ExchangeAdapter] = None
        self.strategy: Optional[StrategyEngine] = None
        self.risk: Optional[RiskEngine] = None
        self.execution: Optional[ExecutionEngine] = None
        self.rollover: Optional[RolloverManager] = None

    async def start(self):
        """Startup sequence."""
        logger.info("Supervisor: Starting up...")
        
        # 1. Init Adapter
        if self.simulated:
            logger.info("Mode: SIMULATED EXCHANGE")
            self.adapter = SimulatedExchange([self.market_id])
        else:
            logger.info("Mode: TURBINE LIVE")
            self.adapter = TurbineAdapter(self.config)
            # Init rollover manager for live mode
            if self.config.get('rollover', {}).get('enabled', True):
                self.rollover = RolloverManager(self.adapter, self.config)
            
        # 2. Register Callbacks
        self.adapter.register_callback(self.on_event)
        
        # 3. Connect
        await self.adapter.connect()
        
        # 4. Init Logic Engines
        self.risk = RiskEngine(self.config, self.state)
        self.strategy = StrategyEngine(self.config, self.state)
        self.execution = ExecutionEngine(self.config, self.adapter, self.state, self.risk, self.strategy)
        
        # 5. Start rollover manager and get initial market (live mode)
        if self.rollover:
            await self.rollover.start()
            # Wait briefly for first poll
            await asyncio.sleep(1.0)
            initial_market = self.rollover.get_current_market()
            if initial_market:
                self.market_id = initial_market
                logger.info(f"Supervisor: Using BTC Quick Market: {self.market_id}")
            else:
                logger.warning("Supervisor: Rollover manager did not detect market yet, using default")
        
        # 6. Subscribe to market
        await self.adapter.subscribe_markets([self.market_id])
        
        # 7. Initial Reconcile / Snapshot
        # In real impl, fetch REST snapshot here and populate StateStore
        
        # 8. Start Loop
        self.running = True
        await self.loop()

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Supervisor: Shutting down...")
        self.running = False
        
        # Stop rollover manager
        if self.rollover:
            await self.rollover.stop()
        
        if self.adapter:
            # Cancel all on exit for safety
            logger.info("Supervisor: Cancelling all orders...")
            await self.adapter.cancel_all()
            await self.adapter.close()

    async def loop(self):
        """Main trading loop."""
        tick_interval = self.config['loop']['tick_interval_ms'] / 1000.0
        heartbeat_interval = 15.0  # Log WS feed health every 15 seconds
        last_heartbeat = time.time()
        
        while self.running:
            start_time = time.time()
            
            try:
                # 1. Check for market rollover (live mode only)
                if self.rollover and self.rollover.has_changed(self.market_id):
                    new_market = self.rollover.get_current_market()
                    logger.warning(f"ROLLOVER DETECTED: {self.market_id} -> {new_market}")
                    
                    # Cancel all orders on old market
                    await self.adapter.cancel_all(market_id=self.market_id)
                    
                    # Reset local order state for old market
                    old_orders = [oid for oid, o in self.state.orders.items() 
                                 if o.market_id == self.market_id]
                    for oid in old_orders:
                        del self.state.orders[oid]
                    
                    # Switch to new market
                    self.market_id = new_market
                    
                    # Subscribe to new market
                    await self.adapter.subscribe_markets([self.market_id])
                    
                    logger.info(f"ROLLOVER COMPLETE: Now trading {self.market_id}")
                
                # 2. Periodic WebSocket feed health check
                if time.time() - last_heartbeat >= heartbeat_interval:
                    ws_last_ts = getattr(self.adapter, '_ws_last_message_ts', None)
                    ws_msg_count = getattr(self.adapter, '_ws_message_count', 0)
                    if ws_last_ts:
                        age = time.time() - ws_last_ts
                        logger.info(f"WS Feed: {ws_msg_count} msgs, last msg {age:.1f}s ago")
                    else:
                        logger.warning("WS Feed: No messages received yet")
                    last_heartbeat = time.time()
                
                # 3. State Maintenance (pruning, etc)
                # 4. Risk Checks (Periodic)
                
                # 5. Reconcile
                if hasattr(self, 'execution'):
                    await self.execution.reconcile(self.market_id)
            
            except Exception as e:
                logger.error("Supervisor: Loop Error: %s", e, exc_info=True)
                
            # Sleep remainder
            elapsed = time.time() - start_time
            sleep_time = max(0.01, tick_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def on_event(self, event):
        """Handle incoming events from Adapter."""
        try:
            if isinstance(event, BookDeltaEvent):
                book = self.state.get_orderbook(event.market_id)
                book.apply_delta(event.seq, event.side, event.price, event.size)
                
            elif isinstance(event, UserFillEvent):
                logger.info(f"Fill: {event.side} {event.size} @ {event.price}")
                self.state.on_fill(event.market_id, event.size, event.price, event.side, event.fee)
                # Update Order Status
                if event.client_order_id in self.state.orders:
                     o = self.state.orders[event.client_order_id]
                     o.filled_size += event.size
                     if o.filled_size >= o.size:
                         o.status = OrderStatus.FILLED
                     else:
                         o.status = OrderStatus.PARTIAL
                         
            elif isinstance(event, OrderAckEvent):
                if event.client_order_id in self.state.orders:
                    o = self.state.orders[event.client_order_id]
                    o.status = event.status
                    o.exchange_order_id = event.exchange_order_id
                    
            elif isinstance(event, HeartbeatEvent):
                self.risk.check_connectivity(event.ts)
                
        except Exception as e:
            logger.error("Event Handler Error: %s", e, exc_info=True)
