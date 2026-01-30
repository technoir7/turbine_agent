"""Turbine exchange adapter implementation using turbine-py-client."""
import logging
import asyncio
import os
from typing import List, Optional, Dict, Any, Callable, Awaitable
from .interface import ExchangeAdapter
from ..core.events import Side, OrderStatus
from ..core.state import Order

logger = logging.getLogger(__name__)

# Verified constants from turbine-py-client examples
DEFAULT_HOST = "https://api.turbinefi.com"
DEFAULT_CHAIN_ID = 137  # Polygon mainnet


class TurbineAdapter(ExchangeAdapter):
    """
    Turbine adapter implementation using turbine-py-client.
    
    Read-only operations work without authentication.
    Trading operations require environment variables:
        - TURBINE_PRIVATE_KEY
        - TURBINE_API_KEY_ID
        - TURBINE_API_PRIVATE_KEY
    """
    
    def __init__(self, config: Dict):
        """Initialize the Turbine adapter.
        
        Args:
            config: Configuration dictionary with exchange settings.
        """
        self.config = config
        self.callbacks: List[Callable] = []
        
        # Load configuration
        self._host = config.get('exchange', {}).get('base_url', DEFAULT_HOST)
        self._chain_id = config.get('exchange', {}).get('chain_id', DEFAULT_CHAIN_ID)
        
        # Load credentials from environment
        # Try TURBINE_* first, then INTEGRATION_* as fallback (from examples)
        self._private_key = os.environ.get('TURBINE_PRIVATE_KEY') or \
                           os.environ.get('INTEGRATION_WALLET_PRIVATE_KEY')
        self._api_key_id = os.environ.get('TURBINE_API_KEY_ID') or \
                          os.environ.get('INTEGRATION_API_KEY_ID')
        self._api_private_key = os.environ.get('TURBINE_API_PRIVATE_KEY') or \
                               os.environ.get('INTEGRATION_API_PRIVATE_KEY')
        
        # Auto-register API credentials if we have private key but not API keys
        # Per SKILL.md Step 3: Auto-registration on first run
        if self._private_key and (not self._api_key_id or not self._api_private_key):
            self._auto_register_api_credentials()
        
        # Initialize REST client
        try:
            from turbine_client import TurbineClient
            
            # If we have all auth credentials, create authenticated client
            if self._private_key and self._api_key_id and self._api_private_key:
                self._rest_client = TurbineClient(
                    host=self._host,
                    chain_id=self._chain_id,
                    private_key=self._private_key,
                    api_key_id=self._api_key_id,
                    api_private_key=self._api_private_key,
                )
                logger.info("TurbineAdapter: Initialized with authentication")
            else:
                # Read-only client
                self._rest_client = TurbineClient(
                    host=self._host,
                    chain_id=self._chain_id,
                )
                logger.info("TurbineAdapter: Initialized in read-only mode")
                logger.warning(
                    "Trading disabled: Set TURBINE_PRIVATE_KEY, TURBINE_API_KEY_ID, "
                    "and TURBINE_API_PRIVATE_KEY to enable trading"
                )
        except ImportError as e:
            logger.error(f"Failed to import turbine_client: {e}")
            raise
        
        # WebSocket client (lazy initialization)
        self._ws_client = None
        self._ws_context = None
        self._ws_connection = None
        self._ws_task = None
        self._ws_message_count = 0  # For debug logging
        self._ws_last_message_ts = None  # For heartbeat monitoring
        self._active_subscriptions: Set[str] = set()
        self._watchdog_task = None
        
        # Market cache for settlement addresses
        self._market_cache: Dict[str, Any] = {}
    
    def _auto_register_api_credentials(self):
        """Auto-register API credentials if only private key is set.
        
        Per SKILL.md Step 3: The bot should automatically register for API 
        credentials on first run and save them to the .env file.
        """
        import re
        from pathlib import Path
        
        logger.info("TurbineAdapter: Auto-registering API credentials...")
        
        try:
            from turbine_client import TurbineClient
            
            credentials = TurbineClient.request_api_credentials(
                host=self._host,
                private_key=self._private_key,
            )
            
            self._api_key_id = credentials["api_key_id"]
            self._api_private_key = credentials["api_private_key"]
            
            # Update environment so we can use them immediately
            os.environ["TURBINE_API_KEY_ID"] = self._api_key_id
            os.environ["TURBINE_API_PRIVATE_KEY"] = self._api_private_key
            
            logger.info(f"TurbineAdapter: API credentials registered (key_id: {self._api_key_id[:8]}...)")
            
            # Save to .env file
            self._save_credentials_to_env()
            
        except Exception as e:
            logger.error(f"TurbineAdapter: Failed to auto-register API credentials: {e}")
            logger.warning("Trading will be disabled. You can manually set credentials in .env")
    
    def _save_credentials_to_env(self):
        """Save API credentials to .env file for future runs."""
        import re
        from pathlib import Path
        
        # Find .env file (look in current directory first, then project root)
        env_path = Path(".env")
        if not env_path.exists():
            env_path = Path(__file__).parent.parent.parent / ".env"
        
        if not env_path.exists():
            logger.warning("TurbineAdapter: No .env file found, cannot save credentials")
            return
        
        try:
            content = env_path.read_text()
            
            # Update TURBINE_API_KEY_ID
            if "TURBINE_API_KEY_ID=" in content:
                content = re.sub(
                    r'^TURBINE_API_KEY_ID=.*$',
                    f'TURBINE_API_KEY_ID={self._api_key_id}',
                    content,
                    flags=re.MULTILINE
                )
            else:
                content = content.rstrip() + f"\nTURBINE_API_KEY_ID={self._api_key_id}"
            
            # Update TURBINE_API_PRIVATE_KEY
            if "TURBINE_API_PRIVATE_KEY=" in content:
                content = re.sub(
                    r'^TURBINE_API_PRIVATE_KEY=.*$',
                    f'TURBINE_API_PRIVATE_KEY={self._api_private_key}',
                    content,
                    flags=re.MULTILINE
                )
            else:
                content = content.rstrip() + f"\nTURBINE_API_PRIVATE_KEY={self._api_private_key}"
            
            env_path.write_text(content + "\n")
            logger.info(f"TurbineAdapter: API credentials saved to {env_path}")
            
        except Exception as e:
            logger.error(f"TurbineAdapter: Failed to save credentials to .env: {e}")

    def _require_auth(self):
        """Raise error if trading credentials are not configured."""
        if not (self._private_key and self._api_key_id and self._api_private_key):
            raise NotImplementedError(
                "Trading requires authentication. Set these environment variables:\n"
                "  - TURBINE_PRIVATE_KEY\n"
                "  - TURBINE_API_KEY_ID\n"
                "  - TURBINE_API_PRIVATE_KEY\n"
            )

    async def connect(self):
        """Connect to WebSocket and start listening for updates."""
        try:
            from turbine_client import TurbineWSClient
            import websockets
            from contextlib import asynccontextmanager
            from typing import AsyncIterator
            from turbine_client.ws.client import WSStream

            # Define Robust Client with Keepalive (Monkeypatch/Subclass strategy)
            class RobustTurbineWSClient(TurbineWSClient):
                """Subclass to inject keepalive settings into websockets.connect."""
                
                @asynccontextmanager
                async def connect(self) -> AsyncIterator[WSStream]:
                    """Connect with ping_interval=20, ping_timeout=20."""
                    try:
                        # 20s ping interval, 20s timeout (total 40s tolerance)
                        self._connection = await websockets.connect(
                            self.url, 
                            ping_interval=20, 
                            ping_timeout=20
                        )
                        stream = WSStream(self._connection)
                        yield stream
                    finally:
                        if self._connection:
                            await self._connection.close()
                            self._connection = None

            # Use ws:// variant of the host
            ws_url = self.config.get('exchange', {}).get('ws_url', self._host)
            
            # Use our robust client
            self._ws_client = RobustTurbineWSClient(host=ws_url)
            
            # Log the ACTUAL URL that will be used (after wss:// conversion + /api/v1/stream)
            logger.info(f"TurbineAdapter: Connecting to WebSocket at {self._ws_client.url}")
            
            # Per official websocket_stream.py example (line 40):
            # Use async with to properly manage the connection context
            # Store the context manager so we can clean it up later
            self._ws_context = self._ws_client.connect()
            self._ws_connection = await self._ws_context.__aenter__()
            
            # Start background task to process WS messages
            self._ws_task = asyncio.create_task(self._process_ws_messages())
            
            # Start watchdog task if not already running
            if not self._watchdog_task or self._watchdog_task.done():
                self._watchdog_task = asyncio.create_task(self._watchdog_loop())
            
            logger.info("TurbineAdapter: WebSocket connected")
        except Exception as e:
            logger.error(f"TurbineAdapter: Failed to connect WebSocket: {e}")
            raise

    async def _process_ws_messages(self):
        """Background task to process incoming WebSocket messages."""
        import time
        from ..core.events import BookDeltaEvent, TradeEvent, Side as InternalSide
        
        try:
            async for message in self._ws_connection:
                self._ws_message_count += 1
                self._ws_last_message_ts = time.time()
                
                # Debug: Log messages if TURBINE_WS_DEBUG is set
                if os.environ.get("TURBINE_WS_DEBUG"):
                    msg_str = str(message)[:500]
                    logger.info(f"TurbineAdapter: WS message #{self._ws_message_count}: {msg_str}")
                
                # Log message type for all messages at DEBUG level
                msg_type = getattr(message, 'type', 'unknown')
                market_id = getattr(message, 'market_id', 'unknown')
                display_id = str(market_id) if market_id else 'None'
                logger.debug(f"TurbineAdapter: WS {msg_type} for {display_id[:16]}...")
                
                # TRANSLATION LAYER: Convert WSMessage to Internal Events
                internal_events = self._translate_to_internal_events(message)
                
                # Dispatch to registered callbacks
                for event in internal_events:
                    for callback in self.callbacks:
                        try:
                            await callback(event)
                        except Exception as e:
                            logger.error(f"Callback error processing {event.__class__.__name__}: {e}", exc_info=True)
                        
        except asyncio.CancelledError:
            logger.info("WebSocket message processor cancelled")
        except Exception as e:
            # Extract close code/reason if available
            close_info = ""
            if hasattr(e, 'code') and hasattr(e, 'reason'):
                close_info = f" (close code: {e.code}, reason: {e.reason})"
            logger.error(f"WebSocket message processor error: {e}{close_info}", exc_info=True)

    def _translate_to_internal_events(self, message) -> list:
        """Translate raw WSMessage to internal events (BookDelta, Trade)."""
        from ..core.events import BookDeltaEvent, TradeEvent, Side as InternalSide
        import time
        
        events = []
        msg_type = getattr(message, 'type', '')
        data = getattr(message, 'data', {})
        market_id = getattr(message, 'market_id', None)
        
        if not market_id or not data:
            return events

        # Scale factor for Turbine (6 decimals usually, check docs carefully or config)
        # Using 1e6 as standard for this exchange based on previous knowledge
        SCALE = 1_000_000.0

        if msg_type == 'orderbook':
            # Data structure: {'bids': [{'price': int, 'size': int}], 'asks': [...], 'lastUpdate': int}
            bids_data = data.get('bids', [])
            asks_data = data.get('asks', [])
            last_update = data.get('lastUpdate', int(time.time() * 1000))
            
            # Debug payload sizes - REMOVED
            # if os.environ.get("TURBINE_WS_DEBUG"):
            #      import logging
            #      logger = logging.getLogger(__name__)
            #      logger.info(f"Translating OrderBook: {market_id[:8]} bids={len(bids_data)} asks={len(asks_data)}")

            # Map Bids (Side 0 -> InternalSide.BID)
            for bid in bids_data:
                events.append(BookDeltaEvent(
                    seq=last_update,  # Using timestamp as substitute for sequence
                    market_id=market_id,
                    side=InternalSide.BID,
                    price=float(bid['price']) / SCALE,
                    size=float(bid['size']) / SCALE
                ))
                
            # Map Asks (Side 1 -> InternalSide.ASK)
            for ask in asks_data:
                events.append(BookDeltaEvent(
                    seq=last_update,
                    market_id=market_id,
                    side=InternalSide.ASK,
                    price=float(ask['price']) / SCALE,
                    size=float(ask['size']) / SCALE
                ))
                
        elif msg_type == 'trade':
            # Data structure: {'price': int, 'size': int, 'side': int, 'timestamp': int, ...}
            # Turbine Side: 0=BUY (Aggr Buyer -> Maker Seller?), 1=SELL
            # TradeEvent needs aggressor_side.
            # Usually 'side' in trade msg indicates the aggressor side.
            raw_side = data.get('side')
            aggressor = InternalSide.BID if raw_side == 0 else InternalSide.ASK
            
            events.append(TradeEvent(
                ts=float(data.get('timestamp', 0)) / 1000.0,
                market_id=market_id,
                price=float(data.get('price', 0)) / SCALE,
                size=float(data.get('size', 0)) / SCALE,
                aggressor_side=aggressor
            ))
            
        return events

    async def _watchdog_loop(self):
        """Monitor WebSocket health and reconnect if stalled."""
        import time
        logger.info("TurbineAdapter: Watchdog started")
        
        # Default 60s per user request.
        stall_threshold = float(os.environ.get("TURBINE_WS_STALL_SECONDS", "60.0"))
        logger.info(f"TurbineAdapter: Watchdog threshold set to {stall_threshold}s")
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if not self._ws_connection:
                    continue
                    
                limit = stall_threshold
                if self._ws_last_message_ts and (time.time() - self._ws_last_message_ts > limit):
                    logger.warning(f"TurbineAdapter: WS stalled (last msg {time.time() - self._ws_last_message_ts:.1f}s ago). Reconnecting...")
                    await self._reconnect()
                    
            except asyncio.CancelledError:
                logger.info("TurbineAdapter: Watchdog cancelled")
                break
            except Exception as e:
                logger.error(f"TurbineAdapter: Watchdog error: {e}")
                await asyncio.sleep(5)
                
    async def _reconnect(self):
        """Reconnect to WebSocket and resubscribe."""
        import random
        
        # 1. Close existing connection (internal cleanup)
        # We don't call self.close() because that cancels the watchdog (us!)
        logger.info("TurbineAdapter: Reconnecting...")
        
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
                
        if self._ws_context and self._ws_connection:
            try:
                await self._ws_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing stalled connection: {e}")
        
        self._ws_client = None
        self._ws_context = None
        self._ws_connection = None
        
        # 2. Backoff before reconnecting
        await asyncio.sleep(1 + random.random())
        
        # 3. Connect again
        try:
            # Re-use logic from connect(), using RobustTurbineWSClient
            from turbine_client import TurbineWSClient
            import websockets
            from contextlib import asynccontextmanager
            from typing import AsyncIterator
            from turbine_client.ws.client import WSStream

            class RobustTurbineWSClient(TurbineWSClient):
                @asynccontextmanager
                async def connect(self) -> AsyncIterator[WSStream]:
                    try:
                        self._connection = await websockets.connect(
                            self.url, 
                            ping_interval=20, 
                            ping_timeout=20
                        )
                        stream = WSStream(self._connection)
                        yield stream
                    finally:
                        if self._connection:
                            await self._connection.close()
                            self._connection = None

            ws_url = self.config.get('exchange', {}).get('ws_url', self._host)
            self._ws_client = RobustTurbineWSClient(host=ws_url)
            
            self._ws_context = self._ws_client.connect()
            self._ws_connection = await self._ws_context.__aenter__()
            self._ws_task = asyncio.create_task(self._process_ws_messages())
            
            # Reset stats
            import time
            self._ws_last_message_ts = time.time() # Reset timer immediately to avoid loop
            
            logger.info("TurbineAdapter: Reconnected")
            
            # 4. Resubscribe
            if self._active_subscriptions:
                logger.info(f"TurbineAdapter: Resubscribing to {len(self._active_subscriptions)} markets...")
                for market_id in self._active_subscriptions:
                    await self._ws_connection.subscribe(market_id)
                logger.info("TurbineAdapter: Resubscribed")
                
        except Exception as e:
            logger.error(f"TurbineAdapter: Reconnect failed: {e}")
            # Watchdog will try again next loop
            
    async def close(self):
        """Clean shutdown."""
        logger.info("TurbineAdapter: Closing connections")
        
        # Cancel Watchdog
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        
        # Cancel WS task
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        
        # Close WS connection via context manager
        if self._ws_context and self._ws_connection:
            try:
                await self._ws_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing WS connection: {e}")
        
        # Close REST client
        if self._rest_client:
            self._rest_client.close()

    async def subscribe_markets(self, market_ids: List[str]):
        """Subscribe to market data topics per turbine-py-client patterns.
        
        Args:
            market_ids: List of market IDs to subscribe to.
        """
        if not self._ws_connection:
            raise RuntimeError("WebSocket not connected. Call connect() first.")
        
        logger.info(f"TurbineAdapter: Subscribing to {len(market_ids)} markets")
        
        for market_id in market_ids:
            try:
                # subscribe() sends {"type": "subscribe", "marketId": market_id}
                # This subscribes to ALL updates: orderbook, trades, order_cancelled
                # subscribe_orderbook() and subscribe_trades() are just aliases that call subscribe()
                # Calling both would subscribe twice to the same market, causing WS close
                await self._ws_connection.subscribe(market_id)
                self._active_subscriptions.add(market_id)  # Track for watchdog reconnect
                logger.debug(f"Subscribed to {market_id[:16]}...")
            except Exception as e:
                # Extract close code/reason if it's a WebSocket exception
                close_info = ""
                if hasattr(e, 'code') and hasattr(e, 'reason'):
                    close_info = f" (close code: {e.code}, reason: {e.reason})"
                logger.error(f"Failed to subscribe to market {market_id}: {e}{close_info}")

    async def place_order(self, order: Order) -> str:
        """Place order via REST API with USDC permit for gasless execution.
        
        Args:
            order: Order to place.
            
        Returns:
            Exchange order ID (order_hash).
            
        Raises:
            NotImplementedError: If authentication is not configured.
        """
        self._require_auth()
        
        try:
            from turbine_client.types import Outcome, Side as TurbineSide
            import time
            
            # Get market settlement address (required for order signing)
            market = await self._get_market(order.market_id)
            settlement_address = market.settlement_address
            
            # Map our Side enum to turbine Side
            turbine_side = TurbineSide.BUY if order.side == Side.BID else TurbineSide.SELL
            
            # For now, assume YES outcome (TODO: derive from order or config)
            turbine_outcome = Outcome.YES
            
            # Convert to turbine scale
            turbine_price = int(order.price * 10000)  # Price: 0-1 -> 0-1000000 (but we use 10k scale for compat)
            turbine_size = int(order.size * 1_000_000)  # Size: shares -> 6 decimals
            
            # Create and sign the order
            if turbine_side == TurbineSide.BUY:
                signed_order = self._rest_client.create_limit_buy(
                    market_id=order.market_id,
                    outcome=turbine_outcome,
                    price=turbine_price,
                    size=turbine_size,
                    settlement_address=settlement_address,
                    expiration=int(time.time()) + 3600,  # 1 hour expiration
                )
                
                # USDC permit for BUY: (size * price / 1e6) + fee + 10% margin
                # Per SKILL.md line 850-852
                buyer_cost = (turbine_size * turbine_price) // 1_000_000
                fee = turbine_size // 100  # ~1% fee estimate
                permit_amount = ((buyer_cost + fee) * 110) // 100  # 10% safety margin
                
            else:
                signed_order = self._rest_client.create_limit_sell(
                    market_id=order.market_id,
                    outcome=turbine_outcome,
                    price=turbine_price,
                    size=turbine_size,
                    settlement_address=settlement_address,
                    expiration=int(time.time()) + 3600,
                )
                
                # USDC permit for SELL: size + 10% margin
                # Per SKILL.md line 874
                permit_amount = (turbine_size * 110) // 100
            
            # Sign USDC permit for gasless execution (per SKILL.md line 827-890)
            # This is REQUIRED - orders without permits will fail
            try:
                permit = self._rest_client.sign_usdc_permit(
                    value=permit_amount,
                    settlement_address=settlement_address,
                )
                signed_order.permit_signature = permit
                logger.debug(f"Attached USDC permit: {permit_amount} units")
            except Exception as permit_err:
                logger.warning(f"Failed to sign USDC permit: {permit_err}. Order may fail.")
            
            # Submit the order
            result = self._rest_client.post_order(signed_order)
            order_hash = result.get('orderHash', signed_order.order_hash)
            
            logger.info(f"Order placed: {order_hash}")
            return order_hash
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    async def cancel_order(self, order: Order):
        """Cancel specific order.
        
        Args:
            order: Order to cancel (must have exchange_order_id set).
        """
        self._require_auth()
        
        try:
            from turbine_client.types import Side as TurbineSide
            
            turbine_side = TurbineSide.BUY if order.side == Side.BID else TurbineSide.SELL
            
            result = self._rest_client.cancel_order(
                order_hash=order.exchange_order_id,
                market_id=order.market_id,
                side=turbine_side,
            )
            
            logger.info(f"Order cancelled: {order.exchange_order_id}")
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise

    async def cancel_all(self, market_id: Optional[str] = None):
        """Cancel all open orders (safety switch).
        
        Args:
            market_id: Optional market ID to restrict cancellation to.
        """
        self._require_auth()
        
        logger.info(f"TurbineAdapter: cancel_all triggered for market {market_id}")
        
        try:
            from turbine_client.types import Side as TurbineSide
            
            # Fetch open orders
            open_orders = self._rest_client.get_orders(
                trader=self._rest_client.address,
                market_id=market_id,
                status="open",
            )
            
            logger.info(f"Cancelling {len(open_orders)} open orders")
            
            for turbine_order in open_orders:
                try:
                    side = TurbineSide.BUY if turbine_order.side == 0 else TurbineSide.SELL
                    self._rest_client.cancel_order(
                        order_hash=turbine_order.order_hash,
                        market_id=turbine_order.market_id,
                        side=side,
                    )
                    logger.info(f"Cancelled order: {turbine_order.order_hash}")
                except Exception as e:
                    logger.error(f"Failed to cancel order {turbine_order.order_hash}: {e}")
                    
        except Exception as e:
            logger.error(f"cancel_all failed: {e}")
            raise

    async def get_positions(self) -> Dict[str, float]:
        """Fetch current positions snapshot.
        
        Returns:
            Dictionary mapping market_id to net position (positive = long).
        """
        if not self._private_key:
            logger.warning("get_positions: No auth configured, returning empty")
            return {}
        
        try:
            positions = self._rest_client.get_user_positions(
                address=self._rest_client.address,
                chain_id=self._chain_id,
            )
            
            # Convert to dict: market_id -> net_position
            result = {}
            for pos in positions:
                # net = yes_shares - no_shares (in contract units, 6 decimals)
                net_shares = (pos.yes_shares - pos.no_shares) / 1_000_000
                result[pos.market_id] = net_shares
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return {}

    async def get_open_orders(self) -> List[Order]:
        """Fetch open orders snapshot.
        
        Returns:
            List of Order objects representing open orders.
        """
        if not self._private_key:
            logger.warning("get_open_orders: No auth configured, returning empty")
            return []
        
        try:
            turbine_orders = self._rest_client.get_orders(
                trader=self._rest_client.address,
                status="open",
            )
            
            # Convert turbine Order objects to our Order objects
            orders = []
            for to in turbine_orders:
                our_side = Side.BID if to.side == 0 else Side.ASK
                orders.append(Order(
                    order_id=to.order_hash,  # Use order_hash as ID
                    market_id=to.market_id,
                    side=our_side,
                    price=to.price / 10000,  # Convert from turbine scale
                    size=to.remaining_size / 1_000_000,  # Convert from 6 decimals
                    exchange_order_id=to.order_hash,
                ))
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    def register_callback(self, callback: Callable[[Any], Awaitable[None]]):
        """Register generic event callback for incoming WS messages.
        
        Args:
            callback: Async function to call with each WebSocket message.
        """
        self.callbacks.append(callback)

    def get_quick_market(self, asset: str = "BTC"):
        """Get active quick market for rollover support.
        
        Args:
            asset: Asset symbol ("BTC" or "ETH").
            
        Returns:
            QuickMarket object with market_id, start_price (strike, 8 decimals), end_time.
        """
        try:
            return self._rest_client.get_quick_market(asset)
        except Exception as e:
            logger.error(f"Failed to get quick market for {asset}: {e}")
            raise
    
    async def _get_market(self, market_id: str) -> Any:
        """Get market details (cached).
        
        Args:
            market_id: Market ID to fetch.
            
        Returns:
            Market object from turbine_client.
        """
        if market_id not in self._market_cache:
            # Fetch all markets and cache them
            markets = self._rest_client.get_markets()
            for m in markets:
                self._market_cache[m.id] = m
        
        if market_id not in self._market_cache:
            raise ValueError(f"Market {market_id} not found")
        
        return self._market_cache[market_id]
