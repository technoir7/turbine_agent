"""Turbine exchange adapter implementation using turbine-py-client."""
import logging
import asyncio
import os
import time
from typing import List, Optional, Dict, Any, Callable, Awaitable
from .interface import ExchangeAdapter
from ..core.events import Side, OrderStatus
from ..core.state import Order

logger = logging.getLogger(__name__)

# Verified constants from turbine-py-client examples
DEFAULT_CHAIN_ID = 137  # Polygon mainnet
DEFAULT_HOST = "https://api.turbinefi.com"

# Module-level imports for testability
try:
    from turbine_client import TurbineClient, TurbineWSClient
    from turbine_client.ws.client import WSStream
    import websockets
    from contextlib import asynccontextmanager
    from typing import AsyncIterator

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

except ImportError:
    # Allow module load even if dependencies missing (will fail at runtime)
    TurbineClient = None
    TurbineWSClient = object  # dummy base
    RobustTurbineWSClient = None


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
        
        # Instrumentation
        self._ws_message_count = 0 
        self._ws_messages_total = 0  # Total raw messages
        self._ws_messages_parsed_ok = 0  # Valid typed messages
        self._ws_messages_by_market: Dict[str, int] = {} # Per market counters
        self._ws_last_message_ts = None  # Any message
        self._ws_last_market_update_ts = None  # Market data (book/trade)
        
        self._active_subscriptions: Set[str] = set()
        self._watchdog_task = None
        
        # Market cache for settlement addresses
        self._market_cache: Dict[str, Any] = {}
        
        # Reconciliation state
        self._reconcile_task = None
        self._last_reconciled_positions: Dict[str, float] = {}
        self._last_reconciled_orders: Dict[str, Any] = {}
        self._reconcile_interval = 5.0  # seconds
    
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
            if RobustTurbineWSClient is None:
                raise ImportError("turbine-py-client not installed")

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
                
            # Start reconciliation loop
            await self.start_reconciliation()
            
            logger.info("TurbineAdapter: WebSocket connected")
        except Exception as e:
            logger.error(f"TurbineAdapter: Failed to connect WebSocket: {e}")
            raise

    def get_last_message_age(self) -> float:
        """Get time in seconds since last *valid market data* message."""
        if not self._ws_last_market_update_ts:
            return float('inf')
        return time.time() - self._ws_last_market_update_ts

    def is_feed_fresh(self, max_age_seconds: float) -> bool:
        """Check if WebSocket feed is fresh."""
        return self.get_last_message_age() <= max_age_seconds

    async def _process_ws_messages(self):
        """Background task to process incoming WebSocket messages."""
        import time
        from ..core.events import BookSnapshotEvent, TradeEvent, Side as InternalSide
        
        try:
            async for message in self._ws_connection:
                self._ws_messages_total += 1
                self._ws_message_count += 1
                self._ws_last_message_ts = time.time()
                
                msg_type = getattr(message, 'type', 'unknown')
                market_id = getattr(message, 'market_id', 'unknown')
                
                # Debug: Log messages if TURBINE_WS_DEBUG is set
                if os.environ.get("TURBINE_WS_DEBUG"):
                    msg_str = str(message)[:500]
                    logger.info(f"TurbineAdapter: WS message #{self._ws_message_count}: {msg_str}")
                
                # Count parsed messages
                if msg_type in ['orderbook', 'trade', 'order_cancelled']:
                    self._ws_messages_parsed_ok += 1
                    
                    # Update market specific counters
                    if market_id:
                        self._ws_messages_by_market[market_id] = self._ws_messages_by_market.get(market_id, 0) + 1
                        
                        # Only update global freshness if it's a relevant market
                        self._ws_last_market_update_ts = time.time()
                
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
            close_info = ""
            if hasattr(e, 'code') and hasattr(e, 'reason'):
                close_info = f" (close code: {e.code}, reason: {e.reason})"
            logger.error(f"WebSocket message processor error: {e}{close_info}", exc_info=True)

    def _translate_to_internal_events(self, message) -> list:
        """Translate raw WSMessage to internal events (BookSnapshot, Trade)."""
        from ..core.events import BookSnapshotEvent, TradeEvent, Side as InternalSide
        import time
        
        events = []
        msg_type = getattr(message, 'type', '')
        data = getattr(message, 'data', {})
        market_id = getattr(message, 'market_id', None)
        
        if not market_id or not data:
            return events

        # Scale factor for Turbine (6 decimals usually, check docs carefully or config)
        # Using 1e6 as standard for this exchange based on previous knowledge
        # Price: 0-1,000,000 -> 0.0-1.0
        # Size: 1,000,000 -> 1.0 share
        SCALE = 1_000_000.0

        if msg_type == 'orderbook':
            # Data structure: {'bids': [{'price': int, 'size': int}], 'asks': [...], 'lastUpdate': int}
            bids_data = data.get('bids', [])
            asks_data = data.get('asks', [])
            last_update = data.get('lastUpdate', int(time.time() * 1000))
            
            # Map Bids
            bids_tuples = [
                (float(b['price']) / SCALE, float(b['size']) / SCALE)
                for b in bids_data
            ]
                
            # Map Asks
            asks_tuples = [
                (float(a['price']) / SCALE, float(a['size']) / SCALE)
                for a in asks_data
            ]
            
            events.append(BookSnapshotEvent(
                seq=last_update,
                market_id=market_id,
                bids=bids_tuples,
                asks=asks_tuples
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
        
        # Default 25s (less than the 30s stale limit) to prevent lockout
        stall_threshold = float(os.environ.get("TURBINE_WS_STALL_SECONDS", "25.0"))
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
            if RobustTurbineWSClient is None:
                 raise ImportError("turbine-py-client not installed")

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
                    # Match strict pattern: both aliases
                    await self._ws_connection.subscribe_orderbook(market_id)
                    await self._ws_connection.subscribe_trades(market_id)
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
                
        # Cancel Reconciliation
        if self._reconcile_task:
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass
        
        # Close REST client
        if self._rest_client:
            self._rest_client.close()
    def _check_fresh_or_raise(self, action: str):
        """Raise RuntimeError if feed is stale.
        
        Args:
            action: Description of action being attempted (e.g., "Place Order").
        """
        # Ensure max_age is consistent with execution engine (default 30s)
        max_age = float(os.environ.get("TURBINE_MAX_DATA_AGE_S", 30.0))
        
        if not self.is_feed_fresh(max_age):
            age = self.get_last_message_age()
            msg = f"Stale feed (age {age:.1f}s > {max_age}s). Blocking {action}."
            # We raise RuntimeError to stop execution immediately.
            # Caller can handle logging.
            raise RuntimeError(msg)

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
                # Per turbine-py-client/examples/websocket_stream.py
                # Explicitly call both for strict compliance, even if aliases
                await self._ws_connection.subscribe_orderbook(market_id)
                await self._ws_connection.subscribe_trades(market_id)
                
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
        self._check_fresh_or_raise("place_order")
        
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
            # BUT we gate it on web3 availability to valid crash loops
            if not hasattr(self, '_can_sign_permit'):
                try:
                    import web3
                    self._can_sign_permit = True
                except ImportError:
                    logger.warning("TurbineAdapter: 'web3' module not found. USDC permit signing disabled. Gasless trading may fail.")
                    self._can_sign_permit = False
            
            if getattr(self, '_can_sign_permit', False):
                try:
                    permit = self._rest_client.sign_usdc_permit(
                        value=permit_amount,
                        settlement_address=settlement_address,
                    )
                    signed_order.permit_signature = permit
                    logger.debug(f"Attached USDC permit: {permit_amount} units")
                except Exception as permit_err:
                    logger.warning(f"Failed to sign USDC permit: {permit_err}. Order may fail.")
            elif not hasattr(self, '_warned_permit_disabled'):
                 # Log only once per run if disabled
                 logger.warning("Skipping USDC permit (signing disabled). Order might fail.")
                 self._warned_permit_disabled = True
            
            # Submit the order
            result = self._rest_client.post_order(signed_order)
            order_hash = result.get('orderHash', signed_order.order_hash)
            
            # matches can be a list or an int count depending on API version
            matches_raw = result.get('matches')
            if isinstance(matches_raw, int):
                match_count = matches_raw
            elif isinstance(matches_raw, list):
                match_count = len(matches_raw)
            else:
                match_count = 0
            
            logger.info(f"Order placed: {order_hash} (matches: {match_count})")
            
            # TRUTH CHECK: Updated to verify_order_exists (list based)
            # Only verify if NO matches (if matches exist, it might be filled/gone already)
            if match_count == 0:
                is_verified = await self._verify_order_exists(order_hash)
                if not is_verified:
                    logger.error(f"CRITICAL: Order {order_hash} placed (unmatched) but NOT found in API verification!")
            else:
                 logger.info(f"Order {order_hash} matched immediately ({match_count} fills). Skipping open order check.")
            
            return order_hash
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    async def _verify_order_exists(self, order_hash: str) -> bool:
        """Verify order exists by fetching open orders."""
        try:
            # Use get_open_orders to leverage the list endpoint
            orders = await asyncio.to_thread(
                self._rest_client.get_orders, 
                trader=self._rest_client.address,
                status="open"
            )
            
            # Check existence
            for o in orders:
                if o.order_hash == order_hash:
                    logger.info(f"Truth Check: Order {order_hash} verification PASSED")
                    # Update cache immediately!
                    self._last_reconciled_orders[order_hash] = o
                    return True
        except Exception as e:
            logger.error(f"Truth Check: Order {order_hash} verification FAILED: {e}")
        return False
        
    async def start_reconciliation(self):
        """Start the state reconciliation loop."""
        if not self._reconcile_task or self._reconcile_task.done():
            self._reconcile_task = asyncio.create_task(self._reconciliation_loop())
            logger.info("TurbineAdapter: Reconciliation loop started")

    # _reconciliation_loop moved to bottom to be contiguous with get_positions update
    # Need to keep it method-ordered if possible but I already updated it in previous block
    # Actually wait, I moved _reconciliation_loop definition in the previous block DOWN?
    # No, I replaced get_positions AND get_open_orders in the previous block.
    # The previous block REPLACED _reconciliation_loop logic? No.
    # The previous block ENDED at 856 which was get_open_orders. 
    # Ah, I replaced get_positions and get_open_orders. 
    # I did NOT replace _reconciliation_loop in previous block? 
    # The previous block targeted StartLine:821 which is get_positions. 
    # The context shows get_positions then get_open_orders. 
    # Use view file to check where _reconciliation_loop is.
    # It was at line 692 in previous view.
    
    # I will stick to JUST replacing verification and cancel here.

    async def cancel_order(self, order: Order):
        """Cancel specific order using CANONICAL exchange params."""
        self._require_auth()
        self._check_fresh_or_raise("cancel_order")
        
        try:
            order_hash = order.exchange_order_id or order.client_order_id
            if not order_hash:
                logger.error("Cancel failed: No order hash provided")
                return

            # RECONCILIATION-FIRST LOOKUP
            cached_order = self._last_reconciled_orders.get(order_hash)
            
            target_side_str = None
            target_market = order.market_id
            
            if cached_order:
                # Use cached truth (turbine client object)
                # cached_order.side is int: 0=BUY, 1=SELL
                target_side_str = "buy" if cached_order.side == 0 else "sell"
                target_market = cached_order.market_id
                logger.debug(f"Cancel: Found cached order {order_hash}, side={target_side_str}, mkt={target_market}")
            else:
                # Fallback to local passed order object
                target_side_str = "buy" if order.side == Side.BID else "sell"
                logger.warning(f"Cancel: Order {order_hash} not in recon cache. Using local side {target_side_str}.")

            try:
                # The generic cancel_order calls DELETE /orders/{orderHash} with params
                # We must be explicit with args
                # RE-READING CLIENT:
                # def cancel_order(self, order_hash: str, market_id: str, side: Side)
                # It converts side enum to string "buy"/"sell".
                # We need to pass the correct Enum.
                from turbine_client.types import Side as TurbineSide
                tside = TurbineSide.BUY if target_side_str == "buy" else TurbineSide.SELL
                
                result = await asyncio.to_thread(
                    self._rest_client.cancel_order,
                    order_hash=order_hash,
                    market_id=target_market,
                    side=tside,
                )
                logger.info(f"Order cancelled: {order_hash}")
                
            except Exception as e:
                # Handle 404 - "Not Found"
                if "404" in str(e) or "not found" in str(e).lower():
                    logger.warning(f"Cancel 404 for {order_hash}. Verifying if it is closed...")
                    exists = await self._verify_order_exists(order_hash)
                    if not exists:
                         logger.info(f"Order {order_hash} confirmed gone (404 was valid/redundant).")
                         return
                    else:
                         logger.error(f"Order {order_hash} exists in API list but Cancel 404'd! Parameter mismatch?")
                         raise
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise

    async def cancel_all(self, market_id: Optional[str] = None):
        """Cancel all open orders (safety switch)."""
        self._require_auth()
        self._check_fresh_or_raise("cancel_all")
        
        logger.info(f"TurbineAdapter: cancel_all triggered for market {market_id}")
        
        try:
            from turbine_client.types import Side as TurbineSide
            
            # Fetch open orders to know what to cancel
            open_orders = await asyncio.to_thread(
                self._rest_client.get_orders,
                trader=self._rest_client.address,
                market_id=market_id,
                status="open",
            )
            
            logger.info(f"Cancelling {len(open_orders)} open orders")
            
            for turbine_order in open_orders:
                try:
                    # STRICT MAPPING: 0=BUY, 1=SELL
                    side_enum = TurbineSide.BUY if turbine_order.side == 0 else TurbineSide.SELL
                    
                    await asyncio.to_thread(
                        self._rest_client.cancel_order,
                        order_hash=turbine_order.order_hash,
                        market_id=turbine_order.market_id,
                        side=side_enum,
                    )
                    logger.info(f"Cancelled order: {turbine_order.order_hash}")
                except Exception as e:
                     logger.error(f"Failed to cancel order {turbine_order.order_hash}: {e}")
                     
        except Exception as e:
            logger.error(f"cancel_all failed: {e}")
            raise

    async def get_positions(self) -> Dict[str, float]:
        """Fetch current positions snapshot safely."""
        if not self._private_key:
            return {}
        
        try:
            # Add debug logging for structure
            # positions = await asyncio.to_thread(...)
            
            positions = await asyncio.to_thread(
                self._rest_client.get_user_positions,
                address=self._rest_client.address,
                chain_id=self._chain_id,
            )
            
            # ROBUST PARSING: positions might be None if API returns 204 or null
            if positions is None:
                return {}
                
            result = {}
            for pos in positions:
                # net = yes_shares - no_shares (in contract units, 6 decimals)
                net_shares = (pos.yes_shares - pos.no_shares) / 1_000_000
                result[pos.market_id] = net_shares
            
            return result
            
        except TypeError as te:
            if "'NoneType' object is not iterable" in str(te):
                # This is normal for new accounts or empty portfolios on some endpoints
                logger.debug("get_positions: API returned None (no positions?), treating as empty.")
                return {}
            logger.error(f"Failed to fetch positions (TypeError): {te}")
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return {}

    async def get_open_orders(self) -> List[Order]:
        """Fetch open orders snapshot.
        
        Returns:
            List of Order objects representing open orders.
        """
        if not self._private_key:
            return []
        
        try:
            turbine_orders = await asyncio.to_thread(
                self._rest_client.get_orders,
                trader=self._rest_client.address,
                status="open",
            )
            
            # Convert turbine Order objects to our Order objects
            orders = []
            for to in turbine_orders:
                our_side = Side.BID if to.side == 0 else Side.ASK
                o = Order(
                    client_order_id=to.order_hash,  # Use order_hash as client ID for adopted orders
                    market_id=to.market_id,
                    side=our_side,
                    price=to.price / 10000.0,
                    size=to.remaining_size / 1_000_000.0
                )
                o.exchange_order_id = to.order_hash
                o.status = OrderStatus.OPEN
                orders.append(o)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    # ... (skipping register_callback etc) ...

    async def _reconciliation_loop(self):
        """Periodic loop to fetch authoritative state."""
        while True:
            try:
                await asyncio.sleep(self._reconcile_interval)
                
                # 1. Fetch Positions
                positions = await self.get_positions()
                self._last_reconciled_positions = positions
                
                # 2. Fetch Open Orders
                # Use explicit wrapper to call Thread
                open_orders_list = await asyncio.to_thread(
                    self._rest_client.get_orders,
                    trader=self._rest_client.address,
                    status="open"
                )
                
                # Update cache with RAW turbine orders to enable canonical lookup
                self._last_reconciled_orders = {o.order_hash: o for o in open_orders_list}
                
                # 3. Log Tick Stats
                ws_age = self.get_last_message_age()
                ws_status = "OK" if ws_age < 30 else "STALE"
                ws_msg_cnt = self._ws_messages_parsed_ok
                
                # Dump counters for first few active markets
                top_mkts = list(self._ws_messages_by_market.items())[:3]
                ws_stats = f"{ws_msg_cnt} total | " + " ".join([f"{k[:6]}={v}" for k,v in top_mkts])
                
                if ws_status != "OK":
                     logger.warning(
                        f"ADAPTER WARNING | "
                        f"Pos: {len(positions)} mkts | "
                        f"Orders: {len(open_orders_list)} open | "
                        f"WS: {ws_status} ({ws_stats}) age {ws_age:.1f}s"
                    )
                else:
                    logger.debug(
                        f"ADAPTER TICK | "
                        f"Pos: {len(positions)} mkts | "
                        f"Orders: {len(open_orders_list)} open | "
                        f"WS: {ws_status} ({ws_stats}) age {ws_age:.1f}s"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconciliation error: {e}")
                await asyncio.sleep(5)

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
