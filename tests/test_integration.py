import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
import time

# Integration tests for Turbine adapter wiring
# These tests verify correct integration with turbine-py-client using mocks
# NO strategy assertions - integration only

class TestTurbineAdapter(unittest.IsolatedAsyncioTestCase):
    """Test TurbineAdapter integration wiring with mocked turbine_client."""
    
    @patch('turbine_client.TurbineClient')
    def test_adapter_initialization(self, mock_client_class):
        """Test that TurbineAdapter correctly initializes turbine_client."""
        from src.exchange.turbine import TurbineAdapter
        
        # Setup mock
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Create adapter with config
        config = {
            'exchange': {
                'base_url': 'https://api.turbinefi.com',
                'chain_id': 137
            }
        }
        
        # Mock environment variables
        with patch.dict('os.environ', {'TURBINE_PRIVATE_KEY': '0x123', 
                                        'TURBINE_API_KEY_ID': 'key123',
                                        'TURBINE_API_PRIVATE_KEY': 'privkey123'}):
            adapter = TurbineAdapter(config)
        
        # Verify turbine_client was initialized with correct params
        mock_client_class.assert_called()
        call_kwargs = mock_client_class.call_args[1]
        self.assertEqual(call_kwargs['host'], 'https://api.turbinefi.com')
        self.assertEqual(call_kwargs['chain_id'], 137)
        self.assertIn('private_key', call_kwargs)
        self.assertIn('api_key_id', call_kwargs)
        self.assertIn('api_private_key', call_kwargs)
    
    @patch('turbine_client.TurbineWSClient')
    @patch('turbine_client.TurbineClient')
    @patch('websockets.connect')
    async def test_websocket_subscribe_pattern(self, mock_ws_connect, mock_client_class, mock_ws_class):
        """Test WebSocket subscribe uses correct turbine-py-client pattern."""
        from src.exchange.turbine import TurbineAdapter
        
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock websockets connection
        mock_ws_connection = AsyncMock()
        mock_ws_connect.return_value = mock_ws_connection
        
        # Mock WSStream returned by RobustTurbineWSClient (which instantiates WSStream(connection))
        # Since we can't easily mock the WSStream instantiation inside the local class,
        # we rely on the fact that RobustTurbineWSClient uses the connection we mock.
        # BUT WSStream wraps the connection. WSStream.subscribe calls connection.send.
        # So we can verify connection.send is called.
        
        # HOWEVER, the adapter calls _ws_connection.subscribe_orderbook.
        # _ws_connection is the WSStream instance yielded by RobustTurbineWSClient.connect().
        # RobustTurbineWSClient.connect yields WSStream(self._connection).
        # We need to mock WSStream to verify method calls on it.
        # But WSStream is imported inside connect().
        
        # Alternative: We can mock subscription calls by mocking the SEND on the connection?
        # No, we want to verify subscribe_orderbook is called.
        
        # Use a simpler approach: Patch WSStream where it is imported.
        # It is imported as: from turbine_client.ws.client import WSStream
        # inside connect().
        # We can't patch local import easily.
        
        # Let's verify what we can:
        # The adapter calls 'await self._ws_connection.subscribe_orderbook(market_id)'
        # 'self._ws_connection' IS the object yielded by connect().
        # In RobustTurbineWSClient.connect:
        #   yield WSStream(self._connection)
        
        # If we patch turbine_client.ws.client.WSStream, we can control what is yielded.
        pass

    @patch('src.exchange.turbine.WSStream')
    @patch('src.exchange.turbine.websockets.connect', new_callable=AsyncMock)
    @patch('src.exchange.turbine.TurbineWSClient')
    @patch('src.exchange.turbine.TurbineClient')
    async def test_websocket_subscribe_pattern(self, mock_client_class, mock_ws_class, mock_ws_connect, mock_ws_stream_cls):
        """Test WebSocket subscribe uses correct turbine-py-client pattern."""
        from src.exchange.turbine import TurbineAdapter
        
        # Setup mocks
        mock_ws_stream = AsyncMock() # The instance
        mock_ws_stream.subscribe_orderbook = AsyncMock()
        mock_ws_stream.subscribe_trades = AsyncMock()
        mock_ws_stream_cls.return_value = mock_ws_stream
        
        mock_ws_connection = AsyncMock()
        mock_ws_connect.return_value = mock_ws_connection
        
        config = {'exchange': {'base_url': 'https://api.turbinefi.com', 'chain_id': 137}}
        
        with patch.dict('os.environ', {}):
            adapter = TurbineAdapter(config)
        
        # Connect WebSocket
        await adapter.connect()
        # Verify RobustTurbineWSClient initialized and connect called
        # Verify WSStream instantiated
        mock_ws_stream_cls.assert_called()
        
        # Subscribe to markets
        await adapter.subscribe_markets(['market123'])
        
        # Verify correct subscribe pattern: BOTH orderbook and trades
        # The adapter calls these methods on the stream instance
        mock_ws_stream.subscribe_orderbook.assert_called_with('market123')
        mock_ws_stream.subscribe_trades.assert_called_with('market123')

    @patch('src.exchange.turbine.WSStream')
    @patch('src.exchange.turbine.websockets.connect', new_callable=AsyncMock)
    @patch('src.exchange.turbine.TurbineWSClient')
    @patch('src.exchange.turbine.TurbineClient')
    async def test_ws_message_counters_and_freshness(self, mock_client_class, mock_ws_class, mock_ws_connect, mock_ws_stream_cls):
        """Test that WS message counting and freshness logic works correctly."""
        from src.exchange.turbine import TurbineAdapter
        
        mock_ws_stream = AsyncMock()
        mock_ws_stream_cls.return_value = mock_ws_stream
        
        # Create a mock stream that yields messages
        # Use MagicMock for messages so attributes are not async
        msg1 = MagicMock(type='orderbook', market_id='m1')
        msg1.data = {'bids': [], 'asks': []} # minimal data
        
        msg2 = MagicMock(type='trade', market_id='m1')
        msg2.data = {'price': 100, 'size': 100}
        
        msg3 = MagicMock(type='pong', market_id=None)
        
        async def mock_iter():
            yield msg1
            yield msg2
            yield msg3
            
        # Wire up the iteration
        mock_ws_stream.__aiter__.side_effect = mock_iter
        
        # Also need mock connection for initialization
        mock_ws_connect.return_value = AsyncMock()
        
        config = {'exchange': {}}
        adapter = TurbineAdapter(config)
        
        await adapter.connect()
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Check counters
        self.assertGreaterEqual(adapter._ws_messages_total, 3)
        self.assertGreaterEqual(adapter._ws_messages_parsed_ok, 2)
        
        # Check freshness logic
        self.assertTrue(adapter.is_feed_fresh(max_age_seconds=5.0))
        
        # Manually age the last update
        adapter._ws_last_market_update_ts = time.time() - 10.0
        self.assertFalse(adapter.is_feed_fresh(max_age_seconds=5.0))

    @patch('turbine_client.TurbineClient')
    async def test_order_placement_verification(self, mock_client_class):
        """Test verify_order_exists is called after placing order."""
        from src.exchange.turbine import TurbineAdapter
        from src.core.events import Side
        from src.core.state import Order
        
        # Setup mock client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock get_orders for verification (Success case)
        # Returns list of orders, including ours
        mock_order = MagicMock()
        mock_order.order_hash = "0x123"
        mock_client.get_orders.return_value = [mock_order]
        
        # Mock create/post
        mock_client.create_limit_buy.return_value = MagicMock(order_hash="0x123")
        mock_client.post_order.return_value = {'orderHash': '0x123'}
        
        # Mock get_market (called during placement)
        mock_market = MagicMock(id="m1", market_id="m1", settlement_address="0xsettle")
        mock_client.get_market.return_value = mock_market
        mock_client.get_markets.return_value = [mock_market]
        
        config = {'exchange': {}}
        with patch.dict('os.environ', {'TURBINE_PRIVATE_KEY': '0x1', 
                                        'TURBINE_API_KEY_ID': 'k',
                                        'TURBINE_API_PRIVATE_KEY': 's'}):
            adapter = TurbineAdapter(config)
            # Enable signing
            adapter._can_sign_permit = True
            mock_client.sign_usdc_permit.return_value = "0xsig"
            
            # Force fresh feed to pass firewall
            adapter._ws_last_market_update_ts = time.time()
            
            # Place order
            order = Order(
                client_order_id="1",
                market_id="m1",
                side=Side.BID,
                price=0.5,
                size=10
            )
            order_hash = await adapter.place_order(order)
            
            self.assertEqual(order_hash, "0x123")
            # Verify get_orders call was made (list based verification)
            mock_client.get_orders.assert_called()

    @patch('turbine_client.TurbineClient')
    async def test_get_positions_safeguards(self, mock_client_class):
        """Test get_positions handles None return from API."""
        from src.exchange.turbine import TurbineAdapter
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.address = "0xuser"
        
        # Setup None return
        mock_client.get_user_positions.return_value = None
        
        config = {'exchange': {}}
        with patch.dict('os.environ', {'TURBINE_PRIVATE_KEY': '0x1', 
                                        'TURBINE_API_KEY_ID': 'k',
                                        'TURBINE_API_PRIVATE_KEY': 's'}):
            adapter = TurbineAdapter(config)
            
            # Should not crash, return empty dict
            positions = await adapter.get_positions()
            self.assertEqual(positions, {})

if __name__ == '__main__':
    unittest.main()
