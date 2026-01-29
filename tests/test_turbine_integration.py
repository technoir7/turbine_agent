import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio

# Integration tests for Turbine adapter wiring
# These tests verify correct integration with turbine-py-client using mocks
# NO strategy assertions - integration only

class TestTurbineAdapter(unittest.IsolatedAsyncioTestCase):
    """Test TurbineAdapter integration wiring with mocked turbine_client."""
    
    @patch('src.exchange.turbine.TurbineClient')
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
    
    @patch('src.exchange.turbine.TurbineClient')
    async def test_place_order_with_usdc_permit(self, mock_client_class):
        """Test that place_order includes USDC permit signature."""
        from src.exchange.turbine import TurbineAdapter
        from src.core.state import Order
        from src.core.events import Side
        
        # Setup mocks
        mock_client = Mock()
        mock_signed_order = Mock()
        mock_signed_order.order_hash = '0xabc123'
        mock_signed_order.permit_signature = None  # Will be set by adapter
        
        mock_client.create_limit_buy = Mock(return_value=mock_signed_order)
        mock_client.sign_usdc_permit = Mock(return_value={'v': 27, 'r': '0x...', 's': '0x...'})
        mock_client.post_order = Mock(return_value={'orderHash': '0xabc123'})
        mock_client.get_markets = Mock(return_value=[
            Mock(id='market123', settlement_address='0xsettle123', contract_address='0xcontract123')
        ])
        
        mock_client_class.return_value = mock_client
        
        config = {'exchange': {'base_url': 'https://api.turbinefi.com', 'chain_id': 137}}
        
        with patch.dict('os.environ', {'TURBINE_PRIVATE_KEY': '0x123',
                                         'TURBINE_API_KEY_ID': 'key123',
                                         'TURBINE_API_PRIVATE_KEY': 'privkey123'}):
            adapter = Turb ineAdapter(config)
        
        # Create test order
        order = Order(
            order_id='test1',
            market_id='market123',
            side=Side.BID,
            price=0.5,  # 50%
            size=1.0,   # 1 share
        )
        
        # Place order
        result = await adapter.place_order(order)
        
        # Verify USDC permit was signed and attached
        mock_client.sign_usdc_permit.assert_called_once()
        permit_call_kwargs = mock_client.sign_usdc_permit.call_args[1]
        self.assertIn('value', permit_call_kwargs)
        self.assertIn('settlement_address', permit_call_kwargs)
        
        # Verify order was posted
        mock_client.post_order.assert_called_once_with(mock_signed_order)
        self.assertEqual(result, '0xabc123')
    
    @patch('src.exchange.turbine.TurbineWSClient')
    @patch('src.exchange.turbine.TurbineClient')
    async def test_websocket_subscribe_pattern(self, mock_client_class, mock_ws_class):
        """Test WebSocket subscribe uses correct turbine-py-client pattern."""
        from src.exchange.turbine import TurbineAdapter
        
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_ws = Mock()
        mock_stream = AsyncMock()
        mock_stream.subscribe_orderbook = AsyncMock()
        mock_stream.subscribe_trades = AsyncMock()
        mock_ws.connect = MagicMock()
        mock_ws.connect.return_value.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_ws.connect.return_value.__aexit__ = AsyncMock()
        mock_ws_class.return_value = mock_ws
        
        config = {'exchange': {'base_url': 'https://api.turbinefi.com', 'chain_id': 137}}
        
        with patch.dict('os.environ', {}):
            adapter = TurbineAdapter(config)
        
        # Connect WebSocket
        await adapter.connect()
        
        # Subscribe to markets
        await adapter.subscribe_markets(['market123', 'market456'])
        
        # Verify correct subscribe pattern per websocket_stream.py
        self.assertEqual(mock_stream.subscribe_orderbook.call_count, 2)
        self.assertEqual(mock_stream.subscribe_trades.call_count, 2)
        mock_stream.subscribe_orderbook.assert_any_call('market123')
        mock_stream.subscribe_orderbook.assert_any_call('market456')


class TestTurbineIntegrationSmokeTest(unittest.TestCase):
    """Test that connectivity probe can be imported and has correct structure."""
    
    def test_connectivity_probe_import(self):
        """Verify connectivity probe module structure."""
        from src.tools import connectivity_probe
        
        # Verify main functions exist
        self.assertTrue(hasattr(connectivity_probe, 'main'))
        self.assertTrue(hasattr(connectivity_probe, 'test_basic_connectivity'))
        self.assertTrue(hasattr(connectivity_probe, 'test_websocket'))


if __name__ == '__main__':
    unittest.main()
