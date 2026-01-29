import unittest
from src.core.state import OrderBook, StateStore, Order
from src.core.events import Side
from src.strategy.engine import StrategyEngine
from src.risk.engine import RiskEngine

class TestOrderBook(unittest.TestCase):
    def test_apply_delta_and_imbalance(self):
        ob = OrderBook("TEST")
        # Add Bids
        ob.apply_delta(1, Side.BID, 100.0, 10.0)
        ob.apply_delta(2, Side.BID, 99.0, 10.0)
        # Add Asks
        ob.apply_delta(3, Side.ASK, 101.0, 5.0)
        
        self.assertEqual(ob.get_best_bid(), (100.0, 10.0))
        self.assertEqual(ob.get_best_ask(), (101.0, 5.0))
        self.assertEqual(ob.get_mid(), 100.5)
        
        # Imbalance: Bid Size 20, Ask Size 5 => 4.0
        imb = ob.get_imbalance(5)
        self.assertAlmostEqual(imb, 4.0)

class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.config = {
            'strategy': {
                'base_spread': 2.0,
                'min_price': 0.0,
                'max_price': 1000.0,
                'skew_factor': 10.0,
                'imbalance_depth_n': 5,
                'imbalance_threshold': 2.0,
                'overlay_bias': 1.0,
                'persistence_seconds': 0
            },
            'risk': {'max_inventory_units': 100.0}
        }
        self.state = StateStore()

    def test_skew_long(self):
        # Position +50, Max 100 => Ratio 0.5
        # Skew = -0.5 * 10 = -5.0
        self.state.positions["TEST"] = 50.0
        
        engine = StrategyEngine(self.config, self.state)
        
        # Setup Book for Mid=100
        ob = self.state.get_orderbook("TEST")
        ob.apply_delta(1, Side.BID, 99.0, 1.0)
        ob.apply_delta(2, Side.ASK, 101.0, 1.0)
        
        bid, ask = engine.get_desired_quotes("TEST")
        
        expected_mid = 100.0
        expected_skew = -5.0
        # No imbalance overlay (ratio 1.0)
        
        # Bid = Mid - 1 + Skew = 100 - 1 - 5 = 94
        # Ask = Mid + 1 + Skew = 100 + 1 - 5 = 96
        self.assertEqual(bid, 94.0)
        self.assertEqual(ask, 96.0)

class TestRisk(unittest.TestCase):
    def setUp(self):
        self.config = {
            'risk': {
                'max_inventory_units': 100.0,
                'max_portfolio_exposure': 10000.0
            }
        }
        self.state = StateStore()
        self.engine = RiskEngine(self.config, self.state)

    def test_inventory_check(self):
        self.state.positions["TEST"] = 90.0
        
        # Buy 11 => 101 => Fail
        o1 = Order("1", "TEST", Side.BID, 100, 11)
        self.assertFalse(self.engine.check_order_risk(o1))
        
        # Sell 11 => 79 => Pass
        o2 = Order("2", "TEST", Side.ASK, 100, 11)
        self.assertTrue(self.engine.check_order_risk(o2))

class TestTurbineAdapter(unittest.TestCase):
    def test_missing_auth_fails_closed(self):
        """Verify that trading operations fail closed when auth is missing."""
        import os
        
        # Save original env vars
        orig_pk = os.environ.get('TURBINE_PRIVATE_KEY')
        orig_api_key = os.environ.get('TURBINE_API_KEY_ID')
        orig_api_secret = os.environ.get('TURBINE_API_PRIVATE_KEY')
        
        try:
            # Clear all auth env vars
            for key in ['TURBINE_PRIVATE_KEY', 'TURBINE_API_KEY_ID', 
                       'TURBINE_API_PRIVATE_KEY', 'INTEGRATION_WALLET_PRIVATE_KEY',
                       'INTEGRATION_API_KEY_ID', 'INTEGRATION_API_PRIVATE_KEY']:
                os.environ.pop(key, None)
            
            # Import here to avoid module-level import issues
            from src.exchange.turbine import TurbineAdapter
            
            config = {
                'exchange': {
                    'base_url': 'https://api.turbinefi.com',
                    'chain_id': 137,
                }
            }
            
            # Should initialize without error (read-only mode)
            adapter = TurbineAdapter(config)
            
            # Create a dummy order
            test_order = Order("test_id", "market_123", Side.BID, 50.0, 1.0)
            
            # place_order should fail closed
            import asyncio
            with self.assertRaises(NotImplementedError) as ctx:
                asyncio.run(adapter.place_order(test_order))
            
            self.assertIn("authentication", str(ctx.exception).lower())
            self.assertIn("TURBINE_PRIVATE_KEY", str(ctx.exception))
            
        finally:
            # Restore original env vars
            if orig_pk:
                os.environ['TURBINE_PRIVATE_KEY'] = orig_pk
            if orig_api_key:
                os.environ['TURBINE_API_KEY_ID'] = orig_api_key
            if orig_api_secret:
                os.environ['TURBINE_API_PRIVATE_KEY'] = orig_api_secret

if __name__ == '__main__':
    unittest.main()
