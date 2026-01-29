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
                'persistence_seconds': 0,
                # Extremes config - set to not interfere with mid=100
                'extreme_low': 0.10,
                'extreme_high': 990.0,  # High enough to not trigger
                'extreme_spread_mult': 2.0,
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

class TestStrategyExtremes(unittest.TestCase):
    def setUp(self):
        self.config = {
            'strategy': {
                'base_spread': 0.02,  # 2%
                'min_price': 0.01,
                'max_price': 0.99,
                'skew_factor': 0.01,
                'imbalance_depth_n': 5,
                'imbalance_threshold': 2.0,
                'overlay_bias': 0.005,
                'persistence_seconds': 0,
                'extreme_low': 0.10,
                'extreme_high': 0.90,
                'extreme_spread_mult': 2.0,
            },
            'risk': {'max_inventory_units': 100.0}
        }
        self.state = StateStore()

    def test_extremes_risk_control(self):
        """Verify that quotes widen in extreme zones."""
        # Neutral position
        self.state.positions["TEST"] = 0.0
        
        engine = StrategyEngine(self.config, self.state)
        
        # Setup orderbook with extreme mid (5% = 0.05)
        ob = self.state.get_orderbook("TEST")
        ob.apply_delta(1, Side.BID, 0.04, 10.0)
        ob.apply_delta(2, Side.ASK, 0.06, 10.0)
        # Mid = 0.05, which is < extreme_low (0.10)
        
        bid, ask = engine.get_desired_quotes("TEST")
        
        # Should have quotes
        self.assertIsNotNone(bid)
        self.assertIsNotNone(ask)
        
        # Spread should be widened (> base_spread * extreme_spread_mult * 0.8)
        # Base spread = 0.02, widened should be ~0.04
        spread = ask - bid
        base_spread = self.config['strategy']['base_spread']
        expected_min_spread = base_spread * 1.5  # Conservative check
        self.assertGreater(spread, expected_min_spread,
                          f"Spread {spread} not widened enough (expected > {expected_min_spread})")
        
        # Should flag as extreme zone
        self.assertTrue(engine.is_extreme_zone("TEST"))

if __name__ == '__main__':
    unittest.main()
