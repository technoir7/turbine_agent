import unittest
import asyncio
from src.supervisor import Supervisor

class TestIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_full_loop_simulation(self):
        # 1. Init Supervisor in SIM mode
        bot = Supervisor("config.yaml", simulated=True)
        
        # 2. Start (stubbed start to return task)
        task = asyncio.create_task(bot.start())
        
        # 3. Wait for some activity
        await asyncio.sleep(5.0)
        
        # 4. Check State
        # Should have orderbook data
        ob = bot.state.get_orderbook(bot.market_id)
        self.assertIsNotNone(ob.get_mid())
        
        # Should have placed orders
        open_orders = [o for o in bot.state.orders.values() if o.status.name == 'OPEN']
        # We expect orders because simulated exchange sends data and strategy quotes
        self.assertTrue(len(open_orders) > 0 or len(bot.state.orders) > 0)
        
        # 5. Stop
        await bot.stop()
        await task

if __name__ == '__main__':
    unittest.main()
