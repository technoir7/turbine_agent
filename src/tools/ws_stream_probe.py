"""WebSocket stream probe for Turbine integration verification.
Regression tool for connection stalling.

Tests that:
1. WebSocket connection works
2. Subscribe works
3. Messages flow continuously for at least 30 seconds
4. Fails if silence > 10 seconds

Exit codes:
- 0: Success (continuous flow)
- 1: Failure (timeout, stall, or error)
"""
import asyncio
import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from turbine_client import TurbineClient, TurbineWSClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting WebSocket Regression Probe...")
    
    # Step 1: Get BTC market
    try:
        rest_client = TurbineClient(host="https://api.turbinefi.com", chain_id=137)
        quick_market = rest_client.get_quick_market("BTC")
        market_id = quick_market.market_id
        logger.info(f"Target Market: {market_id}")
        rest_client.close()
    except Exception as e:
        logger.error(f"Failed to fetch market: {e}")
        return 1

    # Step 2: Connect
    # Use Robust Client to match production fix
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

    ws_client = RobustTurbineWSClient(host="https://api.turbinefi.com")
    
    try:
        async with ws_client.connect() as stream:
            logger.info("Connected.")
            await stream.subscribe(market_id)
            logger.info(f"Subscribed to {market_id}")
            
            start_time = time.time()
            last_msg_time = time.time()
            msg_count = 0
            
            # Run for 30 seconds
            while time.time() - start_time < 30:
                try:
                    # Wait for message with 10s timeout (stall detection)
                    async with asyncio.timeout(10.0):
                        # We use __aiter__ manually to get one message at a time
                        # But stream is an iterator, so we can just use an async loop or next()
                        # To control timing precisely, let's use the iterator in a way we can breakout
                        
                        # Note: async for loop might block, so we wrapping next(iter) is better 
                        # but standard pattern is async for.
                        # We'll use a task to wrap the fetch? 
                        # Actually, just waiting on the stream iterator is simplest.
                        
                        message = await stream.__aiter__().__anext__()
                        
                        msg_count += 1
                        last_msg_time = time.time()
                        
                        msg_type = getattr(message, 'type', 'unknown')
                        logger.info(f"Msg #{msg_count}: {msg_type}")
                        
                except asyncio.TimeoutError:
                    logger.error("❌ STALL DETECTED: No message for 10 seconds!")
                    return 1
                except StopAsyncIteration:
                    logger.error("❌ Stream closed unexpectedly.")
                    return 1
                
                # Check safeguards
                if time.time() - last_msg_time > 10.0:
                     logger.error("❌ STALL DETECTED (Safeguard): No message for 10 seconds!")
                     return 1
                     
            if msg_count < 3:
                logger.error(f"❌ Low message count: {msg_count} (Expected > 3)")
                return 1
                
            logger.info("✅ PROBE PASSED: Continuous stream for 30s.")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
