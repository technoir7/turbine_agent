"""WebSocket stream probe for Turbine integration verification.

Tests that:
1. WebSocket connection works
2. subscribe() successfully subscribes to a market
3. At least one message is received within 10 seconds

Exit codes:
- 0: Success (received at least one message)
- 1: Failure (timeout or error)
"""
import asyncio
import sys
import logging
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
    """Test WebSocket stream reception."""
    
    # Step 1: Get BTC quick market via REST
    logger.info("Fetching BTC quick market via REST...")
    try:
        rest_client = TurbineClient(
            host="https://api.turbinefi.com",
            chain_id=137,
        )
        quick_market = rest_client.get_quick_market("BTC")
        market_id = quick_market.market_id
        logger.info(f"BTC Quick Market: {market_id}")
        rest_client.close()
    except Exception as e:
        logger.error(f"Failed to fetch quick market: {e}")
        return 1
    
    # Step 2: Connect to WebSocket
    logger.info("Connecting to WebSocket...")
    ws_client = TurbineWSClient(host="https://api.turbinefi.com")
    
    try:
        async with ws_client.connect() as stream:
            logger.info(f"WebSocket connected to {ws_client.url}")
            
            # Step 3: Subscribe to market
            logger.info(f"Subscribing to market {market_id}...")
            await stream.subscribe(market_id)
            logger.info("Subscribed successfully")
            
            # Step 4: Wait for one message (10 second timeout)
            logger.info("Waiting for WebSocket messages (10s timeout)...")
            try:
                async with asyncio.timeout(10):
                    async for message in stream:
                        logger.info(f"✅ RECEIVED MESSAGE: type={message.type}, market_id={getattr(message, 'market_id', 'N/A')}")
                        
                        # Show message details
                        if message.type == "orderbook":
                            logger.info(f"   Orderbook update received")
                            if hasattr(message, 'data') and message.data:
                                logger.info(f"   Data keys: {list(message.data.keys()) if isinstance(message.data, dict) else 'N/A'}")
                        elif message.type == "trade":
                            logger.info(f"   Trade update received")
                        
                        # Success - received at least one message
                        logger.info("✅ SUCCESS: WebSocket is receiving messages")
                        return 0
                        
            except asyncio.TimeoutError:
                logger.error("❌ TIMEOUT: No messages received within 10 seconds")
                logger.error("Possible issues:")
                logger.error("  1. Market has no activity")
                logger.error("  2. Subscribe message not sent correctly")
                logger.error("  3. WebSocket server not sending data")
                return 1
                
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
