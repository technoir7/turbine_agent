"""Integration smoke test for Turbine API.

This script performs a comprehensive connectivity test using the TurbineAdapter:
1. Loads configuration from config.yaml
2. Constructs TurbineAdapter with proper config
3. Tests read-only endpoints (health, quick market, orderbook)
4. Optionally tests WebSocket streaming

Usage:
    # Basic connectivity test (read-only, no auth required)
    python -m src.tools.connectivity_probe

    # With WebSocket test
    python -m src.tools.connectivity_probe --ws-test
    
Exit codes:
    0 - Success (all tests passed)
    1 - Failure (any test failed)
"""
import sys
import argparse
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.loader import load_config
from src.exchange.turbine import TurbineAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_connectivity(adapter: TurbineAdapter) -> bool:
    """Test basic read-only endpoints.
    
    Args:
        adapter: Initialized TurbineAdapter instance.
        
    Returns:
        True if all tests pass, False otherwise.
    """
    print("\n=== Turbine Integration Smoke Test ===\n")
    
    # Test 1: Health check
    print("[1/3] Testing API health...")
    try:
        health = adapter._rest_client.get_health()
        print(f"  ✓ API Health: {health}\n")
    except Exception as e:
        print(f"  ✗ ERROR: {e}\n")
        return False
    
    # Test 2: Get BTC quick market
    print("[2/3] Fetching BTC quick market...")
    try:
        qm = adapter.get_quick_market("BTC")
        strike_price = qm.start_price / 1e8
        print(f"  ✓ Market ID: {qm.market_id}")
        print(f"  ✓ Strike Price: ${strike_price:,.2f}")
        print(f"  ✓ End Time: {qm.end_time}")
        print(f"  ✓ Resolved: {qm.resolved}\n")
        market_id = qm.market_id
    except Exception as e:
        print(f"  ✗ ERROR: {e}\n")
        return False
    
    # Test 3: Get orderbook
    print("[3/3] Fetching orderbook...")
    try:
        ob = adapter._rest_client.get_orderbook(market_id)
        print(f"  ✓ Last update: {ob.last_update}")
        
        if ob.bids:
            best_bid = ob.bids[0]
            print(f"  ✓ Best Bid: {best_bid.price / 10000:.2f}% ({best_bid.size / 1_000_000:.2f} shares)")
        else:
            print("  ℹ No bids")
            
        if ob.asks:
            best_ask = ob.asks[0]
            print(f"  ✓ Best Ask: {best_ask.price / 10000:.2f}% ({best_ask.size / 1_000_000:.2f} shares)")
        else:
            print("  ℹ No asks")
            
        print(f"  ✓ Total: {len(ob.bids)} bids, {len(ob.asks)} asks\n")
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}\n")
        return False
    
    return True


async def test_websocket(adapter: TurbineAdapter) -> bool:
    """Test WebSocket connectivity and message reception.
    
    Args:
        adapter: Initialized TurbineAdapter instance.
        
    Returns:
        True if test passes, False otherwise.
    """
    print("\n=== WebSocket Streaming Test ===\n")
    
    try:
        # Get a market to subscribe to
        qm = adapter.get_quick_market("BTC")
        market_id = qm.market_id
        
        print(f"Connecting to WebSocket...")
        await adapter.connect()
        print("  ✓ Connected\n")
        
        print(f"Subscribing to market {market_id[:16]}...")
        await adapter.subscribe_markets([market_id])
        print("  ✓ Subscribed\n")
        
        print("Waiting for 5 messages (10 second timeout)...")
        message_count = 0
        
        async def count_messages(msg):
            nonlocal message_count
            message_count += 1
            msg_type = getattr(msg, 'type', 'unknown')
            print(f"  [{message_count}] Received: {msg_type}")
        
        adapter.register_callback(count_messages)
        
        # Wait up to 10 seconds for messages
        await asyncio.sleep(10)
        
        if message_count >= 5:
            print(f"\n  ✓ Received {message_count} messages\n")
            return True
        elif message_count > 0:
            print(f"\n  ⚠ Only received {message_count}/5 messages\n")
            return True  # Partial success counts as pass
        else:
            print(f"\n  ✗ No messages received\n")
            return False
            
    except Exception as e:
        print(f"  ✗ ERROR: {e}\n")
        return False
    finally:
        # Clean up
        try:
            await adapter.close()
        except:
            pass


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Turbine Integration Smoke Test')
    parser.add_argument('--ws-test', action='store_true',
                       help='Include WebSocket streaming test')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}...")
    try:
        config = load_config(args.config)
        print("  ✓ Config loaded\n")
    except Exception as e:
        print(f"  ✗ ERROR loading config: {e}\n")
        return 1
    
    # Create adapter
    print("Initializing TurbineAdapter...")
    try:
        adapter = TurbineAdapter(config)
        print("  ✓ Adapter initialized\n")
    except Exception as e:
        print(f"  ✗ ERROR: {e}\n")
        return 1
    
    # Run basic tests
    if not test_basic_connectivity(adapter):
        print("=== FAILED: Basic connectivity tests failed ===\n")
        adapter._rest_client.close()
        return 1
    
    # Run WebSocket test if requested
    if args.ws_test:
        if not await test_websocket(adapter):
            print("=== FAILED: WebSocket test failed ===\n")
            return 1
    
    # Cleanup
    adapter._rest_client.close()
    
    print("=== ✓ ALL TESTS PASSED ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
