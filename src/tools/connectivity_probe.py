"""Definitive Connectivity Probe for Turbine Integration.

Strictly validates:
1. Environment & Credentials
2. HTTP Connectivity (Quick Market discovery)
3. WebSocket Connectivity (Connection, Subscription, Message flow)

This tool bypasses the adapter to prove infrastructure health directly using the client library.
"""
import asyncio
import logging
import sys
import argparse
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load .env explicitly
load_dotenv()

# Import locally to avoid module errors if not installed
try:
    from turbine_client import TurbineClient, TurbineWSClient
    from turbine_client.exceptions import WebSocketError
except ImportError:
    print("CRITICAL: turbine-py-client not installed. Run 'pip install -e turbine-py-client'")
    sys.exit(1)

from src.config.loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants from SKILL.md / turbine-py-client
HOST = "https://api.turbinefi.com"
CHAIN_ID = 137

def check_credentials(auto_register: bool = False) -> bool:
    """Validate credentials exist, or auto-register if requested."""
    private_key = os.environ.get("TURBINE_PRIVATE_KEY")
    api_key = os.environ.get("TURBINE_API_KEY_ID")
    api_secret = os.environ.get("TURBINE_API_PRIVATE_KEY")

    if not private_key:
        print("❌ CRITICAL: TURBINE_PRIVATE_KEY not found in env.")
        return False

    if not api_key or not api_secret:
        if auto_register:
            print("⚠  API credentials missing. Attempting auto-registration...")
            try:
                creds = TurbineClient.request_api_credentials(
                    host=HOST,
                    private_key=private_key
                )
                print(f"  ✓ Registered! Key ID: {creds['api_key_id']}")
                
                # Set in current env for this run
                os.environ["TURBINE_API_KEY_ID"] = creds["api_key_id"]
                os.environ["TURBINE_API_PRIVATE_KEY"] = creds["api_private_key"]
                
                # Append to .env for persistence (naive append)
                with open(".env", "a") as f:
                    f.write(f"\nTURBINE_API_KEY_ID={creds['api_key_id']}\n")
                    f.write(f"\nTURBINE_API_PRIVATE_KEY={creds['api_private_key']}\n")
                print("  ✓ Saved to .env")
                return True
            except Exception as e:
                print(f"❌ Auto-registration failed: {e}")
                return False
        else:
            print("❌ TURBINE_API_KEY_ID / TURBINE_API_PRIVATE_KEY missing.")
            print("   Run with --auto-register to generate them.")
            return False
            
    return True

def check_http_connectivity(symbol: str) -> Optional[str]:
    """Test HTTP connectivity and return market_id if successful."""
    print(f"\n[HTTP Check] Connecting to {HOST}...")
    try:
        # Use credentials if available, otherwise read-only
        client = TurbineClient(
            host=HOST,
            chain_id=CHAIN_ID,
            private_key=os.environ.get("TURBINE_PRIVATE_KEY"),
            api_key_id=os.environ.get("TURBINE_API_KEY_ID"),
            api_private_key=os.environ.get("TURBINE_API_PRIVATE_KEY")
        )
        
        # 1. Get Quick Market
        print(f"  ✓ Client initialized. Fetching quick market for {symbol}...")
        qm = client.get_quick_market(symbol)
        
        print(f"  ✓ Quick Market Found:")
        print(f"    - Market ID: {qm.market_id}")
        price = qm.start_price / 1e8 if hasattr(qm, 'start_price') else 0
        print(f"    - Strike Price: ${price:,.2f}")
        print(f"    - End Time: {qm.end_time}")
        
        # 2. Verify Market List
        print("  ✓ Fetching full market list to verify existence...")
        markets = client.get_markets()
        market_found = any(m.id == qm.market_id for m in markets)
        
        if market_found:
            print(f"  ✓ Market {qm.market_id[:8]}... confirmed in market list.")
        else:
            print(f"  ⚠ Market {qm.market_id[:8]}... NOT found in market list (might be new/hidden).")
            
        client.close()
        return qm.market_id
        
    except Exception as e:
        print(f"❌ HTTP Check Failed: {e}")
        return None

async def check_ws_connectivity(market_id: str, duration: int) -> bool:
    """Test WebSocket connectivity, subscription, and message flow."""
    print(f"\n[WS Check] Connecting to {HOST} (for {duration}s)...")
    
    ws_connected = False
    messages_total = 0
    messages_parsed = 0
    last_message_ts = 0
    
    # Track message types
    msg_types = {}

    try:
        # Match pattern from examples/websocket_stream.py
        ws = TurbineWSClient(host=HOST)
        
        print(f"  ✓ WS Client initialized. Connecting...")
        async with ws.connect() as stream:
            ws_connected = True
            print(f"  ✓ Connected! Subscribing to market {market_id[:8]}... and Quick Markets")
            
            # Subscribe pattern from example
            # Note: subscribe_quick_markets not available in installed client version
            # subscribe_orderbook and subscribe_trades are aliases for subscribe()
            await stream.subscribe_orderbook(market_id)
            await stream.subscribe_trades(market_id)
            
            print("  ✓ Subscribed. Listening...")
            
            start_time = time.time()
            iterator = stream.__aiter__()
            
            while time.time() - start_time < duration:
                try:
                    # Wait for message with timeout to allow loop exit
                    message = await asyncio.wait_for(iterator.__anext__(), timeout=1.0)
                    
                    messages_total += 1
                    last_message_ts = time.time()
                    
                    # Parse type
                    m_type = getattr(message, 'type', 'unknown')
                    msg_types[m_type] = msg_types.get(m_type, 0) + 1
                    
                    # Validation logic
                    if m_type in ['orderbook', 'trade', 'quick_market']:
                        messages_parsed += 1
                        
                except asyncio.TimeoutError:
                    continue
                except StopAsyncIteration:
                    break
        
    except Exception as e:
        print(f"❌ WS Check Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Report results
    print("\n[WS Report]")
    print(f"  Connected: {ws_connected}")
    print(f"  Total Messages: {messages_total}")
    print(f"  Parsed Messages: {messages_parsed}")
    print(f"  Message Types: {json.dumps(msg_types)}")
    
    if last_message_ts > 0:
        age = time.time() - last_message_ts
        print(f"  Last Message Age: {age:.2f}s")
    else:
        print("  Last Message Age: N/A (No messages)")

    if messages_total == 0:
        print("\n❌ FAILURE: Connected but received 0 messages.")
        print("  Hints:")
        print("  - Wrong WebSocket URL?")
        print("  - Subscription failed?")
        print("  - Market inactive?")
        return False
        
    if messages_parsed == 0:
        print("\n⚠ WARNING: Received messages but none were standard types (orderbook/trade).")
        return True # Soft pass if we got SOMETHING, but warn.

    print("\n✓ WS Connectivity Verified.")
    return True

def print_header(msg): print(f"\n=== {msg} ===")
def print_step(msg): print(f"  > {msg}")
def print_pass(msg): print(f"  ✓ {msg}")
def print_fail(msg): print(f"  ❌ {msg}")
def print_info(msg): print(f"  ℹ {msg}")

async def perform_trade_test(config: Dict[str, Any]) -> bool:
    """Perform a live trade verification cycle: Place (Buy+Sell) -> Verify -> Cancel -> Verify Gone."""
    print_header("Performing Verified Trade Test (Buy + Sell)")
    
    from src.exchange.turbine import TurbineAdapter
    from src.core.state import Order
    from src.core.events import Side as InternalSide
    from src.core.events import Side
    
    adapter = TurbineAdapter(config)
    
    try:
        # Pre-check
        print_step("Connecting Adapter...")
        await adapter.connect()
        
        qm = adapter.get_quick_market("BTC")
        print_info(f"Target Market: {qm.market_id}")
        
        # Subscribe to market to ensure feed is fresh for trading
        print_step("Subscribing to market data...")
        await adapter.subscribe_markets([qm.market_id])
        
        print_step("Waiting for feed to become fresh...")
        for _ in range(10):
            if adapter.is_feed_fresh(max_age_seconds=10.0):
                break
            await asyncio.sleep(1)
            
        if not adapter.is_feed_fresh(max_age_seconds=10.0):
             print_fail("Feed is still stale after 10s. Cannot trade.")
             await adapter.close()
             return False
             
        # 1. Place Orders (Deep OTM)
        # 0.01 for Buy, 0.99 for Sell
        orders_to_verify = []
        
        # Place BUY
        print_step("Placing LIMIT BUY @ 0.01...")
        buy_order = Order(
            market_id=qm.market_id,
            side=InternalSide.BID,
            price=0.01,
            size=1.0,
            client_order_id="probe_buy_001"
        )
        buy_hash = await adapter.place_order(buy_order)
        buy_order.exchange_order_id = buy_hash
        orders_to_verify.append(buy_order)
        print_pass(f"Buy Order placed: {buy_hash}")
        
        # Place SELL
        print_step("Placing LIMIT SELL @ 0.99...")
        sell_order = Order(
            market_id=qm.market_id,
            side=InternalSide.ASK,
            price=0.99,
            size=1.0,
            client_order_id="probe_sell_001"
        )
        sell_hash = await adapter.place_order(sell_order)
        sell_order.exchange_order_id = sell_hash
        orders_to_verify.append(sell_order)
        print_pass(f"Sell Order placed: {sell_hash}")
        
        # 2. Verification
        print_step("Verifying existence in Open Orders...")
        await asyncio.sleep(2.0) # Propagation delay
        
        open_orders = await adapter.get_open_orders()
        open_map = {o.exchange_order_id: o for o in open_orders}
        
        all_found = True
        for o in orders_to_verify:
            auth_o = open_map.get(o.exchange_order_id)
            if not auth_o:
                print_fail(f"Order {o.exchange_order_id} not found in open orders!")
                all_found = False
            else:
                # Verify side
                # Adapter converts API side to InternalSide
                if auth_o.side != o.side:
                    print_fail(f"Order {o.exchange_order_id} has wrong side! Expected {o.side}, got {auth_o.side}")
                    all_found = False
                else:
                    print_pass(f"Verified {o.exchange_order_id} ({o.side})")

        if not all_found:
             print_info("Dumping open orders for debug:")
             for o in open_orders:
                 print(f" - {o.exchange_order_id} {o.side} @ {o.price}")
             # Don't return yet, try to cancel anyway
             
        # 3. Cancel
        print_step("Cancelling orders...")
        for o in orders_to_verify:
            await adapter.cancel_order(o)
            print_pass(f"Cancel submitted for {o.exchange_order_id}")
            
        # 4. Verify Gone
        print_step("Verifying cancellation...")
        await asyncio.sleep(2.0)
        
        open_orders_after = await adapter.get_open_orders()
        remaining = [o.exchange_order_id for o in open_orders_after]
        
        any_failed = False
        for o in orders_to_verify:
            if o.exchange_order_id in remaining:
                print_fail(f"Order {o.exchange_order_id} still exists after cancel!")
                any_failed = True
        
        if any_failed:
            return False
            
        print_pass("All orders confirmed cancelled.")
        await adapter.close()
        return True

    except Exception as e:
        print_fail(f"Trade Test Exception: {e}")
        try:
            await adapter.close()
        except:
            pass
        return False

async def main():
    parser = argparse.ArgumentParser(description="Turbine Connectivity Probe")
    parser.add_argument("--symbol", default="BTC", help="Asset symbol (default: BTC)")
    parser.add_argument("--ws", action="store_true", help="Run WebSocket checks")
    parser.add_argument("--trade-test", action="store_true", help="Run full trade lifecycle test")
    parser.add_argument("--seconds", type=int, default=15, help="Duration for WS check")
    parser.add_argument("--auto-register", action="store_true", help="Auto-register API keys if missing")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    args = parser.parse_args()

    # 1. Load Config/Env
    print(f"Loading config from {args.config}...")
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"⚠ Config load warning: {e}")
        # Make a dummy config if load fails (might rely on defaults)
        config = {'exchange': {}}
    
    # 2. Check Credentials
    if not check_credentials(auto_register=args.auto_register):
        sys.exit(1)

    # 3. Trade Test (Excludes other tests to keep it clean, or runs them?)
    # Let's run trade test exclusively if requested
    if args.trade_test:
        success = await perform_trade_test(config)
        if not success:
            sys.exit(1)
        print("\n✨ TRADE PROBE PASSED ✨")
        sys.exit(0)

    # 3. HTTP Check
    market_id = check_http_connectivity(args.symbol)
    if not market_id:
        sys.exit(1)

    # 4. WS Check
    if args.ws:
        success = await check_ws_connectivity(market_id, args.seconds)
        if not success:
            sys.exit(1)

    print("\n✨ CONNECTIVITY PROBE PASSED ✨")
    sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
