"""Read-only connectivity probe for Turbine API.

This script verifies basic connectivity to the Turbine API without requiring
authentication. It fetches the current BTC quick market and its orderbook.

Usage:
    python -m src.tools.connectivity_probe
"""
import sys
from turbine_client import TurbineClient


def main():
    """Run the connectivity probe."""
    client = TurbineClient(
        host="https://api.turbinefi.com",
        chain_id=137,  # Polygon mainnet
    )
    
    print("=== Turbine Connectivity Probe ===\n")
    
    # Health check
    try:
        health = client.get_health()
        print(f"API Health: {health}\n")
    except Exception as e:
        print(f"ERROR checking API health: {e}\n")
    
    # Fetch BTC quick market
    try:
        qm = client.get_quick_market("BTC")
        print(f"BTC Quick Market:")
        print(f"  Market ID: {qm.market_id}")
        print(f"  Strike Price: ${qm.start_price / 1e8:,.2f}")
        print(f"  End Time: {qm.end_time}")
        print(f"  Resolved: {qm.resolved}\n")
        
        # Fetch orderbook for this market
        ob = client.get_orderbook(qm.market_id)
        print(f"Orderbook (last update: {ob.last_update}):")
        
        if ob.bids:
            best_bid = ob.bids[0]
            print(f"  Best Bid: {best_bid.price / 10000:.2f}% ({best_bid.size / 1_000_000:.2f} shares)")
        else:
            print("  No bids")
            
        if ob.asks:
            best_ask = ob.asks[0]
            print(f"  Best Ask: {best_ask.price / 10000:.2f}% ({best_ask.size / 1_000_000:.2f} shares)")
        else:
            print("  No asks")
            
        print(f"  Total Bids: {len(ob.bids)}")
        print(f"  Total Asks: {len(ob.asks)}\n")
        
    except Exception as e:
        print(f"ERROR fetching BTC market: {e}")
        client.close()
        return 1
    
    # Fetch all markets
    try:
        markets = client.get_markets()
        print(f"Total Markets: {len(markets)}")
        if markets:
            print(f"Sample Market: {markets[0].question}")
    except Exception as e:
        print(f"ERROR fetching markets: {e}")
    
    client.close()
    print("\n=== Probe Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
