import heapq
from typing import Dict, List, Optional, Tuple
from .events import Side, OrderStatus
import time

class OrderBook:
    def __init__(self, market_id: str):
        self.market_id = market_id
        self.bids: Dict[float, float] = {} # price -> size
        self.asks: Dict[float, float] = {}
        self.last_seq = 0
        self.last_update_ts = 0.0

    def apply_delta(self, seq: int, side: Side, price: float, size: float):
        if seq < self.last_seq:
            # Ignore stale (but allow same sequence for batch updates)
            return
        if seq > self.last_seq + 1 and self.last_seq != 0:
            # Gap detected - in real impl, this raises exception/flag
            pass 
        
        self.last_seq = seq
        self.last_update_ts = time.time()
        
        target = self.bids if side == Side.BID else self.asks
        
        if size <= 0:
            target.pop(price, None)
        else:
            target[price] = size

    def apply_snapshot(self, seq: int, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Apply a full orderbook snapshot, replacing existing state."""
        if seq < self.last_seq:
            return # Ignore stale
        
        self.last_seq = seq
        self.last_update_ts = time.time()
        
        # Replace maps
        self.bids = {price: size for price, size in bids if size > 0}
        self.asks = {price: size for price, size in asks if size > 0}

    def get_best_bid(self) -> Optional[Tuple[float, float]]:
        if not self.bids: return None
        best_price = max(self.bids.keys())
        return (best_price, self.bids[best_price])

    def get_best_ask(self) -> Optional[Tuple[float, float]]:
        if not self.asks: return None
        best_price = min(self.asks.keys())
        return (best_price, self.asks[best_price])

    def get_mid(self) -> Optional[float]:
        bb = self.get_best_bid()
        ba = self.get_best_ask()
        if bb and ba:
            return (bb[0] + ba[0]) / 2.0
        return None

    def get_imbalance(self, depth_n: int) -> float:
        # Sum top N sizes
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:depth_n]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth_n]
        
        bid_depth = sum(s for p, s in sorted_bids)
        ask_depth = sum(s for p, s in sorted_asks)
        
        return bid_depth / (ask_depth + 1e-9)

class Order:
    def __init__(self, client_order_id: str, market_id: str, side: Side, price: float, size: float):
        self.client_order_id = client_order_id
        self.market_id = market_id
        self.side = side
        self.price = price
        self.size = size
        self.filled_size = 0.0
        self.status = OrderStatus.PENDING_ACK
        self.created_ts = time.time()
        self.last_ack_ts = 0.0
        self.exchange_order_id: Optional[str] = None

class StateStore:
    def __init__(self):
        self.orderbooks: Dict[str, OrderBook] = {}
        self.orders: Dict[str, Order] = {} # client_order_id -> Order
        self.positions: Dict[str, float] = {} # market_id -> net_qty
        self.cash = 0.0
        self.realized_pnl = 0.0
        
    def get_orderbook(self, market_id: str) -> OrderBook:
        if market_id not in self.orderbooks:
            self.orderbooks[market_id] = OrderBook(market_id)
        return self.orderbooks[market_id]

    def on_fill(self, market_id: str, size: float, price: float, side: Side, fee: float):
        qty = size if side == Side.BID else -size
        
        # PnL tracking (Approximation via avg cost could go here, 
        # but spec asks for realized PnL on fills. 
        # Simple realization logic: if closing, realize diff.)
        current_pos = self.positions.get(market_id, 0.0)
        
        # Very simple Cash accounting
        cost = size * price
        if side == Side.BID:
            self.cash -= (cost + fee)
        else:
            self.cash += (cost - fee)
            
        self.positions[market_id] = current_pos + qty
        
        # Realized PnL logic could be more complex (FIFO/LIFO), 
        # for now we track total cash change as rough proxy + separate metric
        # Strictly speaking, realized PnL updates happen when checking diff vs book cost
