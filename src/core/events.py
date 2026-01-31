from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple

class Side(Enum):
    BID = "BID"
    ASK = "ASK"

class OrderStatus(Enum):
    PENDING_ACK = auto()
    OPEN = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    UNKNOWN = auto()

@dataclass
class BookDeltaEvent:
    seq: int
    market_id: str
    side: Side
    price: float
    size: float

@dataclass
class BookSnapshotEvent:
    seq: int
    market_id: str
    bids: List[Tuple[float, float]] # [(price, size), ...]
    asks: List[Tuple[float, float]]

@dataclass
class TradeEvent:
    ts: float
    market_id: str
    price: float
    size: float
    aggressor_side: Optional[Side] = None

@dataclass
class UserFillEvent:
    ts: float
    market_id: str
    client_order_id: str
    side: Side
    price: float
    size: float
    exchange_order_id: Optional[str] = None
    fee: float = 0.0

@dataclass
class OrderAckEvent:
    ts: float
    client_order_id: str
    status: OrderStatus
    reason: Optional[str] = None
    exchange_order_id: Optional[str] = None

@dataclass
class HeartbeatEvent:
    ts: float
    seq: int
