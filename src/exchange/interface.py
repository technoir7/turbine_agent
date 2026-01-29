from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable, Awaitable
from ..core.events import Side, OrderStatus
from ..core.state import Order

class ExchangeAdapter(ABC):
    """
    Abstract interface for exchange interactions.
    All external dependencies (REST/WS) are abstracted here.
    """
    
    @abstractmethod
    async def connect(self):
        """Connect to WS and authenticate."""
        pass

    @abstractmethod
    async def close(self):
        """Clean shutdown."""
        pass

    @abstractmethod
    async def subscribe_markets(self, market_ids: List[str]):
        """Subscribe to market data topics."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """
        Place order via REST/WS.
        Returns: exchange_order_id (if immediate) or raises Error.
        """
        pass

    @abstractmethod
    async def cancel_order(self, order: Order):
        """Cancel specific order."""
        pass
    
    @abstractmethod
    async def cancel_all(self, market_id: Optional[str] = None):
        """Cancel all open orders (safety switch)."""
        pass

    @abstractmethod
    async def get_positions(self) -> Dict[str, float]:
        """Fetch current positions snapshot."""
        pass
        
    @abstractmethod
    async def get_open_orders(self) -> List[Order]:
        """Fetch open orders snapshot."""
        pass

    @abstractmethod
    def register_callback(self, callback: Callable[[Any], Awaitable[None]]):
        """Register generic event callback for incoming WS messages."""
        pass
