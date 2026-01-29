import logging
import time
from typing import Dict
from ..core.state import StateStore, Order
from ..core.events import Side

logger = logging.getLogger(__name__)

class RiskEngine:
    """
    Gatekeeper for all trading actions.
    Enforces hard limits and circuit breakers.
    """
    def __init__(self, config: Dict, state: StateStore):
        self.config = config['risk']
        self.state = state
        self.soft_drawdown_mode = False
        self.triggered_circuit_breakers = set()

    def check_order_risk(self, order: Order) -> bool:
        """
        Check if a new order violates any risk limits.
        Returns True if safe, False if rejected.
        """
        # 1. Check Circuit Breakers
        if self.triggered_circuit_breakers:
             logger.warning("Risk Rejection: active circuit breakers: %s", self.triggered_circuit_breakers)
             return False

        # 2. Check Inventory Limits (post-trade approximation)
        current_pos = self.state.positions.get(order.market_id, 0.0)
        max_units = self.config['max_inventory_units']
        
        projected_pos = current_pos + order.size if order.side == Side.BID else current_pos - order.size
        
        if abs(projected_pos) > max_units:
             logger.warning("Risk Rejection: Inv Cap %s > %s", projected_pos, max_units)
             return False

        # 3. Check Portfolio Exposure
        # Simplified: sum of abs(pos)
        current_exposure = sum(abs(p) for p in self.state.positions.values())
        if current_exposure + order.size > self.config['max_portfolio_exposure']:
            logger.warning("Risk Rejection: Max Exposure")
            return False

        return True

    def check_market_updates(self, market_id: str):
        """
        Called on market data updates to check volatility/drawdown.
        """
        book = self.state.get_orderbook(market_id)
        mid = book.get_mid()
        if not mid: return

        # Volatility Guard logic would go here (need history tracking)
        # For this skeleton, we assume basic checks pass
        
        # Drawdown check
        # if max_pnl - current_pnl > limit: trigger hard stop
        pass

    def check_connectivity(self, last_heartbeat_ts: float):
        """
        Check for stale connection.
        """
        now = time.time()
        timeout = 5.0 # hardcoded guard
        if now - last_heartbeat_ts > timeout:
            logger.error("Risk Alert: WS Silence > %s s", timeout)
            self.triggered_circuit_breakers.add("WS_SILENCE")
            return False
        else:
            if "WS_SILENCE" in self.triggered_circuit_breakers:
                logger.info("Risk: WS Connectivity restored")
                self.triggered_circuit_breakers.discard("WS_SILENCE")
        return True
