import logging
from typing import Dict, Optional, Tuple
from ..core.state import StateStore
from ..core.events import Side

logger = logging.getLogger(__name__)

class StrategyEngine:
    """
    Computes DESIRED quotes based on market state and config.
    """
    def __init__(self, config: Dict, state: StateStore):
        self.config = config # Store full config
        self.strategy_config = config['strategy']
        self.risk_config = config['risk']
        self.state = state

    def get_desired_quotes(self, market_id: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns (bid_price, ask_price) or (None, None) if no quote.
        """
        book = self.state.get_orderbook(market_id)
        mid = book.get_mid()
        
        if not mid:
            return None, None
            
        # 1. Fair Price
        fair = mid  # + EMA logic if implemented
        
        # 2. Base Spread
        base_spread = self.strategy_config['base_spread']
        base_half_spread = base_spread / 2.0
        
        # 3. Extremes Risk Control (applied to spread BEFORE skew/overlay)
        extreme_low = self.strategy_config.get('extreme_low', 0.10)
        extreme_high = self.strategy_config.get('extreme_high', 0.90)
        extreme_spread_mult = self.strategy_config.get('extreme_spread_mult', 2.0)
        
        in_extreme_zone = fair < extreme_low or fair > extreme_high
        
        if in_extreme_zone:
            widened_half_spread = base_half_spread * extreme_spread_mult
            logger.debug(f"Extremes zone: fair={fair:.3f}, widening spread by {extreme_spread_mult}x")
        else:
            widened_half_spread = base_half_spread
        
        # 4. Preliminary bid/ask around FAIR (with widened spread if extreme)
        preliminary_bid = fair - widened_half_spread
        preliminary_ask = fair + widened_half_spread
        
        # 5. Inventory Skew
        # skew = -1 * (pos / max_pos) * skew_factor
        # If long, bias down (lower bid, lower ask) to sell
        # If short, bias up (higher bid, higher ask) to buy
        pos = self.state.positions.get(market_id, 0.0)
        max_pos = self.risk_config.get('max_inventory_units', 1000.0)
        skew_factor = self.strategy_config['skew_factor']
        
        skew = -1 * (pos / max_pos) * skew_factor
        
        # 6. Imbalance Overlay
        imbalance = book.get_imbalance(self.strategy_config['imbalance_depth_n'])
        overlay = 0.0
        
        threshold = self.strategy_config['imbalance_threshold']  # e.g. 2.0
        bias = self.strategy_config['overlay_bias']
        
        if imbalance > threshold:
             overlay = -bias
        elif imbalance < (1.0/threshold):
             overlay = bias
             
        # 7. Apply skew and overlay to preliminary quotes
        final_bid = preliminary_bid + skew + overlay
        final_ask = preliminary_ask + skew + overlay
        
        # 8. Clamp to min/max price
        min_p = self.strategy_config['min_price']
        max_p = self.strategy_config['max_price']
        
        final_bid = max(min_p, min(final_bid, max_p))
        final_ask = max(min_p, min(final_ask, max_p))
        
        # 9. Sanity check
        if final_bid >= final_ask:
            # Spread crossed due to skew/overlay - back off
            return None, None
            
        return final_bid, final_ask

    def is_extreme_zone(self, market_id: str) -> bool:
        """Check if current fair price is in extreme zone (near 0 or 1).
        
        Args:
            market_id: Market ID to check
            
        Returns:
            True if mid price is below extreme_low or above extreme_high
        """
        book = self.state.get_orderbook(market_id)
        mid = book.get_mid()
        if not mid:
            return False
        
        extreme_low = self.strategy_config.get('extreme_low', 0.10)
        extreme_high = self.strategy_config.get('extreme_high', 0.90)
        return mid < extreme_low or mid > extreme_high
