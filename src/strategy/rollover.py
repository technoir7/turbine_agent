"""BTC Quick Market Rollover Manager.

Monitors the active BTC 15-minute quick market and detects when it changes,
signaling the need to rollover to the new market.
"""
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class RolloverManager:
    """Manages automatic rollover between BTC 15-minute quick markets."""
    
    def __init__(self, adapter, config):
        """Initialize the rollover manager.
        
        Args:
            adapter: Exchange adapter with _rest_client.get_quick_market() method
            config: Bot configuration dictionary
        """
        self.adapter = adapter
        self.config = config
        self.current_market_id: Optional[str] = None
        self.poll_interval = config.get('rollover', {}).get('poll_interval', 10.0)
        self._task = None
        self._running = False
        
    async def start(self):
        """Start the rollover monitoring task."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("RolloverManager: Started")
        
    async def stop(self):
        """Stop the rollover monitoring task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("RolloverManager: Stopped")
        
    async def _monitor_loop(self):
        """Poll for BTC quick market changes."""
        while self._running:
            try:
                # Fetch current active BTC quick market
                quick_market = self.adapter._rest_client.get_quick_market("BTC")
                new_market_id = quick_market.market_id
                
                if self.current_market_id is None:
                    # First detection
                    self.current_market_id = new_market_id
                    logger.info(f"RolloverManager: Initial market {new_market_id}")
                    
                elif new_market_id != self.current_market_id:
                    logger.warning(
                        f"RolloverManager: Market changed! "
                        f"{self.current_market_id} -> {new_market_id}"
                    )
                    self.current_market_id = new_market_id
                    # Signal will be handled by Supervisor
                    
            except Exception as e:
                logger.error(f"RolloverManager: Poll error: {e}")
                
            await asyncio.sleep(self.poll_interval)
            
    def get_current_market(self) -> Optional[str]:
        """Get the currently active BTC quick market ID.
        
        Returns:
            Market ID string or None if not yet detected
        """
        return self.current_market_id
        
    def has_changed(self, last_known: str) -> bool:
        """Check if market has changed since last known.
        
        Args:
            last_known: Previously known market ID
            
        Returns:
            True if current market is different from last_known
        """
        return self.current_market_id is not None and self.current_market_id != last_known
