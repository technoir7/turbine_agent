import asyncio
import logging
import signal
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing other modules
load_dotenv()

from src.supervisor import Supervisor

# Setup logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)

async def main():
    parser = argparse.ArgumentParser(description='Turbine Trading Agent')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--simulated', action='store_true', help='Run against simulated exchange')
    args = parser.parse_args()

    bot = Supervisor(args.config, simulated=args.simulated)

    # Signal Handling
    loop = asyncio.get_running_loop()
    stop_signal = asyncio.Event()

    def signal_handler():
        logging.info("Received exit signal")
        stop_signal.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Run bot
    bot_task = asyncio.create_task(bot.start())

    # Wait for signal
    await stop_signal.wait()
    
    # Shutdown
    await bot.stop()
    await bot_task

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
