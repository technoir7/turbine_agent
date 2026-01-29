# Turbine Trading Agent Runbook

## 1. Setup

### Requirements
- Python 3.11+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  Note: This project uses `src` layout. Ensure `.` is in PYTHONPATH or install in editable mode.
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  ```

### Configuration
Edit `config.yaml` to set your credentials and risk limits.
**CRITICAL**: You must replace all `UNKNOWN` values in `config.yaml` before connecting to the real Turbine exchange.

Env Var Overrides supported:
```bash
export TURBINE_EXCHANGE_API_KEY="my_key"
```

## 2. Running Verification

Run the test suite to verify logic and simulation:
```bash
python -m unittest discover tests
```

## 3. Running the Agent

### Simulation Mode (Risk Free)
Runs against the internal random-walk match engine.
```bash
python src/main.py --simulated
```

## Connectivity Probe

### Verify Turbine API Access (Read-Only)

Before running the bot in live mode, verify your connection to the Turbine API:

```bash
python -m src.tools.connectivity_probe
```

**Expected Output:**
- API health status
- Current BTC Quick Market details (strike price, expiration)
- Live orderbook snapshot (best bid/ask, depth)
- Total number of markets

This probe runs **read-only** and requires **no authentication**. If it fails, check your network connection or the Turbine API status.

## Live Trading Setup

### Prerequisites

To enable live trading on Turbine, you need three environment variables:

1. **TURBINE_PRIVATE_KEY**: Your Ethereum wallet private key (for signing orders)
2. **TURBINE_API_KEY_ID**: API key identifier
3. **TURBINE_API_PRIVATE_KEY**: Ed25519 private key for API authentication

> [!CAUTION]
> **Use a dedicated trading wallet with limited funds.** Never use your main wallet.

### Configuration

Edit your `.env` file (copy from `env.example` if needed):

```bash
TURBINE_PRIVATE_KEY=0x...your_private_key_here...
TURBINE_API_KEY_ID=...your_api_key_id...
TURBINE_API_PRIVATE_KEY=...your_api_private_key...
```

### Run Live Bot

```bash
python src/main.py
```

The bot will:
- Connect to Turbine on Polygon Mainnet (chain ID 137)
- Subscribe to configured markets via WebSocket
- Place/cancel orders according to your strategy
- **Fail closed** if any auth credentials are missing

## Safety Features

- **Fail-Closed Auth**: Trading methods will refuse to execute if credentials are incomplete
- **Read-Only Fallback**: Without auth, the adapter operates in read-only mode
- **Market Cache**: Settlement addresses are cached to reduce API calls

### Live Mode
**WARNING**: Real money trading. Ensure `config.yaml` is correct.
```bash
python src/main.py
```

## 4. Monitoring
- Logs are printed to stdout and `turbine_agent.log`.
- Monitor `Expected Fill` vs `Actual Fill` in logs.
- Watch for `Risk Rejection` warnings.

## 5. Fail-Safe Behavior
The agent is designed to fail-closed:
- If Websocket disconnects for > 5s: Cancels all orders.
- If Sequence Gap detected: Pauses and Resyncs (Simulated).
- If Volatility spikes: Pauses trading.
