# Next Steps & Future Plans

## ~~Phase 1: Integration & Connectivity~~ ✅ COMPLETE

The turbine-py-client is now fully integrated into the TurbineAdapter with fail-closed authentication.

**Completed:**
- ✅ Refactored `src/exchange/turbine.py` with verified API calls
- ✅ Created read-only connectivity probe (`src/tools/connectivity_probe.py`)
- ✅ Added fail-closed auth validation tests
- ✅ Updated all documentation (RUNBOOK, README)
- ✅ Verified against live API (1031 markets, BTC orderbook functional)

---

## Phase 2: Live Trading Validation

### 2.1 API Credentials Setup
**Goal**: Register for Turbine API credentials using your wallet.

**Action**:
```bash
# In Python REPL or script
from turbine_client import TurbineClient

credentials = TurbineClient.request_api_credentials(
    host="https://api.turbinefi.com",
    private_key="your_wallet_private_key",
)

# Save these to .env:
# TURBINE_API_KEY_ID=...
# TURBINE_API_PRIVATE_KEY=...
```

> [!CAUTION]
> The API private key is shown **only once**. Save it immediately to `.env`.

### 2.2 Minimal Order Lifecycle Test
**Goal**: Place a tiny order (1 share at extreme price) and immediately cancel it.

**Validation**:
1. Order successfully posts to Turbine
2. Order appears in your open orders
3. Cancel succeeds
4. No funds lost (order was at extreme price, no fill)

**Script**: Create `src/tools/minimal_order_test.py` (requires user approval to run)

---

## Phase 3: Strategy Integration

### 3.1 WebSocket Event Bridging
**Current State**: TurbineAdapter processes WS messages but doesn't bridge them to the internal state machine yet.

**Action**:
- Map turbine_client `OrderBookUpdate` → `src/core/events.MarketDataUpdate`
- Wire the adapter's `_process_ws_messages()` to call `StateStore.apply_event()`

### 3.2 Market Configuration
**Goal**: Configure which BTC 15-minute Quick Markets to trade.

**Action**:
- Update `config.yaml` to specify market selection criteria
- Implement automatic market rotation (as 15-min markets expire)

### 3.3 Inventory Skew with Real Positions
**Goal**: Use `get_positions()` to seed initial inventory state.

**Action**:
- On startup, fetch positions from Turbine
- Initialize `StateStore.positions` with live data
- Ensure skew logic accounts for existing positions

---

## Phase 4: Advanced Features

### 4.1 USDC Permit Signing
**Goal**: Enable gasless order execution by signing USDC permits.

**Status**: turbine-py-client supports `sign_usdc_permit()` but not yet wired into `place_order()`.

**Action**:
- Modify `TurbineAdapter.place_order()` to:
  1. Calculate required USDC collateral
  2. Call `self._rest_client.sign_usdc_permit(value=collateral_amount)`
  3. Attach `permit_signature` to `SignedOrder` before `post_order()`

### 4.2 Winnings Auto-Claim
**Goal**: Automatically claim winnings from resolved markets.

**Action**:
- Track markets we've traded in
- Poll for resolution status
- Call `self._rest_client.claim_winnings(market_contract_address)` when resolved

### 4.3 Multi-Market Liquidity
**Goal**: Run the bot across multiple BTC quick markets simultaneously.

**Challenges**:
- Risk limits need to be portfolio-wide
- Capital allocation needs balancing logic

---

## Phase 5: Production Hardening

### 5.1 Logging & Monitoring
- Structured JSON logs for trade execution
- Prometheus metrics for fill ratios, PnL, latency
- Alert on fail-closed events

### 5.2 Graceful Shutdown
- Cancel all orders on SIGTERM
- Persist state to disk for restart continuity

### 5.3 Backtesting Harness
- Record live orderbook snapshots
- Replay for strategy tuning
