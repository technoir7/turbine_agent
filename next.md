# Next Steps & Future Plans

## ~~Phase 1: Integration & Connectivity~~ ✅ COMPLETE
## ~~Phase 2: Strategy Doctrine Implementation~~ ✅ COMPLETE

**Accomplished:**
- ✅ Inventory-aware market making with skew logic
- ✅ Extremes risk control (widen spread + reduce size near 0/1)
- ✅ BTC quick market auto-rollover (polling every 10s)
- ✅ All unit tests pass (5/5)

---

## Phase 3: Tuning & Optimization

### 3.1 Spread Optimization
**Goal**: Find profitable spread settings for BTC quick markets.

**Tasks**:
- Run bot in live mode with tiny sizes (1-5 shares)
- Monitor fill rates and realized spreads
- Adjust `base_spread` based on market volatility
- Test extremes widening effectiveness

### 3.2 WebSocket Instant Rollover
**Goal**: Reduce rollover latency from 10s polling to instant detection.

**Implementation**:
- Add `subscribe_quick_markets("BTC")` in Supervisor startup
- Listen for `quick_market` WS message type
- Trigger rollover immediately on market change event
- Fallback to polling if WS disconnects

### 3.3 PnL Tracking & Metrics
**Goal**: Add real-time profit/loss monitoring.

**Tasks**:
- Track realized PnL from fills
- Track unrealized PnL from positions
- Log hourly PnL snapshots
- Add Prometheus metrics (optional)

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
