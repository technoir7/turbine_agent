# Next Steps & Future Plans

## ~~Phase 1: Integration & Connectivity~~ ✅ COMPLETE
## ~~Phase 2: Strategy Doctrine Implementation~~ ✅ COMPLETE
## ~~Extremes Risk Control Ordering Fix~~ ✅ COMPLETE

**Accomplished:**
- ✅ Inventory-aware market making with skew logic
- ✅ Extremes risk control (widen spread around fair price BEFORE skew)
- ✅ BTC quick market auto-rollover (polling every 10s)
- ✅ All unit tests pass (5/5)

---

## Phase 3: Tuning & Optimization

### 3.0 Evaluate Expiry-Time (T-minus) Flattening Logic
**Goal**: Determine if bot should reduce inventory and widen spreads as market approaches expiry.

**Rationale**: As 15-minute markets approach expiration, liquidity may thin and resolution risk increases.

**Consideration**: Evaluate whether to implement T-minus logic or rely on rollover + extremes control.

### 3.1 Spread Optimization
**Goal**: Find profitable spread settings for BTC quick markets.

**Tasks**:
- Run bot in live mode with tiny sizes (1-5 shares)
- Monitor fill rates and realized spreads
- Adjust `base_spread` based on market volatility
- Test extremes widening effectiveness

### 3.2 WebSocket Instant Rollover
**Goal**: Reduce rollover latency from 10s polling to instant detection.

**Status**: ~~Blocked by WebSocket subscription issues~~ → **UNBLOCKED** (2026- 01-29)
- WebSocket subscription now working correctly после context manager fix
- Ready to implement `subscribe_quick_markets("BTC")`

**Implementation**:
- Add `subscribe_quick_markets("BTC")` in Supervisor startup (need to verify this method exists or use equivalent)
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

### 4.1 ~~USDC Permit Signing~~ ✅ COMPLETE
**Goal**: Enable gasless order execution by signing USDC permits.

**Status**: IMPLEMENTED (2026-01-28)

**Changes Made**:
- Modified `TurbineAdapter.place_order()` to:
  1. Calculate required USDC collateral based on order type:
     - BUY: `(size * price / 1e6) + fee + 10% margin`
     - SELL: `size + 10% margin`
  2. Call `self._rest_client.sign_usdc_permit(value=permit_amount, settlement_address=...)`
  3. Attach `permit_signature` to `SignedOrder` before `post_order()`
- All orders now include USDC permits for gasless execution
- Per SKILL.md integration requirements (lines 827-890)

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

---

## Integration Notes & Unknowns

### Verified Integration Details
- ✅ WebSocket connection and subscription working correctly (fixed 2026-01-29)
- ✅ USDC permit signatures implemented and attached to all orders
- ✅ WebSocket subscribe pattern verified against `turbine-py-client/examples/websocket_stream.py`
- ✅ Quick market rollover support via `get_quick_market("BTC")`
- ✅ Price/size decimal conversions: price scale 1e6 (but 10k for compat), size 6 decimals, strike 8 decimals
- ✅ Settlement addresses fetched from `get_markets()` and cached

### Open Questions
(No unknowns identified - all integration requirements sourced from turbine-py-client code and examples)
