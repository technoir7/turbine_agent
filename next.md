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

### 3.0 Spike Bot Evaluation
**Goal**: Evaluate stability of the single-file `spike_bot.py` implementation.
**Tasks**:
- Run `spike_bot.py` in production for 24h.
- Compare fill rates and error logs vs multi-module bot.
- If successful, deprecate multi-module architecture or refactor into it.

### 3.0 Evaluate Expiry-Time (T-minus) Flattening Logic
**Goal**: Determine if bot should reduce inventory and widen spreads as market approaches expiry.

**Rationale**: As 15-minute markets approach expiration, liquidity may thin and resolution risk increases.

**Consideration**: Evaluate whether to implement T-minus logic or rely on rollover + extremes control.

### 3.1 Event Translation Layer (IMMEDIATE)
**Goal**: Connect WebSocket messages to state updates.

**Status**: ✅ **COMPLETE** (2026-01-30)

**Current State** (2026-01-30):
- ✅ WebSocket connects successfully
- ✅ Messages parsed and translated to `BookSnapshotEvent` and `TradeEvent`
- ✅ Supervisor consumes events via `on_event`
- ✅ End-to-end flow verified via `--event-test`

**Implementation**:
- In `TurbineAdapter._process_ws_messages()`:
  - `type="orderbook"` → `BookSnapshotEvent` (via `_translate_to_internal_events`)
  - `type="trade"` → `TradeEvent`


### 3.2 Spread Optimization
**Goal**: Find profitable spread settings for BTC quick markets.

**Status**: ✅ **COMPLETE** (2026-01-31)
- ✅ Widened `base_spread` to 0.40 for safety.
- ✅ Verified maker-only behavior (avoiding "matched immediately" logs).

### 3.3 WebSocket Instant Rollover
**Goal**: Reduce rollover latency from 10s polling to instant detection.

**Status**: **UNBLOCKED** (2026-01-30)
- WebSocket subscription working correctly
- Message reception confirmed
- Event translation layer active
- Ready to implement instant rollover

**Implementation**:
- Subscribe to `quick_market` message type (check if exists in official client)
- Listen for market change events in `_process_ws_messages()`
- Trigger rollover immediately on market change event
- Fallback to polling if WS disconnects

### 3.4 Feed Safety & Resync
**Goal**: Safety gates for feed freshness and state consistency.

**Status**: ✅ **COMPLETE** (2026-01-30)
- ✅ Freshness Gate (max age 30s)
- ✅ Adapter-Level Firewall (blocks network calls if stale)
- ✅ State Resync (on 404 cancel)
- ✅ Web3 dependency handling

### 3.5 PnL Tracking & Metrics
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

**Status**: ✅ **COMPLETE** (2026-01-31)
- ✅ Integrated into `official_spike_bot.py`.
- ✅ Market tracking persists across switches.
- ✅ Claim monitor runs every 30s.

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
- ✅ WebSocket connection and reliability hardened
- ✅ USDC permit signatures implemented for gasless trading
- ✅ Position awareness verified via polling loop
- ✅ Automatic Winnings Claiming active
- ✅ Automatic Market Rollover active
- ✅ Rate limit mitigation (2s tick + inter-request delays)

### Open Questions
- **Native User Fills**: Currently relying on HTTP polling (5s for positions, 10s for bot internal state) for position updates. This is stable but polling-based.
- **Order Modification**: Evaluating if `modify_order` (if supported) is better than `cancel + place`.
