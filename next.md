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
- ✅ Messages received and logged
- ✅ Messages translated and applied to state (verified via Supervisor logs)

**Implementation**:
- In `TurbineAdapter._process_ws_messages()`, parse `WSMessage` objects BEFORE dispatching:
  - `type="orderbook"` + `data` → Create `BookDeltaEvent` from orderbook snapshot
  - `type="trade"` + `data` → Create `TradeEvent` with fill details
  - Use exact field mappings from official client's `WSMessage` types
- Only then dispatch translated events to supervisor callbacks
- No invented fields - mirror official client exactly

### 3.2 Spread Optimization
**Goal**: Find profitable spread settings for BTC quick markets.

**Tasks**:
- Run bot in live mode with tiny sizes (1-5 shares)
- Monitor fill rates and realized spreads
- Adjust `base_spread` based on market volatility
- Test extremes widening effectiveness

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
- ✅ WebSocket connection working correctly
- ✅ WebSocket message reception verified (Strict Probe passed 2026-01-30)
- ✅ WebSocket reliability hardened (Keepalive + Watchdog + Auto-Reconnect)
- ✅ WebSocket logging productionized
- ✅ USDC permit signatures implemented and attached to all orders
- ✅ WebSocket subscribe pattern STRICTLY aligned with `turbine-py-client` examples
- ✅ Quick market rollover support via `get_quick_market("BTC")`
- ✅ Price/size decimal conversions verified
- ✅ Settlement addresses fetched from `get_markets()` and cached
- ✅ State Reconciliation: Periodic "ADAPTER TICK" logs authoritative position/order counts
- ✅ Trade Verification: `connectivity_probe.py --trade-test` proves ability to place/verify/cancel orders
- ✅ Robust State Parsing: Handles NoneType positions and 404 cancels gracefully
- ✅ WS Message Filtering: Tracks messages per target market to ensure relevant feed freshness

### Open Questions
- Need to implement full event translation layer to connect WS messages to state updates (partially done in Adapter now, need Supervisor hookup)
- Quick market instant rollover: `subscribe_quick_markets()` does NOT exist in installed client (verified via Probe failure). Must stick to polling or use `subscribe` on specific markets.
- WS User Fills: Currently relying on HTTP polling (5s loop) for position updates. Need to verify if `trade` event covers own trades effectively or if specific `fill` event exists.
