
## 0) How to Use This Guide
- This document is designed as a print-ready cram sheet.
- It is tailored to your background: BlueCrest (FX vol surface + robot strategy), BAM (equity alt-data), AB (credit RV factors).
- Focus first on: Sections 2, 3, 5, 7, 8, 10.

---

## 1) Role Framing: What a Vol Overlay Team Does

### Core mission
A Vol Overlay team typically sits above individual books and optimizes **firm-level volatility risk** while seeking risk-adjusted alpha.

### In practical terms
1. Aggregate exposures across desks (delta/gamma/vega/vanna/volga/theta).
2. Decide target risk shape under regime constraints.
3. Execute overlays using liquid options/futures/swaps to:
   - neutralize undesired exposures,
   - reduce hedge cost,
   - opportunistically harvest relative-value in vol surface.

### One-line distinction
- **Vol arbitrage** = pure mispricing alpha.
- **Vol overlay** = portfolio-level risk shaping that may use vol-arb tools.

---

## 2) Product Scope: What Is Likely In Scope at  

## Most likely primary scope
1. **Listed equity index vol**: SPX/SPY/NDX/QQQ/RUT, VIX ecosystem.
2. **Listed rates vol**: Treasury options (e.g., TY/US futures options), SOFR options.
3. **Liquid listed commodity vol**: depending on desk overlap.

## Possible but less central
- Listed FX options (exchange-listed variants).
- Some cleared OTC-like products depending on org structure.

## Usually less likely for this role as core
- Bespoke OTC single-name CDS options.
- Highly customized bilateral OTC structures.

##  question-safe phrasing
"My assumption is this role centers on liquid listed vol complexes and firm-level risk aggregation; if there is OTC overlap, it is likely in standardized/cleared forms with strong operational controls."

---

## 3) Your Background Mapping: Why You Are Relevant

## A) BlueCrest FX (Otto + robot)
Transferable strengths:
- Internal fair-vol surface thinking.
- Relative-value in vol space (market IV vs internal fair IV).
- Greek-constrained portfolio construction.
- Optimizer-driven execution vs discretionary bias.
- Black-swan lesson (SNB 2015): tail risk, crowding, regime breaks.

## B) AB fixed income credit RV
Transferable strengths:
- Fair value modeling and residual/mean-reversion logic.
- Cross-sectional alpha construction and constraint-aware portfolioing.
- Data quality handling in fragmented markets.
- Productionized research discipline.

## C) BAM alternative data
Transferable strengths:
- PIT correctness (`valid_from`/`valid_to`, revision-aware pipelines).
- Leakage prevention and timestamp governance.
- Feature engineering discipline and model monitoring.

Non-transferable or less-direct parts:
- Consumer alt-data (credit card/app/jobs/flights/weather) is less direct for short-horizon vol overlay triggers.

How to frame:
"The datasets differ by horizon, but the research and production discipline transfers 1:1."

---

## 4) Greeks Refresher (with  question Intuition)

Let option value be $V(S,\sigma,t,...)$.

- **Delta**: $\Delta=\partial V/\partial S$
  - Directional sensitivity to spot.
- **Gamma**: $\Gamma=\partial^2 V/\partial S^2$
  - Convexity; sensitivity of delta to spot changes.
- **Vega**: $\nu=\partial V/\partial \sigma$
  - Sensitivity to implied vol level.
- **Theta**: $\Theta=\partial V/\partial t$
  - Time decay/carry.
- **Vanna**: $\partial\Delta/\partial\sigma = \partial\nu/\partial S$
  - How delta changes when vol changes (or vega changes when spot moves).
- **Volga/Vomma**: $\partial^2V/\partial\sigma^2$
  - Vega convexity to vol-of-vol shocks.

### Useful decomposition
For short horizon:
$$
\Delta P \approx \Theta\,\Delta t + \Delta\,\Delta S + \tfrac12\Gamma(\Delta S)^2 + \nu\,\Delta\sigma + ...
$$

###  question Q: "What happens to vanna if spot rallies 5%?"
Correct answer template:
- It depends on strike/moneyness and maturity.
- ATM call tends to have vanna near zero under standard assumptions.
- OTM/ITM options can have nonzero vanna; as spot moves moneyness shifts, so portfolio vanna profile changes materially.

---

## 5) "Long All Greeks" — What It Really Means

### Strictly speaking
In a single plain-vanilla option, you cannot free-lunch all exposures.

### What PM language often means in practice
- Long gamma where it matters.
- Theta not necessarily positive in pure option terms, but improved by structure/carry/relative-value.
- Neutral-to-controlled vega.
- Intentional second-order smile exposures (vanna/volga) via structure.

### Accurate phrasing
"It is not breaking Black-Scholes identities; it is shaping net exposures across strikes/tenors using relative mispricing and constraints."

---

## 6) Buy ATM / Sell Wings and Smile Premium

## Why this trade appears often
- Wings can embed insurance demand premium.
- Selling rich wings funds ATM convexity.

## Risk
- Short wings = short tail convexity.
- Quiet periods harvest carry; jumps can produce severe losses.

## Key lesson from SNB-style events
- Crowd can over-sell tails due to narrative certainty.
- A disciplined model may buy underpriced tails when market complacency is extreme.

##  question-safe nuance
"Buy-ATM/sell-wings is one possible optimizer output, not a permanent dogma. The engine should be agnostic and trade edge wherever it exists on the surface."

---

## 7) Robot Strategy: Objective + Constraints (How to Explain)

### Conceptual objective (important notation clarification)
`w_i` is usually **position size** (contracts/notional), not vega itself.

Two equivalent ways to write the objective:

1. **Contract-weight formulation**
$$
\max_w \sum_i w_i\Big[(IV_{fair,i}-IV_{mkt,i})\cdot Vega_i\Big]-\text{Costs}(w)
$$
Here, vega converts IV spread into approximate dollar edge per contract.

2. **Vega-notional formulation**
Define $x_i := w_i\cdot Vega_i$.
$$
\max_x \sum_i x_i\,(IV_{fair,i}-IV_{mkt,i})-\text{Costs}(x)
$$
In this form, decision variables are vega notionals.

So: weight is **not** automatically vega. Vega is usually a scaling/exposure term (unless you explicitly choose vega-notional variables).

### Typical constraints
- Delta target (often near zero).
- Gamma bounds.
- Vega bounds or neutrality band.
- Vanna/volga bounds.
- Liquidity/size/notional/turnover limits.
- Risk/capital/margin constraints.

### Typical constraints in math form (contract-weight variables $w_i$)
Let each instrument $i$ have per-contract Greeks and costs:
$\Delta_i,\Gamma_i,\nu_i,\text{Vanna}_i,\text{Volga}_i$, spread/liquidity metrics, and margin coefficient $m_i$.

1. **Delta neutrality / target band**
$$
\Delta^{L} \le \sum_i w_i\Delta_i \le \Delta^{U}
$$
Often $\Delta^L=-\varepsilon_\Delta,\ \Delta^U=+\varepsilon_\Delta$.

2. **Gamma exposure control**
$$
\Gamma^{L} \le \sum_i w_i\Gamma_i \le \Gamma^{U}
$$
For long-convexity overlays, typically $\Gamma^{L}>0$.

3. **Vega neutrality / band**
$$
\nu^{L} \le \sum_i w_i\nu_i \le \nu^{U}
$$
Strict neutral is $\nu^L=\nu^U=0$.

4. **Vanna / Volga limits**
$$
	ext{Vanna}^{L} \le \sum_i w_i\text{Vanna}_i \le \text{Vanna}^{U}
$$
$$
	ext{Volga}^{L} \le \sum_i w_i\text{Volga}_i \le \text{Volga}^{U}
$$

5. **Position bounds (per instrument)**
$$
w_i^{\min} \le w_i \le w_i^{\max}
$$

6. **Liquidity / participation cap**
$$
|w_i| \le \alpha_i\,ADV_i
$$
or spread-aware risk budget
$$
\sum_i |w_i|\,\text{SpreadCost}_i \le C_{\max}
$$

7. **Turnover constraint (rebalance from current holdings $w_i^{old}$)**
$$
\sum_i |w_i-w_i^{old}| \le TO_{\max}
$$

8. **Margin / capital budget**
$$
\sum_i m_i|w_i| \le M_{\max}
$$

9. **Optional integer/cardinality constraints (listed options reality)**
$$
w_i \in \mathbb{Z}
$$
and with binary selection variables $z_i\in\{0,1\}$,
$$
|w_i| \le U_i z_i,\quad \sum_i z_i \le K
$$
to limit number of active legs.

10. **Stress/scenario constraints (recommended in production)**
For each scenario $s\in\mathcal{S}$:
$$
	ext{PnL}_s(w) \ge -L_s
$$
which caps worst-case losses under pre-defined shocks.

### Why vega-neutral often used
To isolate **shape-relative** alpha (skew/term/wings) from broad parallel vol-level moves.

If using contract weights, vega neutrality is:
$$
\sum_i w_i\cdot Vega_i \approx 0
$$
not $\sum_i w_i=0$.

### Important correction
Vega-neutral does **not** mean guaranteed immunity to all vol moves.
- Surface moves are not parallel.
- Vol-of-vol and smile dynamics remain.
- Model and execution error remain.

---

## 8) System Design Round (Most Important for This Role)

This is not social-media system design. It is **trading data + modeling + risk + execution** design.

## What the  questioner likely optimizes for
1. Correctness under stress.
2. Latency where needed.
3. Failure-safe behavior.
4. Replayability/auditability.
5. Research-to-production consistency.

## Candidate architecture to describe
1. **Ingestion layer**
   - Exchange/venue feeds, normalization, schema contracts, clock sync.
2. **Quality layer**
   - stale/crossed/locked quote filters, dedupe, anomaly flags.
3. **State layer**
   - low-latency latest state store for tradable surface inputs.
4. **Model layer**
   - IV inversion, no-arb checks, surface fitting/interpolation.
5. **Optimization layer**
   - objective + constraints + fallback if infeasible.
6. **Risk gate**
   - hard limits and kill switch before order release.
7. **Execution + feedback**
   - order placement, fill attribution, slippage and inventory feedback.
8. **Historical lake/replay**
   - immutable event logs for deterministic replay and post-mortem.

## Data modeling entities
- instrument master (versioned mappings)
- market events (event-time + processing-time)
- quote snapshots
- surface snapshots (model version + quality metrics)
- greeks snapshots
- risk states
- decisions/optimizer outputs
- orders/fills/rejections

## Partitioning guidance
- Partition by date + underlying (+ expiry bucket).
- Avoid over-partitioning by strike.
- Keep hot state separate from long-term archive.

## PIT / bitemporal must-have
Store both business validity and system ingestion time.

## Failure scenarios to proactively mention
- Feed gap
- stale model
- optimizer infeasible
- venue outage
- bad instrument mapping
- stale risk snapshot

Expected behavior: degrade safely, reduce size, fallback hedge mode, preserve audit trail.

---

## 9) Kafka/Redis/KDB/Q Positioning (What to Say)

You do not need to claim KDB expertise to pass.

Strong framing:
- "I optimize for requirements, not brand loyalty in tech stack."
- Kafka/Redis can work well for streaming + state.
- Key is SLA, correctness, observability, and replay.

If asked about KDB/Q:
- Acknowledge it is widely used in trading due to columnar/time-series strengths.
- Emphasize your ability to integrate with existing stack and data contracts.

---

## 10) Data Exploration Round: What Good Looks Like

## First 5 steps on any quote dataset
1. Sanity-check schema and clocks.
2. Remove crossed/locked/zero/stale quotes.
3. Reconstruct tradable mids/spreads with flags.
4. Invert to implied vol.
5. Build simple edge features and validate no leakage.

## Good mini-signals
- market IV minus fair IV by moneyness/tenor bucket
- spread-normalized edge
- flow-pressure/imbalance-conditioned edge
- regime-conditioned edge (vol-of-vol/high-stress flags)

## What to avoid
- reporting only IC without implementation costs.
- no timestamp hygiene.
- no reproducibility path.

---

## 11) Alternative Data: What Is Relevant to Vol Overlay

## Most relevant classes
- options microstructure/flow
- dealer positioning proxies
- event intensity/surprise signals
- cross-asset stress/liquidity proxies

## Less direct but still useful
- fundamental consumer alt-data as event/jump priors and dispersion context.

## Your strongest pitch
"I know how to productionize PIT-safe feature pipelines and evaluate real tradability impact, then adapt feature sets to the correct decision horizon."

---

## 12) High-Probability  question Questions + Model Answers

## Q1: Vol overlay vs vol arbitrage?
A: Vol arb is pure mispricing alpha; overlay is firm-level risk shaping. Overlay may use vol-arb techniques to hedge cheaper or monetize dislocations while satisfying risk targets.

## Q2: Why vega-neutral constraints?
A: To reduce exposure to broad vol-level shifts and isolate relative shape views. But vega-neutral is not risk-free; smile dynamics and vol-of-vol still matter.

## Q3: Why can short-wing strategy lose with no spot move?
A: If implied vol rises, short vega hurts even when $\Delta S\approx0$. PnL includes $\nu\Delta\sigma$ and higher-order terms.

## Q4: Explain SNB lesson in one sentence.
A: Crowd sold tails on policy narrative; systematic fair-value discipline can identify underpriced convexity when consensus is overconfident.

## Q5: What would you modernize in legacy LP stack?
A: Keep objective/constraints logic, add stronger cost/risk realism (convex penalties, integer constraints where needed), robust fallback, and full replay/audit.

## Q6: How do you prevent data leakage?
A: strict PIT joins, bitemporal fields, lag-aware feature generation, immutable snapshots, replayable research pipeline.

## Q7: How do you judge signal quality?
A: out-of-sample stability, cost-adjusted PnL, exposure decomposition, regime robustness, and operational robustness.

## Q8: What is your system design priority order?
A: correctness -> risk safety -> observability -> latency optimization -> cost.

## Q9: How would you respond if model says trade but market is stressed?
A: enforce dynamic risk gates/regime-aware constraints; reduce size or switch to hedge-only mode; never bypass hard controls.

## Q10: Why are you a fit?
A: You combine vol-surface/optimizer experience (BlueCrest), large-scale alpha/research discipline (AB), and PIT alt-data engineering rigor (BAM).

---

## 13) Your Personal 90-Second Story (Memorize)

"I’ve worked across three complementary domains: vol-surface/optimizer trading at BlueCrest, systematic cross-sectional RV modeling in fixed income at AB, and PIT-safe alternative-data engineering from BAM-style workflows. What excites me about   Vol Overlay is the intersection of all three: high-quality real-time data engineering, robust fair-value surface modeling, and constrained optimization to shape firm-level risk. I’m especially motivated by building systems that remain correct under stress — deterministic replay, strict timestamp governance, and risk-safe degradation — so we can both protect capital and systematically capture dislocations."

---

## 14) Questions to Ask James Tong (High Signal)

1. "What is currently the biggest bottleneck: data quality, model stability, or execution cost?"
2. "How do you separate fast hedge loops from slower optimization loops operationally?"
3. "How do you monitor surface quality drift and model confidence intraday?"
4. "What are your hardest post-mortem cases in the last year, and what changed?"
5. "How much of overlay objective is risk transfer vs standalone alpha?"
6. "What are the required controls before an optimizer output can go live?"

---

## 15) Day-Before  question Checklist

- Rehearse 3 stories:
  1) SNB black swan lesson,
  2) Building/using fair vol surface + optimizer,
  3) PIT data-quality governance and leakage prevention.
- Be ready to draw pipeline: feed -> clean -> surface -> optimize -> risk gate -> execution -> replay.
- Be ready to explain one objective function and one constraints set in plain language.
- Be explicit about trade-offs and failure handling.

---

## 16) Red Flags to Avoid in Answers

- "vega-neutral means no vol risk" (incorrect).
- "latency is everything" (not true for all stages).
- "I only care about model IC" (must include implementation).
- "we override model by gut feel" (risky framing).
- "I used alt data so it must work here" (horizon mismatch).

---

## 17) Final  question Positioning

Your edge as a candidate:
- Not just quant theory.
- Not just data engineering.
- Not just research.

You can connect:
**market microstructure + PIT-safe data systems + optimizer/risk constraints + production controls**.

That combination is exactly what a Vol Overlay team needs.
