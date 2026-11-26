# Model cards (short form)

TechCart starts with no orders, customers or logs: every model has a day-one mode that needs no history and a trained mode that switches on behind a data gate. `artifacts/training_summary.json` and the admin Models panel report which mode each one is in.

## intent: query intent classifier (tier 1 of the cascade)
- **Model**: TF-IDF (word 1-2-grams + char_wb 2-5-grams) → Logistic Regression, isotonic/sigmoid calibrated (CV=3).
- **Labels**: product_search, comparison, recommendation, order_status, return_refund, policy_question, purchase, quote_request, other.
- **Data**: about 2.3k template-generated utterances with noise. The test set is 47 hand-written queries with typos, Arabic and slang.
- **Metrics**: macro-F1 0.86; accuracy when confident (p ≥ 0.55) 0.92 at 83% coverage. Below τ, the LLM tier decides.
- **Caveats**: the test set was consulted while improving the templates, so treat it as a dev set. The real holdout is labelled production traffic, collected from the low-confidence queue on the admin dashboard (active learning).
- **Monitoring**: intent-mix PSI, mean confidence, LLM-fallback rate.

## recommender: hybrid next-purchase recommender
- **Model**: item-item co-basket cosine + sequential transition probabilities (previous 3 baskets → next), with recency decay (90-day half-life), a category-complement prior and TF-IDF content similarity. Weights are chosen by grid search on a temporal split.
- **Business rules**: never recommend owned items, suppress durables bought in the last 180 days, in-stock only, at most 2 per category, consent-gated.
- **Cold start (< 300 orders)**: no co-basket or sequence signal exists, so the score is the category-complement prior (laptop → mice, bags, docks, headsets…) × the item's popularity inside its category, from real Amazon review counts, plus content similarity. Where a category has no basket evidence the prior gets full weight, so title similarity can't push an RTX desktop after an RTX laptop. Reasons say "Goes with your Laptops", never "customers who bought".
- **Trained mode (≥ 300 orders)**: learned affinities blend in proportionally to the basket evidence per category, weights are tuned on a temporal split, and NDCG@10 / HitRate@10 are reported against co-basket and popularity baselines.
- **Scaling**: sparse co-occurrence and transition matrices; category affinity and content rows computed on demand. With large catalogs, item-level data gets thin (thousands of SKUs, each sold only a few times), so the model backs off to *learned* category affinity × the item's popularity within its category. It also reports category-level hit rate and catalog coverage.
- **Promotion gate**: a paired-bootstrap 95% CI of NDCG(hybrid − popularity). The model is promoted unless it is *significantly worse* than the baseline. Point estimates are noisy at around 2% hit rates.
- **Caveats**: tuned and reported on the same split; production needs a validation window plus a test window, then online A/B (CTR, attach rate, revenue per session). Prior/content terms help cold start, which this offline metric doesn't reward.

## fraud: order risk (rules → rules + IsolationForest)
- **Day one**: every order is scored at checkout by transparent rules (new account < 2 days 0.30, 1–5 AM 0.15, ship city ≠ profile city 0.20, 3+ high-value units 0.30, basket > 20k SAR 0.20); ≥ 0.6 goes to the review queue. Score, reasons and scorer are stored on the order.
- **From 500 orders**: the IsolationForest below is trained on real orders and its percentile is combined with the rules.
- **Features**: log total, lines, max qty, account age, night order, ship/profile city mismatch, card payment, prior orders, high-value category share, max unit price.
- **Output**: percentile versus the training score distribution. The top 1% (≥ 0.99) goes to manual review with rule-based reasons.
- **Metrics**: measured once confirmed chargebacks exist (PR-AUC at the analyst's alert budget).
- **Caveats**: unsupervised until chargebacks are confirmed. Next step is semi-supervised: train a gradient-boosted classifier on confirmed chargebacks and use the IsolationForest score as a feature. Tune the alert budget to analyst capacity. Fairness checks are in RESPONSIBLE_AI.md.

## forecast: daily revenue
- **Gate**: hidden until 56 days of sales exist; then backtested on the last 28 days.
- **Model**: ridge regression on log revenue with trend, weekday dummies, annual Fourier terms, and event flags (White Friday, Ramadan, back-to-school).
- **Metrics (28-day backtest)**: weekly WAPE vs seasonal-naive, reported in the training summary once the gate opens.
- **Next**: forecast B2C and B2B separately, add quantile regression for proper intervals, and forecast at category level for inventory.
