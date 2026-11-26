# Model cards (short form)

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
- **Metrics (60-day temporal holdout)**: NDCG@10 0.199, HitRate@10 0.46 vs co-basket 0.177/0.40 and popularity 0.122/0.30.
- **Scaling**: sparse co-occurrence and transition matrices; category affinity and content rows computed on demand. With large catalogs, item-level data gets thin (thousands of SKUs, each sold only a few times), so the model backs off to *learned* category affinity × the item's popularity within its category. It also reports category-level hit rate and catalog coverage.
- **Promotion gate**: a paired-bootstrap 95% CI of NDCG(hybrid − popularity). The model is promoted unless it is *significantly worse* than the baseline. Point estimates are noisy at around 2% hit rates.
- **Caveats**: tuned and reported on the same split; production needs a validation window plus a test window, then online A/B (CTR, attach rate, revenue per session). Prior/content terms help cold start, which this offline metric doesn't reward.

## fraud: order risk (IsolationForest)
- **Features**: log total, lines, max qty, account age, night order, ship/profile city mismatch, card payment, prior orders, high-value category share, max unit price.
- **Output**: percentile versus the training score distribution. The top 1% (≥ 0.99) goes to manual review with rule-based reasons.
- **Metrics**: ROC-AUC 0.97, PR-AUC 0.24 at a 0.8% base rate (≈30× random); alert precision about 0.26 and recall about 0.34 at the 1% budget.
- **Caveats**: labels are synthetic. Next step is semi-supervised: train a gradient-boosted classifier on confirmed chargebacks and use the IsolationForest score as a feature. Tune the alert budget to analyst capacity. Fairness checks are in RESPONSIBLE_AI.md.

## forecast: daily revenue
- **Model**: ridge regression on log revenue with trend, weekday dummies, annual Fourier terms, and event flags (White Friday, Ramadan, back-to-school).
- **Metrics (28-day backtest)**: weekly WAPE 0.15 vs seasonal-naive 0.31. Daily WAPE is noisy (0.51) because B2B bulk orders are lumpy.
- **Next**: forecast B2C and B2B separately, add quantile regression for proper intervals, and forecast at category level for inventory.
