# Report — AppleSupport Twitter Support Agent

## 1. Problem framing

**Brand chosen:** AppleSupport.

**What "good" means for this agent:** a message should be classified into the right intent, the reply should reflect how AppleSupport actually responds to that kind of issue (not a generic AI-sounding reply), and the escalation decision should match what a human reviewer would actually choose to hand off — not just flag anything that "sounds negative."

**What I deliberately chose not to build:**
- No multi-turn conversation handling — the agent classifies and replies to a customer's first message in a thread only, not ongoing back-and-forth.
- No fine-tuned classifier — intent classification is LLM-based (few-shot), not a trained model, given the dataset size available for this scope.
- No FastAPI wrapper — the pipeline runs as scripts; wrapping it as a service was out of scope for the time available.
- Intent taxonomy is not exhaustive of every possible Apple issue — it covers the 9 categories that actually appeared with meaningful volume in the sampled data.

**Intent taxonomy is a design decision, not a clustering artifact.** Unsupervised clustering (KMeans on sentence embeddings) did not cleanly separate real intents on its own: the software-bug topic is so dominant that it split across multiple clusters (all describing the same underlying issue differently) instead of leaving room for smaller-but-real categories like billing to form their own cluster. Billing showed up only as a small sub-pattern buried inside a larger cluster because it's rare relative to dominant issues like app crashes and connectivity problems. Final intent labels were derived by reading cluster samples and applying judgment — merging duplicate clusters, splitting mixed ones, and manually pulling out low-volume-but-high-stakes intents like billing rather than taking KMeans's output as-is.

Final intents: `app_performance_issue`, `audio_media_issue`, `battery_charging_issue`, `billing_issue`, `connectivity_issue`, `file_transfer_issue`, `general_bug_other`, `keyboard_input_bug`, `sync_icloud_issue`.

## 2. Results vs. baselines

| Metric | Trivial baseline | Simple (keyword) baseline | This system |
|---|---|---|---|
| Intent accuracy | 17.2% (always predicts majority class `keyboard_input_bug`) | 50.9% | **95.1%** |
| Intent macro F1 | — | 0.511 | **0.851** |
| Escalation accuracy | 71.8% (never escalates) | 69.9% | **69.3%** |
| Escalation precision (escalate class) | 0.000 | 0.400 | **0.452** |
| Escalation recall (escalate class) | 0.000 | 0.130 | **0.413** |
| Escalation F1 (escalate class) | 0.000 | 0.197 | **0.432** |
| Reply | one fixed canned sentence | template per keyword | retrieval-grounded LLM reply |

**Reading this table correctly matters more than the headline numbers themselves.** All three escalation-accuracy figures cluster around 69–72%, which at a glance makes the real system look barely better than either baseline — but accuracy is the wrong metric to compare on here, because ~72% of golden-set messages don't need escalation, so "never escalate" scores well by construction while being completely useless (0 precision/recall). The simple keyword baseline's escalation *precision* (0.400) is close to the real system's (0.452), but its *recall* (0.130) is far worse — the keyword rule (billing/profanity/caps) only catches a small fraction of true escalations, while the real system catches noticeably more (0.413 vs 0.130) even though both are still weak in absolute terms. On intent, the gap is unambiguous: the real system (95.1% accuracy, 0.851 macro F1) clearly beats both the trivial baseline (17.2%) and the keyword baseline (50.9%, dragged down by billing recall of just 0.17 and general_bug_other precision of 0.19 — keyword matching can't distinguish nuanced complaints the way the LLM classifier can).

## 3. Evaluation harness

- **Golden set:** 163 hand-labeled examples (intent, escalation decision + reason, reply-quality notes), sampled from clustered candidates and labeled by hand.
- **Intent/escalation metrics:** accuracy, macro precision/recall/F1, confusion matrix — computed in `evaluate_intent_and_escalation.py`.
- **LLM-as-judge:** replies scored on a 4-criteria rubric (accuracy, tone, completeness, relevance) across the full golden set.
- **Judge-vs-human agreement:** validated on a 30-sample check — **0.91 overall correlation, 98% of scores within one point** of human ratings. Tone showed perfect agreement (1.00); relevance and completeness were slightly lower (0.84–0.85) but still strong. This is direct evidence the judge's scores can be trusted as a human-scoring proxy.

## 4. Failure analysis — top 5 failure modes

1. **Escalation rule conflates tone with severity.** The sentiment-threshold rule tracks *how negative the language sounds*, not *how serious the underlying issue is*. False negatives were calm-but-serious messages ("asked to visit service center"); false positives were angry-sounding venting over ordinary bugs (all-caps, profanity) that a human didn't think needed escalation. Precision/recall on the escalate class (~0.45/0.41) show this is close to a coin flip on the decision that matters most.
   - Example (missed escalation): *"...disappointed with iPhone8Plus. Phone status after upgrade to 11.0.1 for last 48 hours Asked to visit service center!"* — labeled true, predicted false.
   - Example (false escalation): *"YA BOUT TWO SECONDS FROM PISSING ME OFF WITH THIS FREEZING BULLSHIT..."* — labeled false, predicted true.

2. **A meaningful fraction of "customer" messages are mid-thread replies, not original complaints.** Some rows in `customer_text_clean` are answers to the brand's clarifying questions (e.g. "iPhone 7", "yes, purchased Friday") rather than a problem statement. These are unclassifiable in isolation and inflate the `general_bug_other` bucket — a genuine data-quality limitation, not a model failure.

3. **Two intent categories have real boundary confusion.** `sync_icloud_issue` currently absorbs file-transfer-via-iTunes/USB complaints, which are conceptually distinct (cloud sync vs. local transfer) but collapsed into one label. Separately, the keyboard/input intent is being used as a catch-all for "anything about texting" (including spam/blocking issues), not just genuine keyboard bugs. `keyboard_input_bug` recall (0.79) is the weakest of all 9 intents, consistent with this.

4. **Reply completeness scores lower than accuracy/tone (2.90/5 vs. higher on other criteria).** Hypothesis: the retrieval grounding pulls real AppleSupport *first-touch* replies, which are often a single clarifying question rather than a full resolution — the model is faithfully mimicking real brand behavior (ask one thing at a time) rather than underperforming. This may not be a flaw to fix so much as a property of grounding in real, incremental support conversations instead of one-shot resolutions.

5. **A reasoning-model integration gotcha during reply generation.** `openai/gpt-oss-120b` is a reasoning model that silently consumed its entire token budget on internal reasoning tokens under a low `max_tokens`, returning empty output with no error. Fixed by setting `reasoning_effort="low"` and raising `max_tokens` — a non-obvious practical issue worth surfacing rather than glossing over.

## 5. What's misleading about my headline numbers

- **95.1% intent accuracy sounds strong, but is boosted by several very clean, easily-separable categories** (battery, connectivity, sync — 100% each) alongside two weaker ones (`keyboard_input_bug` at 0.79 recall, `billing_issue` at 0.92 recall). The macro F1 (0.851) is the more honest number since it doesn't let easy categories hide the harder ones. Also, `file_transfer_issue` has zero support in the golden set — it exists in the taxonomy but was never actually sampled, so no real conclusion can be drawn about it.
- **69.3% escalation accuracy is the most misleading number in this report.** Because most messages in the golden set don't require escalation, a model that rarely escalates will score well on raw accuracy while being nearly useless at the actual job (catching the ~30% of messages that do need a human). Precision/recall on the escalate class (~0.45/0.41) is the number that actually reflects whether this system can be trusted to make that call, and it's close to chance.
- **The judge-vs-human agreement (0.91) was measured on only 30 samples**, not the full golden set — a reasonable spot-check, not a large-sample validation. It's suggestive, not conclusive.
- **Reply quality is reported per-criterion, not as a single average**, deliberately — an overall average would hide that completeness/relevance are meaningfully weaker than accuracy/tone (see failure mode 4).

## 6. What I'd improve with one more week

- Replace the sentiment-threshold escalation rule with one that incorporates content/severity signals (e.g. phrases like "service center," "replace device," "refund," repeated-contact language) alongside sentiment, rather than sentiment alone.
- Split `sync_icloud_issue` into cloud-sync vs. local-transfer, and tighten the keyboard/input category's definition and few-shot examples so it stops catching unrelated texting complaints.
- Filter out mid-thread customer replies more aggressively upstream (currently only first-message-in-thread filtering is applied; some slip through as short, context-free answers).
- Expand the golden set specifically for `billing_issue` and `file_transfer_issue`, both under-sampled currently.
- Run judge-vs-human agreement on a larger sample (60–100 rows) rather than 30 for a more reliable estimate.

## 7. Decision log

1. **Chose AppleSupport as the brand** — high message volume, varied and recognizable issue types (battery, connectivity, iCloud, app bugs).
2. **Filtered to first-message-in-thread pairs only** — mid-thread replies (answers to clarifying questions) are unclassifiable without full thread context; excluding them keeps the training/golden set to genuine initial complaints.
3. **Found and fixed a join-direction bug in the data-prep SQL** — the first version selected customer *follow-up* messages instead of original complaints, because it matched on "replies to the brand" rather than "the brand's replies pointing back to the original." Caught via an unexpected `.isna()` count of 0, traced back to the join logic.
4. **Derived the final intent taxonomy by hand from clustering output, not directly from KMeans labels** — KMeans over-split the dominant software-bug topic and under-represented rare-but-important categories like billing; the taxonomy required merging, splitting, and manual pulling of low-volume intents.
5. **Used LLM-based (few-shot) intent classification instead of training a classifier** — appropriate given the available labeled data volume and project timeline.
6. **Built reply generation as retrieval-grounded (RAG over past resolved complaint-reply pairs)** rather than pure prompt-based generation, so replies reflect the brand's actual historical tone and resolution style.
7. **Set `reasoning_effort="low"` and raised `max_tokens`** for the reply-generation model after discovering it silently returned empty output — a reasoning model was consuming its full token budget on internal reasoning under a low limit.
8. **Chose a hybrid rule + sentiment-model approach for escalation** (certain intents always escalate, e.g. billing; strongly negative sentiment escalates; unclear intent escalates) rather than a second LLM call, for speed and explainability — later found this conflates tone with severity (see failure analysis).
9. **Raised the sentiment escalation threshold from 0.85 to 0.95** after testing showed 0.85 over-triggered on routine complaint language — describing a broken feature reads as "negative" in sentiment terms even when the customer isn't unusually upset. Confirmed with real before/after examples, not guesswork.
10. **Chose macro-averaged precision/recall/F1 over micro/weighted for intent evaluation** so that rarer intents (e.g. billing) aren't washed out by dominant ones (app performance) in the headline metric.
11. **Validated the LLM judge against human scoring on a 30-sample subset** before trusting it across the full golden set, rather than assuming judge-model agreement.
12. **Built the trivial and simple baselines to include escalation and reply, not just intent** — a baseline that only covers one of the three tasks understates how much the real system needs to beat.
13. **Kept the simple baseline's escalation rule content-based (billing/profanity/caps) rather than copying the real system's sentiment approach** — so the baseline is genuinely a different, simpler strategy, not a weaker copy of the same idea.
