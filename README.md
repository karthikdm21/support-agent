# AppleSupport Twitter Support Agent

An AI customer-support agent built for AppleSupport's Twitter interactions, using real historical brand replies from the Kaggle Customer Support on Twitter dataset. Given an incoming customer tweet, the agent:

1. Classifies the message into one of 9 intents derived from the data
2. Drafts a reply grounded in how AppleSupport has actually resolved similar past complaints (RAG over historical complaint-reply pairs)
3. Decides whether to auto-handle or escalate to a human, with a stated reason

Built for the Hiver SDE Intern take-home assignment.

## Setup

\`\`\`bash
git clone https://github.com/karthikdm21/support-agent.git
cd support-agent
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
\`\`\`

Create a `.env` file in the project root:

\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`

Download the dataset from Kaggle ([`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)) and place `twcs.csv` in `data/twcs.csv`. The raw file (~3M rows) is not committed to this repo.

## Reproducing the results (< 15 min)

Run in order:

\`\`\`bash
python clean_data.py                        # filters + cleans AppleSupport complaint-reply pairs → data/clean_pairs.csv
python discover_intents.py                  # clusters sample messages to surface candidate intents
python classify_intent.py                   # classifies messages into the final intent set
python build_retrieval_index.py             # builds the Chroma vector index over historical pairs
python build_golden_set_sample.py           # samples candidates for hand-labeling → data/golden_set_candidates.csv
\`\`\`

`data/golden_set.csv` (163 hand-labeled rows) is already included, so the next steps can be run directly without re-labeling:

\`\`\`bash
python evaluate_intent_and_escalation.py    # intent + escalation accuracy against golden set
python run_judge_on_golden_set.py           # LLM-as-judge reply quality scoring
python compare_judge_vs_human.py            # judge-vs-human agreement check
python baseline_trivial.py                  # trivial baseline (majority class, canned reply, never escalate)
python baseline_simple.py                   # simple baseline (keyword intent match, template reply, rule-based escalation)
\`\`\`

Each script prints its metrics to the console and writes a results CSV to `data/`.

To generate a reply for a single message (interactive/manual check):

\`\`\`bash
python generate_reply.py
\`\`\`

## Project structure

\`\`\`
clean_data.py                      # raw CSV → cleaned first-message complaint/reply pairs
discover_intents.py                # unsupervised clustering to surface candidate intents
intents.py                         # final intent taxonomy + descriptions
classify_intent.py                 # LLM-based intent classification
build_retrieval_index.py           # embeds historical pairs into ChromaDB
generate_reply.py                  # retrieval-grounded reply generation
decide_escalation.py               # rule + sentiment-based escalation decision
build_golden_set_sample.py         # sampling for hand-labeling
evaluate_intent_and_escalation.py  # accuracy metrics against golden set
judge_reply.py / run_judge_on_golden_set.py   # LLM-as-judge reply scoring
compare_judge_vs_human.py          # judge agreement validation
baseline_trivial.py                # required trivial baseline
baseline_simple.py                 # required simple (keyword) baseline
data/golden_set.csv                # 163 hand-labeled ground-truth examples
REPORT.md                          # full write-up: framing, results, failure analysis, decision log
\`\`\`

## Key results (see REPORT.md for full detail)

- Intent classification accuracy: 95.1% (macro F1 0.851) against the golden set
- Escalation decision accuracy: 69.3%, but escalate-class precision/recall ~0.45/0.41 — see REPORT.md for why this headline number is misleading
- LLM-judge agreement with human scoring: 0.91 correlation, 98% within one point (30-sample check)

## Notes

- Only a subsample of the full ~3M-row dataset is used, as encouraged by the assignment.
- AI coding assistance (Claude) was used throughout for debugging, code review, and drafting this documentation; all code was written and understood incrementally, not submitted unread.