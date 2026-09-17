<div align="center">

# 🍎 AppleSupport Twitter Support Agent

### A retrieval-grounded AI pipeline that classifies, replies to, and triages real customer support tweets — built for the Hiver SDE Intern take-home

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph-style pipeline](https://img.shields.io/badge/Pipeline-Multi--stage-orange.svg)]()
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-6E56CF.svg)]()
[![Groq](https://img.shields.io/badge/LLM-Groq%20API-F55036.svg)]()

<a href="REPORT.md"><b>📊 Full Report</b></a> ·
<a href="#-setup"><b>⚙️ Setup</b></a> ·
<a href="#-reproducing-the-results--15-min"><b>▶️ Reproduce Results</b></a> ·
<a href="#-known-limitations"><b>⚠️ Limitations</b></a>

</div>

<br>

## 🧠 What It Does

> Most support bots reply with generic AI-sounding text. This agent grounds every reply in how AppleSupport has actually resolved similar issues in the past — and knows when to hand off to a human instead of guessing.

Given an incoming customer tweet, the pipeline:

<table>
<tr><td width="40px" align="center">🏷️</td><td><b>Intent Classifier</b></td><td>Sorts the message into one of 9 intents derived from the real data (not a fixed taxonomy handed to us)</td></tr>
<tr><td align="center">🔎</td><td><b>Retrieval Agent</b></td><td>Finds similar past complaints and AppleSupport's actual replies to them via ChromaDB</td></tr>
<tr><td align="center">✍️</td><td><b>Reply Generator</b></td><td>Drafts a reply grounded in those real historical resolutions, in AppleSupport's tone</td></tr>
<tr><td align="center">🚦</td><td><b>Escalation Agent</b></td><td>Decides auto-handle vs. escalate to a human, with a stated reason (hybrid rule + sentiment)</td></tr>
<tr><td align="center">⚖️</td><td><b>LLM Judge</b></td><td>Scores reply quality on a 4-criteria rubric, validated against human ratings</td></tr>
</table>

<br>

## 🗺️ Pipeline

<div align="center">

```text
  ┌─────────────┐      ┌───────────────┐      ┌──────────────────┐
  │  Raw tweets │─────▶│  Clean & pair │─────▶│ Intent classifier │
  │  (Kaggle)   │      │  (first-msg)  │      │   (few-shot LLM)  │
  └─────────────┘      └───────────────┘      └─────────┬─────────┘
                                                          │
                          ┌───────────────────────────────┘
                          ▼
                 ┌─────────────────┐        ┌───────────────────┐
                 │ Retrieval (RAG) │───────▶│  Reply generator   │
                 │   ChromaDB      │        │  (grounded in      │
                 └─────────────────┘        │  real past replies)│
                                             └─────────┬───────────┘
                                                        │
                                             ┌──────────▼──────────┐
                                             │ Escalation decision │
                                             │ (rules + sentiment) │
                                             └──────────┬──────────┘
                                                        │
                                             ┌──────────▼──────────┐
                                             │   LLM-as-judge +    │
                                             │  golden-set eval    │
                                             └──────────────────────┘
```

</div>

📄 Full write-up — problem framing, baselines, failure analysis, and the decision log → **[REPORT.md](REPORT.md)**

<br>

## 📊 Results

| Metric | Trivial baseline | Simple (keyword) baseline | **This system** |
|---|---|---|---|
| Intent accuracy | 17.2% | 50.9% | **95.1%** |
| Intent macro F1 | — | 0.511 | **0.851** |
| Escalation accuracy | 71.8% | 69.9% | **69.3%** |
| Escalation F1 (escalate class) | 0.000 | 0.197 | **0.432** |
| LLM-judge ↔ human agreement | — | — | **0.91 correlation** |

> ⚠️ The escalation-accuracy numbers look deceptively similar across the board — see [REPORT.md § 5](REPORT.md) for why raw accuracy is the wrong metric to trust here, and what precision/recall actually show.

<br>

## 🧰 Tech Stack

<div align="center">

| Layer | Technology |
|:---|:---|
| 🏷️ Intent classification | Groq LLM API, few-shot prompting |
| 🔎 Retrieval | ChromaDB + `sentence-transformers` (`all-MiniLM-L6-v2`) |
| 🧭 Intent discovery | KMeans clustering on sentence embeddings |
| ✍️ Reply generation | Groq LLM API (`openai/gpt-oss-120b`) |
| 🚦 Escalation | `cardiffnlp/twitter-roberta-base-sentiment-latest` + rule layer |
| ⚖️ Evaluation | scikit-learn metrics + LLM-as-judge |
| 🗃️ Data | Kaggle Customer Support on Twitter |

</div>

<br>

## ⚙️ Setup

<details open>
<summary><b>1. Clone and create environment</b></summary>

```bash
git clone https://github.com/karthikdm21/support-agent.git
cd support-agent
python -m venv venv
venv\Scripts\activate
```

</details>

<details open>
<summary><b>2. Install dependencies</b></summary>

```bash
pip install -r requirements.txt
```

</details>

<details open>
<summary><b>3. Configure environment variables</b></summary>

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-key-here
```

</details>

<details open>
<summary><b>4. Download the dataset</b></summary>

Download from Kaggle ([`thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)) and place it at `data/twcs.csv`. The raw ~3M-row file is not committed to this repo.

</details>

<br>

## ▶️ Reproducing the Results (< 15 min)

```bash
python clean_data.py                        # filters + cleans AppleSupport complaint-reply pairs
python discover_intents.py                  # clusters sample messages to surface candidate intents
python classify_intent.py                   # classifies messages into the final intent set
python build_retrieval_index.py             # builds the Chroma vector index over historical pairs
python build_golden_set_sample.py           # samples candidates for hand-labeling
```

`data/golden_set.csv` (163 hand-labeled rows) is already included, so evaluation can run directly:

```bash
python evaluate_intent_and_escalation.py    # intent + escalation accuracy against golden set
python run_judge_on_golden_set.py           # LLM-as-judge reply quality scoring
python compare_judge_vs_human.py            # judge-vs-human agreement check
python baseline_trivial.py                  # required trivial baseline
python baseline_simple.py                   # required simple baseline
```

Generate a reply for a single message manually:

```bash
python generate_reply.py
```

<br>

## 📦 Project Structure

```text
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
REPORT.md                          # framing, results, failure analysis, decision log
```

<br>

## ⚠️ Known Limitations

- 🔸 Escalation logic conflates message **tone** with issue **severity** — see failure analysis in [REPORT.md](REPORT.md)
- 🔸 Some `customer_text_clean` rows are mid-thread replies, not original complaints, and are unclassifiable in isolation
- 🔸 Intent taxonomy was hand-derived from clustering output, not perfectly clean cluster boundaries (`sync_icloud_issue` vs. file transfer, `keyboard_input_bug` as a catch-all)
- 🔸 Only a subsample of the ~3M-row dataset is used, as encouraged by the assignment
- 🔸 No multi-turn conversation handling — classifies and replies to a customer's first message only

<br>

<div align="center">

Built for the Hiver SDE Intern Assignment by <a href="https://github.com/karthikdm21">karthikdm21</a>

AI coding assistance (Claude) was used for debugging, code review, and documentation — all code was written and understood incrementally.

</div>