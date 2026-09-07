# 🏏 OmniCourt-AI: Autonomous Cricket Third Umpire & DRS Intelligence Cockpit

OmniCourt-AI is an automated Decision Review System (DRS) built to adjudicate high-stakes cricket dismissals (Run Out, Stumping, and Bowled) in under 5 seconds using high-speed computer vision and Google Gemini 3.1 Flash-Lite.

---

## ⚡ What is OmniCourt-AI?

In professional cricket, Third Umpire reviews create bottlenecks:
- **High Review Latency:** Umpires take 45–90 seconds manually scrubbing broadcast video to pinpoint the frame of wicket breakdown.
- **Human Parallax & Grounding Errors:** Angles, shadows, and bouncing bats lead to contentious calls under MCC Law 38 and 30 (*"the line belongs to the umpire"*).
- **Token Inefficiency in Naive AI:** Sending entire multi-gigabyte video files frame-by-frame to LLMs causes high latency and hallucinated line boundaries.

**OmniCourt-AI solves this with a two-stage multimodal pipeline:**
1. **Stage 1 (Temporal Impact Isolation):** Ingests match video clips via the Google GenAI Files API (`gemini-3.1-flash-lite`) to pinpoint the exact microsecond bails dislodge or LED Zing bails illuminate.
2. **Stage 2 (Deterministic Grounding Adjudication):** Extracts a tight window of 5 candidate frames around impact and applies MCC cricket laws using strict Pydantic structured output (`AdjudicationDocket`).

---

## 🏗️ Architecture


[ Match Incident Video (.mp4 / .mov) ]
│
▼
[ cv_engine.py / OpenCV ]
│
├──> Stage 1: Gemini Files API Temporal Scanner
│    (Pinpoints Bail Separation Timestamp T_impact)
│
└──> Target Window Slicer (Extracts 5 candidate frames around T_impact)
│
▼
[ umpire_agent.py / GenAI SDK ]
│
├── Evaluates Crease Grounding & Bail Dislodgment
└── Enforces Strict Pydantic Schema (AdjudicationDocket)
│
▼
[ Streamlit Broadcast Cockpit (app.py) ]
(Verdict Card | MCC Law Reference | Confidence Score)

```

---

## ⚖️ MCC Cricket Laws Enforced

* **MCC Law 38.1 & 30.1 (Run Out / Batter Out of Ground):** The bat or person must be grounded on the turf **past the inside edge** of the popping crease before bails break. Touching on the line is **OUT** (*"the line belongs to the umpire"*).
* **MCC Law 39.1 (Stumped):** The striker is **OUT** if the wicket-keeper breaks the wicket while the striker is out of their ground and not attempting a run.
* **MCC Law 32.1 (Bowled):** The delivered ball breaks the wicket directly. Crease grounding is disregarded and ruled **OUT**.
* **MCC Law 29.1 (Wicket Put Down):** At least one bail must be completely dislodged from the stumps or a stump struck out of the ground.

---

## 🎛️ Broadcast Studio Features

- **Sub-Frame Timeline Scrubber:** Step through match footage with precision buttons (`⏮ -5`, `◀ -1`, `+1 ▶`, `+5 ⏭`).
- **Manual Impact Lock:** Instant override button to lock any active frame as the primary impact reference.
- **Visual Evidence Strip:** Displays the exact candidate frames delivered to Gemini alongside timestamp deltas.
- **Official Verdict Banner:** High-contrast verdict cards (`OUT` / `NOT OUT`) displaying confidence ratings, governing MCC rule citations, and a step-by-step reasoning trace.

---

## 📂 Repository Structure



OmniCourt-AI/
├── app.py                      # Streamlit DRS broadcast studio application
├── cv_engine.py                # Frame extraction & Stage-1 Gemini temporal scanner
├── umpire_agent.py             # Stage-2 MCC rule engine & Pydantic schema
├── generate_sample_videos.py   # Synthetic match footage generator
├── Dockerfile                  # Container definition for Google Cloud Run
├── .dockerignore               # Deployment exclusion list
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
└── README.md                   # Project documentation


---

## 🚀 Quickstart (Run Locally)

### 1. Clone & Set Up Virtual Environment
```bash
git clone [https://github.com/akshaya-siva/OmniCourt-AI.git](https://github.com/akshaya-siva/OmniCourt-AI.git)
cd OmniCourt-AI

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### 2. Configure Environment Variable

Create a `.env` file in the project root:

```env
GEMINI_API_KEY="your_free_tier_gemini_api_key_here"

```

### 3. Launch the Studio

```bash
streamlit run app.py

```

Open your browser at `http://localhost:8501`.

---

## ☁️ Cloud Run Deployment

Deploy directly to Google Cloud Run from source:

```bash
gcloud run deploy omnicourt-ai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars GEMINI_API_KEY="your_free_tier_gemini_api_key_here"

```

---

## 📊 Structured Verdict Schema (`AdjudicationDocket`)

```json
{
  "decision": "OUT",
  "critical_timestamp_sec": 1.50,
  "bat_grounded_behind_crease": false,
  "governing_mcc_law": "MCC Law 38.1 (Run Out)",
  "confidence_score": 0.98,
  "visual_evidence_summary": "At T=1.50s, bails are dislodged while the bat is short of the white popping crease line without ground contact past the line.",
  "agent_reasoning_trace": [
    "Step 1: Examined delivery and verified wicket impact at T=1.50s.",
    "Step 2: Identified exact frame where bails first separate and fly into the air.",
    "Step 3: Inspected bat tip position against popping crease line; confirmed short of crease."
  ]
}

```

---

## 📜 Technology Stack & License

* **Core Reasoning:** Google Gemini 3.1 Flash-Lite via the Google GenAI Python SDK
* **Computer Vision:** OpenCV (headless)
* **Frontend:** Streamlit
* **Container & Hosting:** Docker / Google Cloud Run (Serverless)
* **License:** Distributed under the MIT License