# 🍜 National Cuisine Classification — Applied NLP Mini Project

**Course:** Applied NLP Mini Project  
**Option:** B — Prompt Engineering with LLMs  
**Task:** Classify recipes into their national cuisine using ingredient lists as input  

---

## 📌 Project Overview

This project classifies recipes into one of **6 national cuisines** (Italian, Mexican, Indian, Thai, Japanese, Southern US) using only the ingredient list as input text.

We compare two LLM prompting strategies:

| Strategy | Description |
|---|---|
| **Zero-shot** | Send the recipe directly to the LLM with a label list — no examples |
| **Dynamic Few-shot** | Embed the recipe, retrieve the k most similar training recipes from ChromaDB, use them as examples in the prompt |

The core research question is: **does retrieving semantically similar examples at query time improve classification accuracy compared to no examples at all?**

---

## 🗂️ Project Structure

```
cuisine-classification/
├── data/
│   ├── raw/              ← Put train.json here (downloaded from Kaggle)
│   ├── processed/        ← Auto-generated: train_pool.csv, test_set.csv
│   └── results/          ← Auto-generated: intermediate outputs
├── src/
│   ├── config.py         ← All settings and paths (edit via .env only)
│   ├── dataset.py        ← Data loading, filtering, splitting
│   ├── embeddings.py     ← SentenceTransformer encode functions
│   ├── vectorstore.py    ← ChromaDB build and query operations
│   ├── prompts.py        ← All LLM prompt templates
│   ├── inference.py      ← Runs the full experiment, saves predictions
│   └── evaluate.py       ← Metrics, confusion matrices, figures
├── notebooks/
│   └── analysis.ipynb    ← EDA and result analysis notebook
├── results/
│   ├── predictions/      ← Auto-generated: predictions.csv
│   └── figures/          ← Auto-generated: confusion matrices, charts
├── .env.example          ← Template for your .env file
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone the repository

```bash
git init
git remote add origin https://github.com/nithit-cypherX/cuisine-classification.git
git branch -M main
git pull origin main
```

### Step 2 — Create a virtual environment

```bash
# Create the conda environment
conda create -n cuisine-nlp python=3.10 -y

# Activate it
conda activate cuisine-nlp
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up your environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your OpenAI API key:

```
OPENAI_API_KEY=your_actual_key_here
```

⚠️ **Never commit your `.env` file. It is already in `.gitignore`.**

### Step 5 — Download the dataset

1. Go to: https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset
2. Download `train.json`
3. Place it at: `data/raw/train.json`

⚠️ **Do not commit the dataset.** The `data/raw/` folder is in `.gitignore`.

---

## 🚀 Running the Pipeline

Run each step **in order**. Each script must complete successfully before running the next one.

### Step 1 — Prepare the dataset

```bash
python src/dataset.py
```

**What it does:**
- Loads `data/raw/train.json`
- Filters to 6 target cuisines
- Performs a stratified 80/20 train/test split
- Samples 15 recipes per cuisine for the test set (90 total)
- Saves `data/processed/train_pool.csv` and `data/processed/test_set.csv`

**Expected output:**
```
[dataset] Loaded 39774 recipes across 20 cuisines.
[dataset] After cuisine filter: ~8000 recipes across 6 cuisines.
[dataset] Train pool: ~6400 recipes
[dataset] Test set:   90 recipes (15 per cuisine)
[dataset] Saved train pool → ./data/processed/train_pool.csv
[dataset] Saved test set   → ./data/processed/test_set.csv
```

---

### Step 2 — Build the ChromaDB vector store

```bash
python src/vectorstore.py
```

**What it does:**
- Loads the training pool CSV
- Embeds every recipe using `all-MiniLM-L6-v2`
- Stores all embeddings + metadata in a local ChromaDB collection

⚠️ **The test set is NOT stored here.** Test recipes are only embedded at query time during inference.

**Expected output:**
```
[embeddings] Loading model: all-MiniLM-L6-v2
[embeddings] Encoding ~6400 texts...
[vectorstore] Stored ~6400 recipes in ChromaDB collection: cuisine_recipes
```

This step takes ~2–5 minutes on CPU. Run it only once — ChromaDB persists to disk.

---

### Step 3 — Run the experiment

```bash
python src/inference.py
```

**What it does:**
- Loads the test set (90 recipes)
- For each recipe, runs both prompt strategies:
  - **Zero-shot**: formats prompt → calls OpenAI API
  - **Dynamic few-shot**: embeds recipe → queries ChromaDB for k=3 similar training recipes → formats prompt with retrieved examples → calls OpenAI API
- Parses LLM outputs to clean cuisine labels
- Saves all predictions to `results/predictions/predictions.csv`

**Expected output:**
```
[inference] Running experiment on 90 test recipes...
[inference] Processing 1/90 — italian
[inference] Processing 2/90 — mexican
...
[inference] Predictions saved → ./results/predictions/predictions.csv
[inference] zero_shot_pred invalid rate: 2.2%
[inference] dynamic_few_shot_pred invalid rate: 3.3%
```

⚠️ This makes 180 API calls total (90 × 2 strategies). Save your results CSV immediately — do not re-run unnecessarily.

---

### Step 4 — Evaluate and visualise results

```bash
python src/evaluate.py
```

**What it does:**
- Computes accuracy, macro F1, per-class F1, invalid rate for both strategies
- Prints a classification report for each strategy
- Generates and saves:
  - `results/figures/confusion_matrix_zero_shot.png`
  - `results/figures/confusion_matrix_dynamic_few_shot.png`
  - `results/figures/per_class_f1_comparison.png`

---

## 📊 Expected Output Files

After running all steps, your repository should contain:

```
results/
├── predictions/
│   └── predictions.csv          ← id, true_label, zero_shot_pred, dynamic_few_shot_pred
└── figures/
    ├── confusion_matrix_zero_shot.png
    ├── confusion_matrix_dynamic_few_shot.png
    └── per_class_f1_comparison.png
```

---

## 🧪 Experiment Settings

All settings are controlled via `.env`. Defaults:

| Setting | Value | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM used for classification |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `TARGET_CUISINES` | 6 cuisines | See .env.example |
| `TEST_SAMPLES_PER_CUISINE` | 15 | = 90 total test recipes |
| `FEW_SHOT_K` | 3 | Retrieved examples per query |
| `RANDOM_SEED` | 42 | For reproducibility |
| `LLM_TEMPERATURE` | 0 | Deterministic outputs |

---

## ⚠️ Important Rules for the Team

1. **Never commit `.env`** — your API key must stay private
2. **Never commit `data/raw/` or `data/processed/`** — large files, excluded by `.gitignore`
3. **Never store test set embeddings in ChromaDB** — test recipes are only embedded at query time
4. **Do not modify prompt templates mid-experiment** — if testing a new prompt, add a new function in `prompts.py`
5. **Save predictions.csv before re-running inference** — API calls cost money
6. **All settings go in `.env`** — never hardcode values in source files

---

## 🔬 Analysis Questions (for the report)

After evaluation, answer these questions using your results as evidence:

1. Did dynamic few-shot outperform zero-shot overall? By how much in macro F1?
2. Which cuisines benefited most from retrieval — and why?
3. Did dynamic few-shot ever hurt performance on any class? Examine the confusion matrix.
4. Print 3–5 retrieved ChromaDB examples for interesting test cases. Were the retrievals actually relevant?
5. Was the invalid output rate different between strategies? What does that suggest?

---

## 📦 Dependencies

See `requirements.txt` for full list. Key packages:

| Package | Purpose |
|---|---|
| `openai` | LLM API calls |
| `sentence-transformers` | Recipe embedding |
| `chromadb` | Vector storage and retrieval |
| `scikit-learn` | Metrics and evaluation |
| `matplotlib` / `seaborn` | Visualisation |

---

## 👥 Team

| Role | Name |
|---|---|
| Project owner | [Your Name] |
| Implementation | [Teammate Name] |

---

## 📄 License

For academic use only — ITCS Applied NLP Mini Project.
