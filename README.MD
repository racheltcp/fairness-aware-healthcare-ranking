# Fairness-Aware Healthcare Provider Ranking

## Overview

This project is an information retrieval system for recommending healthcare providers, with a focus on women's healthcare.

Users can enter natural-language queries such as:

`fertility specialist with telehealth in CA`

The system retrieves relevant providers using TF-IDF and cosine similarity, then combines that with provider information such as specialty, location, telehealth availability, affordability, quality, and experience.

Three fairness-aware reranking methods are included:

- Linear fairness reweighting
- MMR-based reranking
- FA*IR-inspired top-k reranking

The goal is to improve representation and exposure of providers relevant to women's healthcare while still maintaining strong search relevance.

## How It Works

The system:

1. Loads and cleans healthcare provider data from CMS
2. Creates features for each provider, including quality, experience, affordability, accessibility, and women's-health relevance
3. Builds a text representation for each provider
4. Creates a TF-IDF index over the provider data
5. Parses natural-language queries for specialty, location, telehealth, affordability, and other preferences
6. Produces a baseline ranking using semantic similarity and structured provider features
7. Applies one of the fairness-aware reranking methods
8. Evaluates the results using ranking and exposure metrics

## Evaluation

The system was tested on 80 queries across categories including:

- Fertility and IVF
- OBGYN and gynecology
- Maternal and pregnancy care
- Oncology
- General women's health
- General healthcare
- Symptom-based queries
- Queries with multiple constraints

The rankings are evaluated using:

- Precision@k
- Recall@k
- NDCG@k
- Representation@k
- Group exposure
- Exposure gap

The `results/` folder contains aggregate results from the experiments.

## Files

- `healthcare_ranking.py` - main system and experiment code
- `test_queries.txt` - queries used for evaluation
- `results/` - experiment metrics and summaries
- `requirements.txt` - Python dependencies

The raw CMS datasets and generated index files are not included because of their size.

## Requirements

Python 3

Packages:

- numpy
- pandas
- scipy
- scikit-learn

Install them with:

```bash
pip install -r requirements.txt
```

## How to Run

### Build the index

```bash
python3 healthcare_ranking.py --mode build_index --data_dir datasets --index_dir ir_index --save_full_provider_table
```

### Run one query

```bash
python3 healthcare_ranking.py --mode search --index_dir ir_index --query "fertility specialist with telehealth in CA" --out_dir outputs --top_k 20 --reranker fair_top_k
```

Available rerankers:

- `linear`
- `mmr`
- `fair_top_k`

### Run the full evaluation

```bash
python3 healthcare_ranking.py --mode run_experiments --index_dir ir_index --out_dir experiment_outputs --top_k 20 --queries_file test_queries.txt
```

## Contributors

- Rachel Miller
- Houda Albazzal
- Vera Sivrieva
- Julia Sochava
- Aimee Wu
