# EECS-486-Final-Project
## Project Files

- `ir_system_queries_updated.py` — main script
- `test_queries.txt` — batch of evaluation queries
- `datasets/` — folder containing all CSV files(Maternal_Health-Hospital.csv, HCAHPS-National.csv, DAC_NationalDownloadableFile.csv
- `ir_index/` — saved index files created after building the index
- `outputs/` — output folder for single-query runs
- `experiment_outputs/` — output folder for experiment runs(when running batch evaluations)

*Note datasets and ir_index are not included in repo because they are to large to push to GitHub

## How to Run
1. To build the index(run once or after data changes)\
`python3 ir_system_updated.py --mode build_index --data_dir datasets --index_dir ir_index --save_full_provider_table`
2. To run a batch evaluation(execute all queries in test_queries.txt)\
`python3 ir_system_updated.py --mode run_experiments --index_dir ir_index --out_dir experiment_outputs --top_k 20 --queries_file test_queries.txt`
3. To run a single query\
`python3 ir_system_updated.py --mode search --index_dir ir_index --query "fertility specialist with telehealth in CA" --out_dir outputs --top_k 20 --reranker fair_top_k`\
*Note you can change the reranker:\
`--reranker baseline`\
`--reranker linear`\
`--reranker mmr`\
`--reranker fair_top_k`
---
