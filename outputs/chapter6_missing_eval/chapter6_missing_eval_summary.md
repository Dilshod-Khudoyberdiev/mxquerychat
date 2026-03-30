# Chapter 6 Missing Evaluation Summary

## 1. Evaluation environment
- Status: measured
- Operating system: `Windows-11-10.0.26200-SP0`
- CPU model: `Intel(R) Core(TM) Ultra 5 125U`
- RAM: `15.52 GiB`
- Python / Streamlit / DuckDB / Vanna / ChromaDB: `3.13.3` / `1.53.0` / `1.4.3` / `2.0.1` / `1.4.1`
- Local runtime / model: `ollama version is 0.17.7` / `mistral:latest`

## 2. Before/after training impact
- Status: measured
- Before ExecAcc / EM / Compile: `0.25` / `0.25` / `0.75`
- After ExecAcc / EM / Compile: `1.0` / `1.0` / `1.0`
- Before generation median / total median (s): `0.006362` / `0.018133`
- After generation median / total median (s): `0.004347` / `0.010294`
- Error state: before=`none` after=`none`

## 3. Semantic error examples
- Status: measured
- Found `3` semantic error example(s) in `10` tested paraphrases.

## 4. Benchmark availability / results
- Status: measured
- Available: `True`
- Reason: `local held-out benchmark CSV found`

## 5. Usability measurability
- Status: partially measured
- Number of sessions: `not measurable automatically`
- Successful question -> run flows: `10`
- Formal usability outcomes require human participants unless participant/session instrumentation is added.
