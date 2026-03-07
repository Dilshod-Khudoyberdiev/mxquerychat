# Thesis Evaluation Summary

## Dataset
- DuckDB path: `C:\Users\DilshodKhudoyberdiev\Documents\GitHub\mxquerychat\mxquerychat.duckdb`
- DuckDB version: `v1.4.3`
- Main fact table: `ticket_verkaeufe` with `15000` rows
- Fact table mapping: ticket_verkaeufe, plan_umsatz, sonstige_angebote
- Dimension table mapping: ticket_produkte, tarifverbuende, postleitzahlen, regionen_bundesland, meldestellen
- Distinct products: `5`
- Distinct tariff networks: `20`
- Distinct postal codes: `396`
- Distinct federal states: `16`

## Domain Test (20 questions)
- ExecAcc: `1.0`
- Exact Match: `1.0`
- Compile Rate: `1.0`
- Generation latency median/p95 (s): `0.001193` / `0.001652`
- Execution latency median/p95 (s): `0.005888` / `0.017607`
- Total latency median/p95 (s): `0.007126` / `0.019116`
- Generation latency min/max (s): `0.000988` / `0.002206`
- Execution latency min/max (s): `0.00199` / `0.024167`
- Total latency min/max (s): `0.003102` / `0.025524`
- Question source: `docs/demo_questions.md (first 20 EN questions)`
- Gold SQL source: `training_data/training_examples.csv (exact normalized question match)`

## Benchmark Test
- Available: `False`
- Notes: Spider/BIRD datasets were not found locally in this repository; benchmark subset skipped.

## Safety Test (15 cases)
- Blocked rate: `1.0`
- Zero writes confirmed: `True`
