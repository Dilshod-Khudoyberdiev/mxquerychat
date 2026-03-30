# Training Impact Report

- Status: measured
- Question subset size: `8`
- Training change: before removed the subset rows from a temporary CSV and isolated Chroma store; after used the full current CSV in a separate isolated Chroma store.
- Removed/disabled example count: `6`

## Before
- ExecAcc: `0.25`
- Exact Match: `0.25`
- Compile Rate: `0.75`
- Generation latency median (s): `0.006362`
- Total latency median (s): `0.018133`
- Error: `none`

## After
- ExecAcc: `1.0`
- Exact Match: `1.0`
- Compile Rate: `1.0`
- Generation latency median (s): `0.004347`
- Total latency median (s): `0.010294`
- Error: `none`

## Removed Or Disabled Examples
- `Per state, show top 3 ticket types by revenue.`
- `Show postal code, city, and state for the top 20 revenues.`
- `Show average price per ticket type (from sales) and compare to ticket_produkte.`
- `Compare actual revenue (ticket_verkaeufe) with planned revenue (plan_umsatz) per month.`
- `Show per tariff association the deviation (actual - plan) for 2025.`
- `Show monthly deviation as a percentage.`
