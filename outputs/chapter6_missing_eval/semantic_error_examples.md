# Semantic Error Examples

- Status: measured
- Tested harder in-domain paraphrases: `10`
- Summary: Found `3` semantic error example(s) in `10` tested paraphrases.

## Error 1
- Question: `Show revenue by tariff association and federal state.`
- Category: `wrong aggregation`
- Explanation: The generated SQL compiled and ran, but it returned a different grouping or filter than the gold query.
- Generated SQL:
```sql
SELECT
    tv.jahr,
    t.name AS tarifverbund_name,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN tarifverbuende t
    ON tv.tarifverbund_id = t.tarifverbund_id
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2

GROUP BY tv.jahr, t.name, rb.bundesland_name
ORDER BY tv.jahr, t.name, umsatz_eur DESC
```
- Corrected SQL:
```sql
SELECT t.name AS tarifverbund_name, rb.bundesland_name, SUM(tv.umsatz_eur) AS umsatz_eur FROM ticket_verkaeufe tv JOIN tarifverbuende t ON tv.tarifverbund_id = t.tarifverbund_id JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 GROUP BY t.name, rb.bundesland_name ORDER BY t.name, umsatz_eur DESC;
```
## Error 2
- Question: `For each state, show top 3 ticket types by revenue.`
- Category: `wrong aggregation`
- Explanation: The generated SQL dropped the ticket-type dimension from the grouping.
- Generated SQL:
```sql
SELECT
    tv.jahr,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2

GROUP BY tv.jahr, rb.bundesland_name
ORDER BY tv.jahr, umsatz_eur DESC
```
- Corrected SQL:
```sql
WITH revenue_by_state AS (SELECT rb.bundesland_name, tp.ticket_name, SUM(tv.umsatz_eur) AS revenue_eur FROM ticket_verkaeufe tv JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 GROUP BY rb.bundesland_name, tp.ticket_name), ranked AS (SELECT bundesland_name, ticket_name, revenue_eur, ROW_NUMBER() OVER (PARTITION BY bundesland_name ORDER BY revenue_eur DESC) AS rn FROM revenue_by_state) SELECT bundesland_name, ticket_name, revenue_eur FROM ranked WHERE rn <= 3 ORDER BY bundesland_name, revenue_eur DESC;
```
## Error 3
- Question: `Show revenue by state and ticket type for 2025.`
- Category: `wrong column`
- Explanation: The generated SQL dropped the ticket-type dimension from the grouping.
- Generated SQL:
```sql
SELECT
    tv.jahr,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2
WHERE tv.jahr = 2025
GROUP BY tv.jahr, rb.bundesland_name
ORDER BY tv.jahr, umsatz_eur DESC
```
- Corrected SQL:
```sql
SELECT rb.bundesland_name, tp.ticket_name, SUM(tv.umsatz_eur) AS revenue_eur FROM ticket_verkaeufe tv JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2 WHERE tv.jahr = 2025 GROUP BY rb.bundesland_name, tp.ticket_name ORDER BY rb.bundesland_name, revenue_eur DESC;
```
