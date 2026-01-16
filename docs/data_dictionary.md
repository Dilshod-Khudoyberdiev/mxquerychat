# mxQueryChat – Data Dictionary

This document describes the **synthetic (mock) dataset** used in **mxQueryChat**.

The dataset is designed for a demo where users can:
- ask questions in natural language (DE/EN),
- get SQL generated automatically,
- run the SQL in **read-only** mode on **DuckDB**,
- and see results as tables/charts.

**Important:**  
All data is **synthetic**. There is **no customer schema** and **no real personal data**.  
Any emails use `example.org`.

# Overview: What types of tables exist?

To keep the dataset understandable, tables are grouped into 4 areas:

1) **Reference tables** = “lookup lists” (states, postal codes, ticket products)  
2) **Fact tables** = “business data you analyze” (sales, planned revenue)  
3) **Distribution tables** = “allocation rules” (shares per Bundesland)  
4) **Users & permissions** = “optional security demo data”  

---

# 1) Reference tables (foundation)

Reference tables are small tables that provide “meaning” or context.
They rarely change and are used in joins.

---

## 1.1 `regionen_bundesland` (Federal states list)

### What it represents
A clean list of all German **Bundesländer** (federal states).

### Typical use cases
- Group revenue by Bundesland
- Filter results to a specific Bundesland
- Show a list of states for UI dropdowns

### Important columns
- `bundesland_id` – internal numeric id
- `bundesland_name` – readable name (e.g., “Bayern”)
- `bundesland_code2` – **2-digit key** used for joins (e.g., “09”)
- `iso_code` – ISO code (e.g., “DE-BY”)

### How to join it
Most commonly you reach Bundesland through PLZ:

`ticket_verkaeufe.plz`
→ `postleitzahlen.plz`
→ `postleitzahlen.bundesland_code2`
→ `regionen_bundesland.bundesland_code2`

### Example questions
- DE: „Zeige alle Bundesländer.“
- EN: “List all federal states.”

---

## 1.2 `postleitzahlen` (Postal code mapping)

### What it represents
A mapping table from **PLZ** (postal codes) to:
- city (`ort`)
- Bundesland (`bundesland_code2`)

### Why this table is important
Your main sales table (`ticket_verkaeufe`) contains a **PLZ**, not a Bundesland directly.
So this table is the bridge that lets you answer questions like:
**“Revenue by Bundesland”**.

### Typical use cases
- “Which city/state does this PLZ belong to?”
- “Revenue by city”
- “Revenue by state” (via PLZ join)

### Important columns
- `plz` – 5-digit postal code (join key)
- `ort` – city name
- `bundesland_code2` – join key to `regionen_bundesland`

### How to join it
- `ticket_verkaeufe.plz` → `postleitzahlen.plz`
- `postleitzahlen.bundesland_code2` → `regionen_bundesland.bundesland_code2`

### Example questions
- DE: „Welche PLZ gehören zu Bayern?“
- EN: “Which postal codes belong to Bavaria?”

---

## 1.3 `ticket_produkte` (Ticket product catalog)

### What it represents
A small catalog of ticket types and their prices.
This is the “dictionary” for ticket codes.

### Typical use cases
- Explain ticket types in results
- Compare unit price in sales vs catalog price
- Filter by a ticket product (e.g., Deutschlandticket)

### Important columns
- `ticket_code` – join key used in sales
- `ticket_name` – readable product name
- `preis_eur` – official price (synthetic but realistic)

### How to join it
- `ticket_verkaeufe.ticket_code` → `ticket_produkte.ticket_code`

### Example questions
- DE: „Wie teuer ist das Deutschlandticket?“
- EN: “What is the price of the Deutschlandticket?”

---

## 1.4 `tarifverbuende` (Tariff associations / transport regions)

### What it represents
A list of “Tarifverbünde” (transport or tariff associations).
Sales and planning are grouped by these organizations.

### Typical use cases
- “Revenue by tariff association”
- “Top 10 tariff associations”
- Filtering: only active associations

### Important columns
- `tarifverbund_id` – join key
- `name` – readable name
- `kuerzel` – abbreviation (good for display)
- `status` – “aktiv” / “inaktiv”

### How to join it
- `ticket_verkaeufe.tarifverbund_id` → `tarifverbuende.tarifverbund_id`
- `plan_umsatz.tarifverbund_id` → `tarifverbuende.tarifverbund_id`

### Example questions
- DE: „Welche Tarifverbünde sind aktiv?“
- EN: “Which tariff associations are active?”

---

## 1.5 `meldestellen` (Reporting offices)

### What it represents
A “Meldestelle” is a reporting office that submits data rows.
This table describes those offices.

### Typical use cases
- “Which reporting office contributes most revenue?”
- “Revenue per reporting office”
- Check which offices belong to a synthetic organization

### Important columns
- `meldestelle_code` – join key used in sales
- `meldestelle_name` – readable name
- `organisation_id` – synthetic organization grouping

### How to join it
- `ticket_verkaeufe.meldestelle_code` → `meldestellen.meldestelle_code`

### Example questions
- DE: „Welche Meldestellen liefern den höchsten Umsatz?“
- EN: “Which reporting offices deliver the highest revenue?”

---

# 2) Fact tables (data you analyze)

Fact tables contain the “real business numbers” (synthetic in our dataset).
Most user questions are answered using these tables.

---

## 2.1 `ticket_verkaeufe` (Main sales table)

### What it represents
The **core analytics table**: each row represents ticket sales for a month/region context.

### Typical use cases
- Revenue analysis (totals, trends, top N)
- Comparisons between ticket types
- Joining to Bundesland (via PLZ)
- Joining to ticket products and tariff associations

### Important columns
- `monat`, `jahr` – reporting period
- `ticket_code` – ticket type
- `anzahl` – number of tickets
- `preis_eur` – unit price
- `umsatz_eur` – total revenue
- `plz` – used to derive city/state
- `tarifverbund_id`, `meldestelle_code` – organization dimensions

### How to join it (most common paths)
- Ticket name & catalog price:
  `ticket_verkaeufe.ticket_code` → `ticket_produkte.ticket_code`

- Tariff association name:
  `ticket_verkaeufe.tarifverbund_id` → `tarifverbuende.tarifverbund_id`

- Bundesland:
  `ticket_verkaeufe.plz` → `postleitzahlen.plz`
  → `regionen_bundesland.bundesland_code2`

- Reporting office details:
  `ticket_verkaeufe.meldestelle_code` → `meldestellen.meldestelle_code`

### Example questions
- DE: „Wie hoch ist der Umsatz pro Bundesland?“
- EN: “How much revenue per federal state?”
- DE: „Welche Ticketarten bringen den höchsten Umsatz?“
- EN: “Which ticket types generate the most revenue?”

---

## 2.2 `sonstige_angebote` (Other offers revenue)

### What it represents
Revenue from ticket groups that are **not** the main ticket product list.
Used for comparisons: “Deutschlandticket vs other products”.

### Typical use cases
- Compare revenue composition
- Trend analysis for “other offers”
- Revenue by tariff association

### Important columns
- `monat`, `jahr`
- `tarifverbund_id`
- `angebot_gruppe` – category identifier
- `umsatz_eur`

### How to join it
- `sonstige_angebote.tarifverbund_id` → `tarifverbuende.tarifverbund_id`

### Example questions
- DE: „Wie viel Umsatz kommt aus sonstigen Angeboten?“
- EN: “How much revenue comes from other offers?”

---

## 2.3 `plan_umsatz` (Planned / target revenue)

### What it represents
Targets / planned revenue (“Planwerte”) per month and tariff association.

### Typical use cases
- KPI reporting: compare actual vs planned
- “Which tariff association is above/below plan?”

### Important columns
- `monat`, `jahr`
- `tarifverbund_id`
- `umsatz_eur` – planned revenue

### How to join it
Compare planned vs actual using shared keys:

`plan_umsatz (monat, jahr, tarifverbund_id)`
↔ `ticket_verkaeufe (monat, jahr, tarifverbund_id)`

### Example questions
- DE: „Ist der Ist-Umsatz über oder unter Plan?“
- EN: “Is actual revenue above or below plan?”

---

# 3) Distribution / allocation tables (shares)

These tables describe how values could be “split” across Bundesländer using shares (`anteil`).
This is useful when a row doesn’t directly belong to just one state, but is allocated.

---

## 3.1 `ticket_verteilung_bundesland` (Ticket allocations by state)

### What it represents
Allocation shares (`anteil`) for ticket sales context:
- same tariff association
- same period
- same ticket code
but distributed across multiple Bundesländer.

### Typical use cases
- “Allocated revenue by Bundesland”
- Explaining distribution logic in a thesis demo

### Join keys
- `jahr`, `monat`, `tarifverbund_id`, `ticket_code`, `bundesland_code2`

### Example questions
- DE: „Wie verteilt sich Ticket-Umsatz auf Bundesländer?“
- EN: “How is ticket revenue distributed across states?”

---

## 3.2 `angebot_verteilung_bundesland`
Same idea as above, but for `sonstige_angebote`.

---

## 3.3 `plan_verteilung_bundesland`
Same idea as above, but for `plan_umsatz`.

---

## 3.4 `verteilung_kopf` + `verteilung_positionen` (Generic distribution model)

### What it represents
A generic “header/detail” model:
- `verteilung_kopf` describes a distribution rule (time range, type)
- `verteilung_positionen` lists which Bundesländer and their shares

### Typical use cases
- Demonstrate relational modeling (1-to-many)
- Show how distribution rules can be configured

### How to join it
- `verteilung_positionen.verteilung_id` → `verteilung_kopf.verteilung_id`
- `verteilung_positionen.bundesland_id` → `regionen_bundesland.bundesland_id`

---

# 4) Users + permissions (optional demo domain)

These tables exist to demonstrate:
- how user accounts can be linked to permissions
- how filters could work in a real system

Your Streamlit app can remain read-only. These are for analytics + demo queries.

---

## 4.1 `nutzer` (Users)

### What it represents
Synthetic user accounts (names are plausible but not real people).

### Typical use cases
- “List users”
- Join to permission tables
- Count how many users have certain rights

### Important columns
- `nutzer_id` – join key
- `benutzername`, `email` – synthetic identifiers

---

## 4.2 Permission tables (who is allowed to do what)

These tables store boolean flags like:
- may view data
- may download data
- may manage access
- etc.

### Tables
- `nutzer_bundesland_rechte` – rights per Bundesland
- `nutzer_tarifverbund_rechte` – rights per tariff association
- `nutzer_zast_rechte` – rights per ZaSt entity (synthetic)
- `nutzer_zentral_zast_rechte` – rights per central entity (synthetic)

### Example questions
- DE: „Welche Nutzer dürfen herunterladen?“
- EN: “Which users are allowed to download?”

---

# Recommended thesis demo prompts (simple but impressive)

1) **Join: Revenue by state**
- DE: „Zeige den Umsatz nach Bundesland für 2025.“
- EN: “Show revenue by federal state for 2025.”

2) **Top organizations**
- DE: „Welche Tarifverbünde haben den höchsten Ticket-Umsatz?“
- EN: “Which tariff associations have the highest ticket revenue?”

3) **Actual vs plan**
- DE: „Vergleiche Ist-Umsatz mit Plan-Umsatz pro Monat.“
- EN: “Compare actual vs planned revenue per month.”

4) **Explainability**
- DE: „Erkläre mir die Tabelle ticket_verkaeufe.“
- EN: “Explain the ticket_verkaeufe table.”
