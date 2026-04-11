"""
Guardrails, deterministic template planner, LLM retry pipeline,
and failure-classification helpers for SQL generation.
"""

from __future__ import annotations

import re
from typing import Any, Callable, MutableMapping, Optional

WRITE_PATTERNS = [
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\balter\b",
    r"\btruncate\b",
    r"\bmerge\b",
    r"\bgrant\b",
    r"\brevoke\b",
]

_OUT_OF_SCOPE_MSG = (
    "This chatbot is designed only for database and SQL-related questions. "
    "Please ask about tables, queries, reports, filters, joins, or other database tasks."
)

# Patterns that unambiguously signal out-of-scope requests
_OUT_OF_SCOPE_PATTERNS = [
    # Greetings / small talk (English)
    r"\bhow are you\b",
    r"^(hi+|hello|hey|sup|yo)\b",
    r"\bgood (morning|afternoon|evening|night)\b",
    r"^(thanks?|thank you)\b",
    # Greetings / small talk (German)
    r"^(hallo|hei|moin|servus|grüß gott|guten (morgen|tag|abend|nacht))\b",
    r"\bwie geht(s| es)\b",
    r"^(danke|vielen dank|tschüss|auf wiedersehen)\b",
    # Content creation (English)
    r"\b(write|compose|draft|create)\b.{0,40}\b(essay|thesis|paper|story|email|letter|poem|song|lyrics|joke|blog|article)\b",
    r"\bessay\b",
    r"\bthesis\b",
    r"\bpoem\b",
    r"\bsong\b",
    r"\blyrics\b",
    r"\bjoke\b",
    r"\bstory\b",
    # Content creation (German)
    r"\b(schreib|verfasse|erstelle)\b.{0,40}\b(aufsatz|gedicht|lied|witz|geschichte|artikel|brief|email)\b",
    r"\bgedicht\b",
    r"\bwitz\b",
    # General knowledge / off-domain (English)
    r"\bweather\b",
    r"\bhomework\b",
    r"\bquantum\b",
    r"\bimage classification\b",
    r"\bdeep learning\b",
    r"\bneural network\b",
    r"\belection\b",
    r"\brecipe\b",
    r"\bcooking\b",
    # General knowledge / off-domain (German)
    r"\bwetter\b",
    r"\bhausaufgaben\b",
    r"\brezept\b",
    r"\bkochen\b",
    r"\bwahl\b",
    r"\bnachrichten\b",
    r"\bpolitik\b",
    # Non-SQL coding
    r"\bjavascript\b",
    r"\bnode\.?js\b",
    r"\bdjango\b",
    r"\bflask\b",
    r"\breact\b",
    r"\bcss\b",
    r"\bhtml\b",
]

# Positive signals that confirm the request is within DB/SQL scope
_SCOPE_SIGNALS = [
    "select", "from", "where", "join", "group by", "order by", "having", "with",
    "sum(", "count(", "avg(", "min(", "max(",
    "revenue", "umsatz", "sales", "verkauf",
    "ticket", "tariff", "tarif", "tarifverbund",
    "bundesland", "state", "region", "plz", "postal",
    "month", "monat", "year", "jahr",
    "total", "gesamt",
    "table", "column", "schema", "query", "sql", "database", "filter",
    "aggregate", "report", "meldestelle", "postleitzahl",
    "how many", "how much", "ranking", "top", "bottom", "average",
    "chart", "breakdown", "distribution",
]


def extract_requested_years(question: str) -> list[int]:
    return [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question or "")]


def extract_requested_year(question: str) -> Optional[int]:
    years = extract_requested_years(question)
    if not years:
        return None
    return years[0]


def nearest_year(target: int, available: list[int]) -> Optional[int]:
    """Return the year from available closest to target, or None if the list is empty."""
    if not available:
        return None
    return min(available, key=lambda y: abs(y - target))


def contains_any(text: str, terms: list[str]) -> bool:
    q = (text or "").lower()
    return any(term in q for term in terms)


def build_year_filter(question: str, alias: str = "tv") -> str:
    requested_year = extract_requested_year(question)
    if requested_year is None:
        return ""
    return f"WHERE {alias}.jahr = {requested_year}"


_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3,
    "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
}

_QUARTER_MAP = {
    "q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12),
    "quarter 1": (1, 3), "quarter 2": (4, 6), "quarter 3": (7, 9), "quarter 4": (10, 12),
    "1st quarter": (1, 3), "2nd quarter": (4, 6), "3rd quarter": (7, 9), "4th quarter": (10, 12),
}

# German state name aliases → canonical DB values
_STATE_ALIASES: dict[str, str] = {
    "bavaria": "Bayern", "bayern": "Bayern",
    "berlin": "Berlin",
    "hamburg": "Hamburg",
    "bremen": "Bremen",
    "saxony": "Sachsen", "sachsen": "Sachsen",
    "thuringia": "Thüringen", "thüringen": "Thüringen", "thuringen": "Thüringen",
    "hesse": "Hessen", "hessen": "Hessen",
    "nrw": "Nordrhein-Westfalen", "north rhine-westphalia": "Nordrhein-Westfalen",
    "nordrhein-westfalen": "Nordrhein-Westfalen",
    "brandenburg": "Brandenburg",
    "saarland": "Saarland",
    "rhineland-palatinate": "Rheinland-Pfalz", "rheinland-pfalz": "Rheinland-Pfalz",
    "lower saxony": "Niedersachsen", "niedersachsen": "Niedersachsen",
    "mecklenburg": "Mecklenburg-Vorpommern",
    "schleswig-holstein": "Schleswig-Holstein",
    "saxony-anhalt": "Sachsen-Anhalt", "sachsen-anhalt": "Sachsen-Anhalt",
    "baden-württemberg": "Baden-Württemberg", "baden-wuerttemberg": "Baden-Württemberg",
    "bad württemberg": "Baden-Württemberg",
}


def extract_state_names(question: str) -> list[str]:
    """Extract up to 2 canonical state names from the question."""
    q = (question or "").lower()
    found = []
    for alias, canonical in _STATE_ALIASES.items():
        if alias in q and canonical not in found:
            found.append(canonical)
        if len(found) == 2:
            break
    return found


def extract_requested_month(question: str) -> Optional[int]:
    q = (question or "").lower()
    for name, num in _MONTH_NAMES.items():
        if name in q:
            return num
    m = re.search(r"\bmonth\s+(\d{1,2})\b", q)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 12:
            return val
    return None


def extract_requested_quarter(question: str) -> Optional[tuple[int, int]]:
    q = (question or "").lower()
    for label, months in _QUARTER_MAP.items():
        if label in q:
            return months
    return None


def extract_top_n(question: str, default: int = 10) -> int:
    m = re.search(r"\btop\s+(\d+)\b|\bbottom\s+(\d+)\b|\b(\d+)\s+postal\b|\b(\d+)\s+state\b|\b(\d+)\s+ticket\b|\b(\d+)\s+tariff\b|\b(\d+)\s+cit", (question or "").lower())
    if m:
        val = next(v for v in m.groups() if v is not None)
        return int(val)
    return default


def build_template_sql(question: str) -> tuple[str, str]:
    """
    Deterministic planner for common analytics questions.
    Returns (sql, note). Empty sql means no template matched.
    """
    q = (question or "").lower()
    wants_revenue = contains_any(q, ["revenue", "umsatz", "sales"])
    wants_state = contains_any(q, ["state", "federal state", "bundesland", "states"])
    wants_month = contains_any(q, ["month", "monat", "monthly"])
    wants_ticket_type = contains_any(
        q, ["ticket type", "ticket types", "ticketart", "ticketarten", "ticket product", "ticket name"]
    )
    wants_tariff = contains_any(
        q,
        [
            "tariff",
            "tariff association",
            "tariff associations",
            "tarif",
            "tarifverbund",
            "tarifverbunde",
            "tarifverbuende",
            "verbund",
            "association",
        ],
    )
    wants_total = contains_any(q, ["total", "gesamt", "overall"])
    wants_quantity = contains_any(q, ["quantity", "anzahl", "sold", "tickets sold", "ticket sales", "how many tickets"])
    wants_city = contains_any(q, ["city", "cities", "ort", "städte"])
    wants_postal = contains_any(q, ["postal", "plz", "postal code", "postleitzahl"])
    wants_plan = contains_any(q, ["plan", "planned", "actual", "deviation", "above plan", "below plan", "vs plan"])
    wants_quarter = extract_requested_quarter(question) is not None
    wants_top = contains_any(q, ["top", "highest", "most", "best", "largest", "bottom", "lowest", "least", "worst"])

    specific_month = extract_requested_month(question)
    specific_year = extract_requested_year(question)
    quarter_months = extract_requested_quarter(question)
    top_n = extract_top_n(q)

    # Revenue by tariff association + state + month
    if wants_revenue and wants_tariff and wants_state and wants_month:
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
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
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tv.monat, t.name, rb.bundesland_name
ORDER BY tv.jahr, tv.monat, t.name, rb.bundesland_name
""".strip()
        return (
            sql,
            "Tables used: ticket_verkaeufe → tarifverbuende, postleitzahlen → regionen_bundesland",
        )

    # Revenue by tariff association + state
    if wants_revenue and wants_tariff and wants_state:
        sql = f"""
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
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, t.name, rb.bundesland_name
ORDER BY tv.jahr, t.name, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Tables used: ticket_verkaeufe → tarifverbuende, postleitzahlen → regionen_bundesland",
        )

    # Revenue by tariff association + month
    if wants_revenue and wants_tariff and wants_month:
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    t.name AS tarifverbund_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN tarifverbuende t
    ON tv.tarifverbund_id = t.tarifverbund_id
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tv.monat, t.name
ORDER BY tv.jahr, tv.monat, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Tables used: ticket_verkaeufe → tarifverbuende",
        )

    # Revenue by tariff association
    if wants_revenue and wants_tariff:
        sql = f"""
SELECT
    tv.jahr,
    t.name AS tarifverbund_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN tarifverbuende t
    ON tv.tarifverbund_id = t.tarifverbund_id
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, t.name
ORDER BY tv.jahr, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Tables used: ticket_verkaeufe → tarifverbuende",
        )

    # Revenue by state + month
    if wants_revenue and wants_state and wants_month:
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tv.monat, rb.bundesland_name
ORDER BY tv.jahr, tv.monat, rb.bundesland_name
""".strip()
        return (
            sql,
            "Tables used: ticket_verkaeufe → postleitzahlen → regionen_bundesland",
        )

    # Revenue by state
    if wants_revenue and wants_state:
        sql = f"""
SELECT
    tv.jahr,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p
    ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb
    ON p.bundesland_code2 = rb.bundesland_code2
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, rb.bundesland_name
ORDER BY tv.jahr, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Tables used: ticket_verkaeufe → postleitzahlen → regionen_bundesland",
        )

    # Revenue by month
    if wants_revenue and wants_month:
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tv.monat
ORDER BY tv.jahr, tv.monat
""".strip()
        return (
            sql,
            "Tables used: ticket_verkaeufe",
        )

    # Revenue by ticket type
    if wants_revenue and wants_ticket_type:
        sql = f"""
SELECT
    tv.jahr,
    tp.ticket_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN ticket_produkte tp
    ON tv.ticket_code = tp.ticket_code
{build_year_filter(question, alias="tv")}
GROUP BY tv.jahr, tp.ticket_name
ORDER BY tv.jahr, umsatz_eur DESC
""".strip()
        return (
            sql,
            "Tables used: ticket_verkaeufe → ticket_produkte",
        )

    # Total revenue
    if wants_revenue and wants_total:
        sql = f"""
SELECT
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
{build_year_filter(question, alias="tv")}
""".strip()
        return (sql, "Tables used: ticket_verkaeufe")

    # Revenue by ticket type + state
    if wants_revenue and wants_ticket_type and wants_state:
        year_filter = f"WHERE tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT
    tp.ticket_name,
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code
JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2
{year_filter}
GROUP BY tp.ticket_name, rb.bundesland_name
ORDER BY tp.ticket_name, umsatz_eur DESC
""".strip()
        return (sql, "Tables used: ticket_verkaeufe → ticket_produkte, postleitzahlen → regionen_bundesland")

    # Revenue by ticket type + month
    if wants_revenue and wants_ticket_type and wants_month:
        year_filter = f"WHERE tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    tp.ticket_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code
{year_filter}
GROUP BY tv.jahr, tv.monat, tp.ticket_name
ORDER BY tv.jahr, tv.monat, umsatz_eur DESC
""".strip()
        return (sql, "Tables used: ticket_verkaeufe → ticket_produkte")

    # Quantity (tickets sold) by state
    if wants_quantity and wants_state:
        year_filter = f"WHERE tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT
    rb.bundesland_name,
    SUM(tv.anzahl) AS anzahl
FROM ticket_verkaeufe tv
JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2
{year_filter}
GROUP BY rb.bundesland_name
ORDER BY anzahl DESC
""".strip()
        return (sql, "Tables used: ticket_verkaeufe → postleitzahlen → regionen_bundesland")

    # Quantity by ticket type
    if wants_quantity and wants_ticket_type:
        year_filter = f"WHERE tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT
    tp.ticket_name,
    SUM(tv.anzahl) AS anzahl
FROM ticket_verkaeufe tv
JOIN ticket_produkte tp ON tv.ticket_code = tp.ticket_code
{year_filter}
GROUP BY tp.ticket_name
ORDER BY anzahl DESC
""".strip()
        return (sql, "Tables used: ticket_verkaeufe → ticket_produkte")

    # Quantity by month
    if wants_quantity and wants_month:
        year_filter = f"WHERE tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    SUM(tv.anzahl) AS anzahl
FROM ticket_verkaeufe tv
{year_filter}
GROUP BY tv.jahr, tv.monat
ORDER BY tv.jahr, tv.monat
""".strip()
        return (sql, "Tables used: ticket_verkaeufe")

    # Total quantity
    if wants_quantity:
        year_filter = f"WHERE tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT SUM(tv.anzahl) AS anzahl
FROM ticket_verkaeufe tv
{year_filter}
""".strip()
        return (sql, "Tables used: ticket_verkaeufe")

    # Revenue by city
    if wants_revenue and wants_city:
        year_filter = f"WHERE tv.jahr = {specific_year}" if specific_year else ""
        limit = f"LIMIT {top_n}" if wants_top else ""
        sql = f"""
SELECT
    p.ort,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz
{year_filter}
GROUP BY p.ort
ORDER BY umsatz_eur DESC
{limit}
""".strip()
        return (sql, "Tables used: ticket_verkaeufe → postleitzahlen")

    # Revenue by postal code (top N)
    if wants_revenue and wants_postal:
        year_filter = f"WHERE tv.jahr = {specific_year}" if specific_year else ""
        limit = f"LIMIT {top_n}" if wants_top else "LIMIT 10"
        sql = f"""
SELECT
    tv.plz,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
{year_filter}
GROUP BY tv.plz
ORDER BY umsatz_eur DESC
{limit}
""".strip()
        return (sql, "Tables used: ticket_verkaeufe")

    # Revenue for specific month + year
    if wants_revenue and specific_month and specific_year:
        sql = f"""
SELECT
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2
WHERE tv.jahr = {specific_year} AND tv.monat = {specific_month}
GROUP BY rb.bundesland_name
ORDER BY umsatz_eur DESC
""".strip()
        return (sql, "Tables used: ticket_verkaeufe → postleitzahlen → regionen_bundesland")

    # Revenue by quarter
    if wants_revenue and wants_quarter and quarter_months:
        m_from, m_to = quarter_months
        year_filter = f"AND tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
WHERE tv.monat BETWEEN {m_from} AND {m_to}
{year_filter}
GROUP BY tv.jahr, tv.monat
ORDER BY tv.jahr, tv.monat
""".strip()
        return (sql, "Tables used: ticket_verkaeufe")

    # Plan vs actual comparison
    if wants_plan and wants_month:
        year_filter = f"AND tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT
    tv.jahr,
    tv.monat,
    SUM(tv.umsatz_eur) AS actual_eur,
    SUM(p.umsatz_eur) AS plan_eur,
    SUM(tv.umsatz_eur) - SUM(p.umsatz_eur) AS deviation_eur
FROM ticket_verkaeufe tv
JOIN plan_umsatz p
    ON tv.tarifverbund_id = p.tarifverbund_id
    AND tv.monat = p.monat
    AND tv.jahr = p.jahr
WHERE 1=1 {year_filter}
GROUP BY tv.jahr, tv.monat
ORDER BY tv.jahr, tv.monat
""".strip()
        return (sql, "Tables used: ticket_verkaeufe, plan_umsatz")

    if wants_plan and wants_tariff:
        year_filter = f"AND tv.jahr = {specific_year}" if specific_year else ""
        sql = f"""
SELECT
    t.name AS tarifverbund_name,
    SUM(tv.umsatz_eur) AS actual_eur,
    SUM(p.umsatz_eur) AS plan_eur,
    SUM(tv.umsatz_eur) - SUM(p.umsatz_eur) AS deviation_eur
FROM ticket_verkaeufe tv
JOIN plan_umsatz p
    ON tv.tarifverbund_id = p.tarifverbund_id
    AND tv.monat = p.monat
    AND tv.jahr = p.jahr
JOIN tarifverbuende t ON tv.tarifverbund_id = t.tarifverbund_id
WHERE 1=1 {year_filter}
GROUP BY t.name
ORDER BY deviation_eur DESC
""".strip()
        return (sql, "Tables used: ticket_verkaeufe, plan_umsatz → tarifverbuende")

    # Year-over-year comparison
    if wants_revenue and contains_any(q, ["compare", "vs", "versus", "growth", "grew", "year-over-year", "yoy", "2024", "2025"]) and wants_month:
        sql = """
SELECT
    tv.monat,
    SUM(CASE WHEN tv.jahr = 2024 THEN tv.umsatz_eur ELSE 0 END) AS umsatz_2024,
    SUM(CASE WHEN tv.jahr = 2025 THEN tv.umsatz_eur ELSE 0 END) AS umsatz_2025,
    SUM(CASE WHEN tv.jahr = 2025 THEN tv.umsatz_eur ELSE 0 END)
        - SUM(CASE WHEN tv.jahr = 2024 THEN tv.umsatz_eur ELSE 0 END) AS difference_eur
FROM ticket_verkaeufe tv
WHERE tv.jahr IN (2024, 2025)
GROUP BY tv.monat
ORDER BY tv.monat
""".strip()
        return (sql, "Tables used: ticket_verkaeufe")

    # State comparison (e.g. Bavaria vs Berlin)
    state_names = extract_state_names(question)
    if wants_revenue and len(state_names) == 2:
        year_filter = f"AND tv.jahr = {specific_year}" if specific_year else ""
        s1, s2 = state_names[0], state_names[1]
        sql = f"""
SELECT
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2
WHERE rb.bundesland_name IN ('{s1}', '{s2}') {year_filter}
GROUP BY rb.bundesland_name
ORDER BY umsatz_eur DESC
""".strip()
        return (sql, f"Tables used: ticket_verkaeufe → postleitzahlen → regionen_bundesland")

    # Single state filter (e.g. revenue in Bavaria by month)
    if wants_revenue and len(state_names) == 1:
        state = state_names[0]
        year_filter = f"AND tv.jahr = {specific_year}" if specific_year else ""
        group_by = "tv.monat" if wants_month else "rb.bundesland_name"
        order_by = "tv.monat" if wants_month else "umsatz_eur DESC"
        select_extra = "tv.monat," if wants_month else ""
        sql = f"""
SELECT
    {select_extra}
    rb.bundesland_name,
    SUM(tv.umsatz_eur) AS umsatz_eur
FROM ticket_verkaeufe tv
JOIN postleitzahlen p ON CAST(tv.plz AS VARCHAR) = p.plz
JOIN regionen_bundesland rb ON p.bundesland_code2 = rb.bundesland_code2
WHERE rb.bundesland_name = '{state}' {year_filter}
GROUP BY {group_by}, rb.bundesland_name
ORDER BY {order_by}
""".strip()
        return (sql, f"Tables used: ticket_verkaeufe → postleitzahlen → regionen_bundesland")

    return "", ""


def extract_sql_only(raw_text: str) -> str:
    """
    Keep only SQL from model output.
    Returns empty string if SQL cannot be extracted.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    code_block = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if code_block:
        return code_block.group(1).strip().rstrip(";")

    select_or_with = re.search(r"\b(select|with)\b[\s\S]*", text, re.IGNORECASE)
    if select_or_with:
        candidate = select_or_with.group(0).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
        return candidate.rstrip(";")

    return ""


def extract_did_you_mean_candidates(error_text: str, schema_tree: dict) -> list[str]:
    if not error_text:
        return []
    quoted = re.findall(r'"([^"]+)"', error_text)
    schema_tables = set(schema_tree.keys())
    return [item for item in quoted if item in schema_tables]


def guess_relevant_tables(question: str, schema_tree: dict) -> list[str]:
    q_lower = (question or "").lower()
    candidates = []

    if any(token in q_lower for token in ["revenue", "umsatz", "sales", "verkauf"]):
        candidates.extend(["ticket_verkaeufe", "plan_umsatz", "sonstige_angebote"])
    if any(token in q_lower for token in ["state", "bundesland", "region"]):
        candidates.extend(["postleitzahlen", "regionen_bundesland"])
    if any(token in q_lower for token in ["plz", "postal", "postleitzahl"]):
        candidates.append("postleitzahlen")
    if any(token in q_lower for token in ["ticket", "type", "produkt"]):
        candidates.extend(["ticket_verkaeufe", "ticket_produkte"])
    if any(token in q_lower for token in ["tariff", "tarif", "association", "verbund"]):
        candidates.append("tarifverbuende")
    if any(token in q_lower for token in ["meldestelle", "reporting office"]):
        candidates.append("meldestellen")

    # Keep order, remove duplicates, only existing tables.
    result = []
    seen = set()
    for table in candidates:
        if table in schema_tree and table not in seen:
            seen.add(table)
            result.append(table)

    if not result:
        # fallback to core fact + dimension tables
        for table in [
            "ticket_verkaeufe",
            "postleitzahlen",
            "regionen_bundesland",
            "ticket_produkte",
            "tarifverbuende",
        ]:
            if table in schema_tree and table not in seen:
                seen.add(table)
                result.append(table)

    return result[:8]


def build_join_hints(question: str) -> str:
    q_lower = (question or "").lower()
    hints = []

    if any(token in q_lower for token in ["state", "bundesland", "region"]):
        hints.append(
            "For state-level analytics, join "
            "ticket_verkaeufe.plz = postleitzahlen.plz, then "
            "postleitzahlen.bundesland_code2 = regionen_bundesland.bundesland_code2."
        )
    if any(token in q_lower for token in ["ticket", "type", "produkt"]):
        hints.append(
            "For ticket type details, join "
            "ticket_verkaeufe.ticket_code = ticket_produkte.ticket_code."
        )
    if any(token in q_lower for token in ["tariff", "tarif", "association", "verbund"]):
        hints.append(
            "For tariff association details, join "
            "ticket_verkaeufe.tarifverbund_id = tarifverbuende.tarifverbund_id."
        )
    if any(token in q_lower for token in ["meldestelle", "reporting office"]):
        hints.append(
            "For reporting office details, join "
            "ticket_verkaeufe.meldestelle_code = meldestellen.meldestelle_code."
        )

    return "\n".join(f"- {hint}" for hint in hints)


def build_retry_prompt(
    question: str, bad_sql: str, compile_error: str, schema_tree: dict
) -> str:
    relevant_tables = guess_relevant_tables(question, schema_tree)
    did_you_mean_tables = extract_did_you_mean_candidates(compile_error, schema_tree)
    for table in did_you_mean_tables:
        if table not in relevant_tables:
            relevant_tables.append(table)

    table_context_lines = []
    for table in relevant_tables:
        cols = schema_tree.get(table, [])
        col_names = ", ".join(col for col, _ in cols[:18])
        table_context_lines.append(f"- {table}({col_names})")
    table_context = "\n".join(table_context_lines)

    join_hints = build_join_hints(question)

    return (
        "Generate DuckDB SQL for mxquerychat.\n"
        "Use ONLY the tables and columns listed below.\n"
        "Return ONLY SQL (no explanation, no markdown).\n"
        "Use SELECT or WITH only.\n\n"
        f"Original question:\n{question}\n\n"
        f"Previous SQL failed:\n{bad_sql}\n\n"
        f"DuckDB error:\n{compile_error}\n\n"
        "Available relevant tables and columns:\n"
        f"{table_context}\n\n"
        "Join hints:\n"
        f"{join_hints if join_hints else '- Use the most suitable joins based on common keys.'}\n"
    )


def build_first_pass_prompt(question: str, schema_tree: dict) -> str:
    """First model call prompt: schema-guided from the start for higher accuracy."""
    relevant_tables = guess_relevant_tables(question, schema_tree)
    table_context_lines = []
    for table in relevant_tables:
        cols = schema_tree.get(table, [])
        col_names = ", ".join(col for col, _ in cols[:20])
        table_context_lines.append(f"- {table}({col_names})")
    table_context = "\n".join(table_context_lines)
    join_hints = build_join_hints(question)

    return (
        "Generate DuckDB SQL for mxquerychat.\n"
        "Use ONLY the tables and columns listed below.\n"
        "Return ONLY SQL (no explanation, no markdown).\n"
        "Use SELECT or WITH only.\n\n"
        f"Question:\n{question}\n\n"
        "Available relevant tables and columns:\n"
        f"{table_context}\n\n"
        "Join hints:\n"
        f"{join_hints if join_hints else '- Use the most suitable joins based on common keys.'}\n"
    )


def build_final_retry_prompt(
    question: str, bad_sql: str, compile_error: str, schema_tree: dict
) -> str:
    """Final strict prompt: force SQL-only or explicit NO_MATCH token."""
    relevant_tables = guess_relevant_tables(question, schema_tree)
    table_context_lines = []
    for table in relevant_tables:
        cols = schema_tree.get(table, [])
        col_names = ", ".join(col for col, _ in cols[:20])
        table_context_lines.append(f"- {table}({col_names})")
    table_context = "\n".join(table_context_lines)
    join_hints = build_join_hints(question)

    return (
        "You are generating SQL for mxquerychat on DuckDB.\n"
        "Rules:\n"
        "1) Use ONLY the listed tables and columns.\n"
        "2) Output ONLY one SQL query using SELECT/WITH, no markdown.\n"
        "3) If the question cannot be answered from this schema, output exactly: NO_MATCH\n\n"
        f"Question:\n{question}\n\n"
        f"Previous failed SQL:\n{bad_sql if bad_sql else '-- no valid SQL produced --'}\n\n"
        f"Compilation/validation issue:\n{compile_error}\n\n"
        "Available tables and columns:\n"
        f"{table_context}\n\n"
        "Join hints:\n"
        f"{join_hints if join_hints else '- Use the most suitable joins based on shared keys.'}\n"
    )


def generate_sql_with_retry(
    generate_sql_fn: Callable[[str], str],
    question_text: str,
    schema_tree: dict,
    compile_sql_fn: Callable[[str], str],
    timeout_seconds: int,
    run_with_timeout_fn: Callable[[Callable[[], Any], int], tuple[Any, Any]],
) -> tuple[str, list[str], str]:
    """
    Returns: (sql, notes, error_code)
    sql is empty when no valid SQL could be generated.
    error_code: "", "timeout", "model_error", "no_match"
    """
    notes: list[str] = []

    def run_attempt(prompt_text: str, attempt_name: str) -> tuple[str, str, str]:
        """
        Returns: (sql, status, detail)
        status: "ok", "timeout", "model_error", "non_sql", "no_match", "compile_error"
        """
        raw_sql, call_error = run_with_timeout_fn(
            lambda: generate_sql_fn(prompt_text),
            timeout_seconds,
        )
        if call_error:
            call_error_text = str(call_error)
            if "timeout" in call_error_text.lower():
                notes.append(f"{attempt_name}: model timeout ({call_error_text}).")
                return "", "timeout", call_error_text
            if "hnsw" in call_error_text.lower() or "nothing found on disk" in call_error_text.lower() or "segment reader" in call_error_text.lower():
                notes.append(f"{attempt_name}: vector store not trained yet.")
                return "", "not_trained", call_error_text
            notes.append(f"{attempt_name}: model error ({call_error_text}).")
            return "", "model_error", call_error_text

        raw_text = (raw_sql or "").strip()
        if raw_text.upper() == "NO_MATCH":
            notes.append(f"{attempt_name}: model returned NO_MATCH.")
            return "", "no_match", "NO_MATCH"

        sql = extract_sql_only(raw_text)
        if not sql:
            notes.append(f"{attempt_name}: model returned non-SQL output.")
            return "", "non_sql", raw_text[:120]

        try:
            compile_error = compile_sql_fn(sql)
        except Exception as exc:
            compile_error = str(exc)

        if compile_error:
            notes.append(f"{attempt_name}: SQL validation failed, trying stricter prompt.")
            return sql, "compile_error", compile_error

        notes.append(f"{attempt_name}: valid SQL generated.")
        return sql, "ok", ""

    first_prompt = build_first_pass_prompt(question_text, schema_tree)
    sql_1, status_1, detail_1 = run_attempt(first_prompt, "Attempt 1 (schema-guided)")
    if status_1 == "ok":
        return sql_1, notes, ""
    if status_1 == "timeout":
        return "", notes, "timeout"
    if status_1 in {"model_error", "not_trained"}:
        return "", notes, status_1

    retry_prompt = build_retry_prompt(
        question_text,
        sql_1 or "-- no valid SQL produced --",
        detail_1 or "No compilable SQL in attempt 1.",
        schema_tree,
    )
    sql_2, status_2, detail_2 = run_attempt(
        retry_prompt, "Attempt 2 (schema-aware retry)"
    )
    if status_2 == "ok":
        notes.append("Resolved using related tables and join hints.")
        return sql_2, notes, ""
    if status_2 == "timeout":
        return "", notes, "timeout"
    if status_2 == "model_error":
        return "", notes, "model_error"

    final_prompt = build_final_retry_prompt(
        question_text,
        sql_2 or sql_1,
        detail_2 or detail_1 or "No compilable SQL in previous attempts.",
        schema_tree,
    )
    sql_3, status_3, _ = run_attempt(final_prompt, "Attempt 3 (final strict attempt)")
    if status_3 == "ok":
        notes.append("Resolved on final strict attempt.")
        return sql_3, notes, ""
    if status_3 == "timeout":
        return "", notes, "timeout"
    if status_3 == "model_error":
        return "", notes, "model_error"

    notes.append("All generation strategies failed: no reliable SQL from available schema.")
    return "", notes, "no_match"


def _has_scope_signal(q: str) -> bool:
    """Return True if the question contains at least one DB/SQL scope signal."""
    return any(signal in q for signal in _SCOPE_SIGNALS)


def get_local_guardrail_message(question: str) -> str:
    """
    Classify the question and return a rejection/clarification message or "".

    Categories (in priority order):
    - Empty / too short     → prompt for more detail
    - Write intent          → block with read-only message
    - OUT_OF_SCOPE          → short rejection message
    - UNCLEAR               → short clarification question
    - IN_SCOPE (SQL/DB)     → "" (allow through to generation)
    """
    if not question or not question.strip():
        return "Please enter a question."

    q = question.strip().lower()

    if len(q.split()) < 2:
        return "Please write a fuller data question."

    # Write intent — always block regardless of scope
    if any(re.search(pattern, q) for pattern in WRITE_PATTERNS):
        return "Read-only mode: write operations are not allowed."

    # OUT_OF_SCOPE — unambiguous non-database request
    if any(re.search(pattern, q) for pattern in _OUT_OF_SCOPE_PATTERNS):
        return _OUT_OF_SCOPE_MSG

    # UNCLEAR — short/vague with no scope signal
    vague_only_phrases = [
        "show me the data",
        "give me the report",
        "give me data",
        "i need information",
        "i need data",
        "get me the data",
        "i want to see",
        "can you show",
        "help me",
    ]
    is_vague = any(phrase in q for phrase in vague_only_phrases) or len(q.split()) <= 3
    if is_vague and not _has_scope_signal(q):
        return (
            "Which data do you need? "
            "Please specify the table, metric, filters, or time period you are interested in."
        )

    # IN_SCOPE — allow through
    return ""


def explain_sql_brief(sql: str) -> str:
    """Return a specific plain-language summary of what this SQL query does."""
    cleaned = (sql or "").strip()
    if not cleaned:
        return "No SQL to explain yet."

    lowered = cleaned.lower()

    # --- metric (what is being measured) ---
    metric = "data"
    sum_match = re.search(r"\bsum\s*\(([^)]+)\)\s+as\s+(\w+)", lowered)
    count_match = re.search(r"\bcount\s*\(([^)]*)\)\s+as\s+(\w+)", lowered)
    avg_match = re.search(r"\bavg\s*\(([^)]+)\)\s+as\s+(\w+)", lowered)
    if sum_match:
        metric = sum_match.group(2).replace("_", " ")
    elif count_match:
        metric = count_match.group(2).replace("_", " ") + " count"
    elif avg_match:
        metric = "average " + avg_match.group(2).replace("_", " ")

    # --- dimensions (GROUP BY columns) ---
    group_match = re.search(r"\bgroup\s+by\s+([\w\s,\.]+?)(?:\bhaving\b|\border\b|\blimit\b|$)", lowered)
    dimensions: list[str] = []
    if group_match:
        raw_cols = group_match.group(1).strip()
        for col in re.split(r",\s*", raw_cols):
            col = col.strip()
            # strip table alias prefix (e.g. "rb.bundesland_name" → "bundesland name")
            col = re.sub(r"^\w+\.", "", col)
            col = col.replace("_", " ").strip()
            if col:
                dimensions.append(col)

    # --- year filter ---
    year_match = re.search(r"\bjahr\s*=\s*(\d{4})", lowered)
    year_filter = year_match.group(1) if year_match else None

    # --- build sentence ---
    parts: list[str] = []

    verb = "Shows" if not any(f in lowered for f in ("sum(", "count(", "avg(", "min(", "max(")) else "Calculates"
    if dimensions:
        parts.append(f"{verb} {metric} broken down by {', '.join(dimensions)}.")
    else:
        parts.append(f"{verb} {metric}.")

    if year_filter:
        parts.append(f"Filtered to year {year_filter}.")

    if "order by" in lowered:
        # extract ORDER BY column(s)
        order_match = re.search(r"\border\s+by\s+([\w\s,\.]+?)(?:\blimit\b|$)", lowered)
        if order_match:
            order_cols = order_match.group(1).strip()
            order_col = re.split(r",\s*", order_cols)[0]
            order_col = re.sub(r"\b(asc|desc)\b", "", order_col).strip()
            order_col = re.sub(r"^\w+\.", "", order_col).replace("_", " ").strip()
            direction = "descending" if "desc" in order_match.group(1) else "ascending"
            if order_col:
                parts.append(f"Sorted by {order_col} ({direction}).")

    return " ".join(parts)


def classify_generation_failure(error_code: str) -> str:
    """Map generation error codes to standardized failure categories."""
    normalized = (error_code or "").strip().lower()
    if normalized == "timeout":
        return "timeout"
    if normalized == "no_match":
        return "no_match"
    if normalized == "compile_fail":
        return "compile_fail"
    if normalized in {"model_error", "runtime_fail"}:
        return "runtime_fail"
    return "runtime_fail"


def classify_execution_failure(error_text: str) -> str:
    """Classify execution-stage failures into stable categories."""
    lowered = (error_text or "").lower()
    if "timeout" in lowered:
        return "timeout"

    compile_markers = [
        "parser error",
        "syntax error",
        "binder error",
        "catalog error",
        "does not exist",
        "no such table",
        "no such column",
    ]
    if any(token in lowered for token in compile_markers):
        return "compile_fail"

    return "runtime_fail"


def reset_question_flow_state(state: MutableMapping[str, Any]) -> None:
    """Reset New Question flow values without touching training data state."""
    state["question"] = ""
    state["generated_sql"] = ""
    state["generated_explanation"] = ""
    state["last_result_df"] = None
    state["last_result_elapsed"] = None
    state["suggestions"] = []
    state["generation_notes"] = []
    state["feedback_last_rating"] = None
    state["feedback_last_question_hash"] = "no_question"


def run_query_if_read_only(
    sql: str,
    validate_fn: Callable[[str], tuple[bool, str]],
    run_fn: Callable[[str], Any],
) -> tuple[bool, Any, str]:
    """
    Execute only when SQL passes read-only validation.
    Returns: (is_allowed, run_result_or_none, message)
    """
    is_ok, message = validate_fn(sql)
    if not is_ok:
        return False, None, message
    return True, run_fn(sql), "OK"


