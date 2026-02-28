from src.core.query_logic import build_template_sql


def test_template_total_revenue_with_year_filter() -> None:
    sql, note = build_template_sql("What is the total revenue in 2025?")
    assert "SUM(tv.umsatz_eur)" in sql
    assert "FROM ticket_verkaeufe tv" in sql
    assert "WHERE tv.jahr = 2025" in sql
    assert "total revenue" in note.lower()


def test_template_revenue_by_month() -> None:
    sql, note = build_template_sql("Show revenue per month for 2024")
    assert "GROUP BY tv.jahr, tv.monat" in sql
    assert "ORDER BY tv.jahr, tv.monat" in sql
    assert "WHERE tv.jahr = 2024" in sql
    assert "revenue by month" in note.lower()


def test_template_revenue_by_state() -> None:
    sql, note = build_template_sql("Show revenue by federal state for 2025")
    assert "JOIN postleitzahlen p" in sql
    assert "JOIN regionen_bundesland rb" in sql
    assert "rb.bundesland_name" in sql
    assert "revenue by state" in note.lower()


def test_template_revenue_by_ticket_type() -> None:
    sql, note = build_template_sql("Which ticket types generate the most revenue?")
    assert "JOIN ticket_produkte tp" in sql
    assert "tp.ticket_name" in sql
    assert "revenue by ticket type" in note.lower()


def test_template_revenue_by_tariff_state_month() -> None:
    sql, note = build_template_sql(
        "Show revenue by tariff association and state by month for 2025"
    )
    assert "JOIN tarifverbuende t" in sql
    assert "JOIN postleitzahlen p" in sql
    assert "JOIN regionen_bundesland rb" in sql
    assert "tv.monat" in sql
    assert "state, and month" in note.lower()


def test_template_no_match_for_unrelated_question() -> None:
    sql, note = build_template_sql("How many users like music?")
    assert sql == ""
    assert note == ""

