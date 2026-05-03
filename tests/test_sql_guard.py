"""Unit tests for app.tools.sql_guard."""

import pytest

from app.tools.sql_guard import GuardRejection, guard_sql


def test_select_injects_company_id_when_missing():
    result = guard_sql(
        "SELECT amount FROM transactions WHERE created_at > '2025-01-01'",
        user_role="finance_manager",
        company_id=1,
    )
    assert result.injected_company_id_filters == 1
    assert result.rewrote_company_id is False
    assert result.audit_action == "sql_executed"
    assert result.audit_severity == "info"
    assert "transactions.company_id = 1" in result.sql.lower().replace('"', "")


def test_select_keeps_correct_company_id():
    result = guard_sql(
        "SELECT amount FROM transactions WHERE company_id = 1",
        user_role="finance_manager",
        company_id=1,
    )
    assert result.injected_company_id_filters == 0
    assert result.rewrote_company_id is False


def test_select_rewrites_foreign_company_id():
    result = guard_sql(
        "SELECT amount FROM transactions WHERE company_id = 99",
        user_role="finance_manager",
        company_id=1,
    )
    assert result.rewrote_company_id is True
    assert result.rewritten_from_values == (99,)
    assert result.audit_action == "tenancy_rewrite"
    assert result.audit_severity == "high"
    assert "company_id = 1" in result.sql
    assert "99" not in result.sql


def test_join_injects_company_id_for_each_tenant_table():
    result = guard_sql(
        "SELECT t.amount, c.limit_monthly FROM transactions t "
        "JOIN cards c ON c.id = t.card_id WHERE t.company_id = 1",
        user_role="finance_manager",
        company_id=1,
    )
    assert result.injected_company_id_filters == 1
    sql = result.sql
    assert "t.company_id = 1" in sql
    assert "c.company_id = 1" in sql


def test_join_with_only_unqualified_company_id_is_rejected_when_ambiguous():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT t.amount FROM transactions t JOIN cards c ON c.id = t.card_id "
            "WHERE company_id = 1",
            user_role="finance_manager",
            company_id=1,
        )
    assert exc_info.value.reason == "company_id_ambiguous"


def test_subquery_in_where_gets_its_own_company_id():
    result = guard_sql(
        "SELECT id FROM transactions WHERE id IN "
        "(SELECT card_id FROM cards WHERE status = 'active')",
        user_role="finance_manager",
        company_id=2,
    )
    assert result.injected_company_id_filters == 2
    sql = result.sql
    assert "transactions.company_id = 2" in sql
    assert "cards.company_id = 2" in sql


def test_cte_reference_does_not_count_as_table_again():
    result = guard_sql(
        "WITH active_cards AS ("
        "SELECT id FROM cards WHERE status = 'active') "
        "SELECT id FROM active_cards",
        user_role="finance_manager",
        company_id=1,
    )
    assert "cards.company_id = 1" in result.sql
    assert result.injected_company_id_filters == 1


def test_default_limit_is_appended_when_missing():
    result = guard_sql(
        "SELECT amount FROM transactions WHERE company_id = 1",
        user_role="finance_manager",
        company_id=1,
    )
    assert result.enforced_limit == 1000
    assert "LIMIT 1000" in result.sql


def test_oversized_limit_is_capped():
    result = guard_sql(
        "SELECT amount FROM transactions WHERE company_id = 1 LIMIT 100000",
        user_role="finance_manager",
        company_id=1,
    )
    assert result.enforced_limit == 10000
    assert "LIMIT 10000" in result.sql


def test_existing_safe_limit_is_preserved():
    result = guard_sql(
        "SELECT amount FROM transactions WHERE company_id = 1 LIMIT 25",
        user_role="finance_manager",
        company_id=1,
    )
    assert result.enforced_limit == 25
    assert "LIMIT 25" in result.sql


@pytest.mark.parametrize(
    "sql, reason",
    [
        ("INSERT INTO transactions (amount) VALUES (1)", "non_select"),
        ("UPDATE transactions SET amount = 1", "non_select"),
        ("DELETE FROM transactions", "non_select"),
        ("DROP TABLE transactions", "non_select"),
        ("TRUNCATE TABLE transactions", "non_select"),
        ("CREATE TABLE x (id int)", "non_select"),
        ("COPY transactions TO STDOUT", "non_select"),
        ("GRANT SELECT ON transactions TO afa", "non_select"),
        ("SELECT 1; SELECT 2", "multi_statement"),
        ("not a sql at all !!!", "parse_error"),
        ("", "empty"),
    ],
)
def test_non_select_or_multi_statement_rejected(sql: str, reason: str):
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(sql, user_role="cfo", company_id=1)
    assert exc_info.value.reason == reason


def test_information_schema_rejected():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT * FROM information_schema.tables",
            user_role="cfo",
            company_id=1,
        )
    assert exc_info.value.reason == "schema_prefix"


def test_pg_catalog_rejected():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql("SELECT relname FROM pg_class", user_role="cfo", company_id=1)
    assert exc_info.value.reason == "system_table"


def test_unknown_role_rejected():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT amount FROM transactions WHERE company_id = 1",
            user_role="hacker",
            company_id=1,
        )
    assert exc_info.value.reason == "unknown_role"


def test_finance_manager_cannot_query_employees():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT name FROM employees WHERE company_id = 1",
            user_role="finance_manager",
            company_id=1,
        )
    assert exc_info.value.reason == "table_not_allowed"
    assert exc_info.value.details["table"] == "employees"


def test_accountant_cannot_query_audit_log():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT id FROM audit_log WHERE company_id = 1",
            user_role="accountant",
            company_id=1,
        )
    assert exc_info.value.reason == "table_not_allowed"


def test_auditor_can_query_audit_log():
    result = guard_sql(
        "SELECT id FROM audit_log WHERE company_id = 1",
        user_role="auditor",
        company_id=1,
    )
    assert "audit_log" in result.tables


def test_finance_manager_cannot_select_card_last4():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT last4 FROM cards WHERE company_id = 1",
            user_role="finance_manager",
            company_id=1,
        )
    assert exc_info.value.reason == "sensitive_column"
    assert exc_info.value.details["column"] == "last4"


def test_accountant_cannot_select_recipient_inn():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT inn FROM payout_recipients WHERE company_id = 1",
            user_role="accountant",
            company_id=1,
        )
    assert exc_info.value.reason == "sensitive_column"


def test_cfo_can_select_card_last4():
    result = guard_sql(
        "SELECT last4 FROM cards WHERE company_id = 1",
        user_role="cfo",
        company_id=1,
    )
    assert "cards" in result.tables


def test_select_star_rejected_for_finance_manager():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT * FROM transactions WHERE company_id = 1",
            user_role="finance_manager",
            company_id=1,
        )
    assert exc_info.value.reason == "select_star_restricted"


def test_qualified_star_rejected_for_accountant():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT t.* FROM transactions t WHERE t.company_id = 1",
            user_role="accountant",
            company_id=1,
        )
    assert exc_info.value.reason == "select_qualified_star_restricted"


def test_select_star_allowed_for_cfo():
    result = guard_sql(
        "SELECT * FROM transactions WHERE company_id = 1",
        user_role="cfo",
        company_id=1,
    )
    assert "transactions" in result.tables


def test_company_id_with_in_clause_rejected():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT amount FROM transactions WHERE company_id IN (1, 2)",
            user_role="finance_manager",
            company_id=1,
        )
    assert exc_info.value.reason == "company_id_complex_predicate"


def test_company_id_with_inequality_rejected():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT amount FROM transactions WHERE company_id <> 99",
            user_role="finance_manager",
            company_id=1,
        )
    assert exc_info.value.reason == "company_id_complex_predicate"


def test_company_id_unknown_alias_rejected():
    with pytest.raises(GuardRejection) as exc_info:
        guard_sql(
            "SELECT amount FROM transactions WHERE x.company_id = 1",
            user_role="finance_manager",
            company_id=1,
        )
    assert exc_info.value.reason == "company_id_unknown_alias"


def test_shared_table_does_not_require_company_id_filter():
    result = guard_sql(
        "SELECT base, quote, rate FROM currency_rates WHERE base = 'USD'",
        user_role="accountant",
        company_id=2,
    )
    assert result.injected_company_id_filters == 0
    assert result.rewrote_company_id is False
    assert "currency_rates" in result.tables


def test_shared_join_with_tenant_table_only_injects_for_tenant():
    result = guard_sql(
        "SELECT t.amount, c.name FROM transactions t "
        "JOIN categories c ON c.code = t.category WHERE t.company_id = 3",
        user_role="finance_manager",
        company_id=3,
    )
    assert result.injected_company_id_filters == 0
    assert {"transactions", "categories"} <= result.tables


def test_normalized_sql_can_be_re_parsed():
    import sqlglot

    result = guard_sql(
        "SELECT amount FROM transactions WHERE created_at > '2025-01-01'",
        user_role="cfo",
        company_id=1,
    )
    sqlglot.parse_one(result.sql, dialect="postgres")


def test_join_with_qualified_foreign_company_id_is_rewritten():
    result = guard_sql(
        "SELECT t.amount FROM transactions t JOIN cards c ON c.id = t.card_id "
        "WHERE t.company_id = 99 AND c.company_id = 99",
        user_role="finance_manager",
        company_id=1,
    )
    assert result.rewrote_company_id is True
    assert sorted(result.rewritten_from_values) == [99, 99]
    assert result.injected_company_id_filters == 0
    assert "99" not in result.sql
