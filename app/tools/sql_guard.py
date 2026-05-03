"""SQL guard: parse, validate, rewrite tenant filters, enforce LIMIT.

Single entry point: ``guard_sql(sql, user_role, company_id) -> GuardResult``.
On rejection raises ``GuardRejection`` whose payload the executor mirrors into
``audit_log``. Successful guarding never silently changes semantics beyond:

  * injecting ``<table>.company_id = <user>`` for tenancy-aware tables that
    have no such filter — flagged as ``injected_company_id_filters``;
  * rewriting an explicit ``company_id = <other>`` to ``= <user>`` (treated
    as a cross-tenant attempt, ``audit_action='tenancy_rewrite'``,
    ``severity='high'``);
  * setting ``LIMIT`` to the configured default if absent, or capping to the
    configured max if too large.

Anything we cannot interpret safely (non-equality predicate on
``company_id``, qualified to an unknown alias, non-integer literal, etc.)
is rejected rather than ``corrected silently``.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final

import sqlglot
from sqlglot import exp

from app.config import get_settings

DIALECT: Final[str] = "postgres"

SHARED_TABLES: Final[frozenset[str]] = frozenset(
    {"companies", "categories", "reason_codes", "currency_rates"}
)

TENANT_TABLES: Final[frozenset[str]] = frozenset(
    {
        "employees",
        "cards",
        "transactions",
        "payouts",
        "payout_recipients",
        "limits",
        "tariffs",
        "tariff_rules",
        "report_snapshots",
        "audit_log",
    }
)

ROLE_TABLE_ALLOWLIST: Final[dict[str, frozenset[str]]] = {
    "cfo": SHARED_TABLES
    | frozenset(
        {
            "employees",
            "cards",
            "transactions",
            "payouts",
            "payout_recipients",
            "limits",
            "tariffs",
            "tariff_rules",
            "report_snapshots",
        }
    ),
    "finance_manager": SHARED_TABLES
    | frozenset(
        {
            "transactions",
            "cards",
            "payouts",
            "limits",
            "tariffs",
            "tariff_rules",
            "report_snapshots",
        }
    ),
    "accountant": SHARED_TABLES
    | frozenset(
        {
            "transactions",
            "payouts",
            "payout_recipients",
            "tariffs",
            "tariff_rules",
            "report_snapshots",
        }
    ),
    "auditor": SHARED_TABLES | TENANT_TABLES,
}

# For each restricted column name, the set of tables that own it. Selecting
# the column elsewhere (e.g. ``companies.inn``) is unrelated and allowed.
SENSITIVE_COLUMN_OWNERS: Final[dict[str, frozenset[str]]] = {
    "inn": frozenset({"payout_recipients"}),
    "account": frozenset({"payout_recipients"}),
    "last4": frozenset({"cards"}),
}
SENSITIVE_ALLOWED_ROLES: Final[frozenset[str]] = frozenset({"cfo", "auditor"})

FORBIDDEN_OPERATIONS: Final[tuple[type[exp.Expression], ...]] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Copy,
    exp.Grant,
    exp.Command,
)

ALLOWED_SCHEMAS: Final[frozenset[str]] = frozenset({"", "public"})


class GuardRejection(Exception):
    """Raised when a query violates a guard rule."""

    def __init__(
        self,
        *,
        message: str,
        reason: str,
        severity: str = "warning",
        **details: object,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.severity = severity
        self.details: dict[str, object] = {"reason": reason, **details}


@dataclass(frozen=True)
class GuardResult:
    """Outcome of a successful guard pass."""

    sql: str
    tables: frozenset[str]
    audit_action: str
    audit_severity: str
    rewrote_company_id: bool
    rewritten_from_values: tuple[int, ...]
    injected_company_id_filters: int
    enforced_limit: int
    audit_details: dict[str, object] = field(default_factory=dict)


def guard_sql(sql: str, *, user_role: str, company_id: int) -> GuardResult:
    """Validate, rewrite and finalize ``sql`` for ``user_role``/``company_id``.

    On any rule violation raises ``GuardRejection``. Returned ``GuardResult.sql``
    is what the executor must run as-is.
    """
    if user_role not in ROLE_TABLE_ALLOWLIST:
        raise GuardRejection(
            message=f"unknown user_role: {user_role!r}",
            reason="unknown_role",
            user_role=user_role,
        )

    settings = get_settings()

    statements = _parse(sql)
    root = _expect_single_select(statements)
    _reject_forbidden_operations(root)
    _reject_select_star_for_restricted_role(root, user_role)

    cte_names = _collect_cte_names(root)
    referenced_tables = _validate_tables(root, cte_names, ROLE_TABLE_ALLOWLIST[user_role])
    _validate_sensitive_columns(root, user_role, referenced_tables)

    rewritten_values: list[int] = []
    injected_total = 0
    for select_node in root.find_all(exp.Select):
        injected, rewritten = _enforce_tenancy_per_select(
            select_node, user_company_id=company_id, cte_names=cte_names
        )
        injected_total += injected
        rewritten_values.extend(rewritten)

    enforced_limit = _enforce_limit(
        root,
        default_limit=settings.sql_default_limit,
        max_limit=settings.sql_max_limit,
    )

    rewrote = bool(rewritten_values)
    audit_action = "tenancy_rewrite" if rewrote else "sql_executed"
    audit_severity = "high" if rewrote else "info"
    audit_details: dict[str, object] = {
        "tables": sorted(referenced_tables),
        "injected_company_id_filters": injected_total,
        "enforced_limit": enforced_limit,
    }
    if rewrote:
        audit_details["rewritten_from_values"] = list(rewritten_values)

    return GuardResult(
        sql=root.sql(dialect=DIALECT),
        tables=frozenset(referenced_tables),
        audit_action=audit_action,
        audit_severity=audit_severity,
        rewrote_company_id=rewrote,
        rewritten_from_values=tuple(rewritten_values),
        injected_company_id_filters=injected_total,
        enforced_limit=enforced_limit,
        audit_details=audit_details,
    )


def _parse(sql: str) -> list[exp.Expression]:
    try:
        statements = sqlglot.parse(sql, dialect=DIALECT)
    except sqlglot.errors.ParseError as exc:
        raise GuardRejection(
            message=f"SQL parse error: {exc}",
            reason="parse_error",
            error=str(exc),
        ) from exc
    return [s for s in statements if s is not None]


def _expect_single_select(statements: list[exp.Expression]) -> exp.Select:
    if not statements:
        raise GuardRejection(message="empty SQL", reason="empty")
    if len(statements) > 1:
        raise GuardRejection(
            message="multiple statements are not allowed",
            reason="multi_statement",
            count=len(statements),
        )
    root = statements[0]
    if not isinstance(root, exp.Select):
        raise GuardRejection(
            message=f"only SELECT is allowed; got {type(root).__name__}",
            reason="non_select",
            kind=type(root).__name__,
        )
    return root


def _reject_forbidden_operations(root: exp.Expression) -> None:
    for forbidden_type in FORBIDDEN_OPERATIONS:
        node = root.find(forbidden_type)
        if node is not None:
            raise GuardRejection(
                message=f"forbidden operation: {forbidden_type.__name__}",
                reason="forbidden_op",
                operation=forbidden_type.__name__,
            )


def _reject_select_star_for_restricted_role(root: exp.Select, user_role: str) -> None:
    if user_role in SENSITIVE_ALLOWED_ROLES:
        return
    for select_node in root.find_all(exp.Select):
        for proj in select_node.expressions or []:
            if isinstance(proj, exp.Star):
                raise GuardRejection(
                    message="SELECT * is not allowed for this role; enumerate columns",
                    reason="select_star_restricted",
                    user_role=user_role,
                )
            if isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star):
                raise GuardRejection(
                    message="SELECT <table>.* is not allowed for this role; enumerate columns",
                    reason="select_qualified_star_restricted",
                    user_role=user_role,
                )


def _collect_cte_names(root: exp.Expression) -> frozenset[str]:
    names = set()
    for cte in root.find_all(exp.CTE):
        alias = cte.alias
        if alias:
            names.add(alias.lower())
    return frozenset(names)


def _validate_tables(
    root: exp.Select, cte_names: frozenset[str], allowlist: frozenset[str]
) -> set[str]:
    referenced: set[str] = set()
    for table in root.find_all(exp.Table):
        name = (table.name or "").lower()
        if not name:
            continue
        if name in cte_names:
            continue
        if name.startswith("pg_"):
            raise GuardRejection(
                message=f"system table not allowed: {table.sql()}",
                reason="system_table",
                table=name,
            )
        db = (table.db or "").lower()
        catalog = (table.catalog or "").lower()
        if catalog or db not in ALLOWED_SCHEMAS:
            raise GuardRejection(
                message=f"schema-qualified tables are not allowed: {table.sql()}",
                reason="schema_prefix",
                table=table.sql(),
            )
        if name not in allowlist:
            raise GuardRejection(
                message=f"table {name!r} is not allowed for this role",
                reason="table_not_allowed",
                table=name,
            )
        referenced.add(name)
    return referenced


def _validate_sensitive_columns(
    root: exp.Expression, user_role: str, referenced_tables: set[str]
) -> None:
    if user_role in SENSITIVE_ALLOWED_ROLES:
        return
    alias_to_table = _build_alias_to_table_map(root)
    for col in root.find_all(exp.Column):
        col_name = (col.name or "").lower()
        if col_name not in SENSITIVE_COLUMN_OWNERS:
            continue
        owners = SENSITIVE_COLUMN_OWNERS[col_name]
        col_table_ref = (col.table or "").lower()
        if col_table_ref:
            resolved = alias_to_table.get(col_table_ref, col_table_ref)
            if resolved in owners:
                raise GuardRejection(
                    message=f"column {resolved}.{col_name} is restricted for this role",
                    reason="sensitive_column",
                    user_role=user_role,
                    table=resolved,
                    column=col_name,
                )
        else:
            shared_owners = owners & referenced_tables
            if shared_owners:
                owner = next(iter(shared_owners))
                raise GuardRejection(
                    message=(f"column {col_name!r} is restricted for this role (owned by {owner})"),
                    reason="sensitive_column",
                    user_role=user_role,
                    table=owner,
                    column=col_name,
                )


def _build_alias_to_table_map(root: exp.Expression) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for table in root.find_all(exp.Table):
        name = (table.name or "").lower()
        if not name:
            continue
        alias = (table.alias or table.name or "").lower()
        mapping[alias] = name
    return mapping


def _enforce_tenancy_per_select(
    select_node: exp.Select,
    *,
    user_company_id: int,
    cte_names: frozenset[str],
) -> tuple[int, list[int]]:
    """Mutate ``select_node`` in place. Returns (injected_count, rewritten_old_values)."""
    tenant_refs = list(_direct_tenant_tables(select_node, cte_names))
    if not tenant_refs:
        return 0, []

    where_node = select_node.args.get("where")
    covered_aliases: set[str] = set()
    rewritten: list[int] = []

    if where_node is not None:
        for col in _direct_columns_under(where_node, select_node):
            if (col.name or "").lower() != "company_id":
                continue
            target_alias = _resolve_company_id_filter(col, tenant_refs, user_company_id)
            value_node = _expect_int_eq_with_column(col)
            cur_val = int(value_node.name)
            if cur_val != user_company_id:
                value_node.set("this", str(user_company_id))
                rewritten.append(cur_val)
            covered_aliases.add(target_alias)

    injected = 0
    for alias, _table_name in tenant_refs:
        if alias in covered_aliases:
            continue
        predicate = exp.EQ(
            this=exp.column("company_id", table=alias),
            expression=exp.Literal.number(user_company_id),
        )
        select_node.where(predicate, append=True, copy=False)
        injected += 1
        covered_aliases.add(alias)

    return injected, rewritten


def _direct_tenant_tables(
    select_node: exp.Select, cte_names: frozenset[str]
) -> Iterator[tuple[str, str]]:
    for table in select_node.find_all(exp.Table):
        if _node_nested_in_inner_select(table, select_node):
            continue
        name = (table.name or "").lower()
        if name in cte_names or name not in TENANT_TABLES:
            continue
        alias = (table.alias or table.name or "").lower()
        yield alias, name


def _direct_columns_under(
    container: exp.Expression, select_node: exp.Select
) -> Iterator[exp.Column]:
    for col in container.find_all(exp.Column):
        if not _node_nested_in_inner_select(col, select_node):
            yield col


def _node_nested_in_inner_select(node: exp.Expression, root_select: exp.Select) -> bool:
    ancestor = node.parent
    while ancestor is not None:
        if ancestor is root_select:
            return False
        if isinstance(ancestor, exp.Select):
            return True
        ancestor = ancestor.parent
    return False


def _expect_int_eq_with_column(col: exp.Column) -> exp.Literal:
    parent = col.parent
    if not isinstance(parent, exp.EQ):
        raise GuardRejection(
            message=(
                "company_id can only be filtered with `=` and an integer literal; "
                f"found {type(parent).__name__ if parent is not None else 'None'}"
            ),
            reason="company_id_complex_predicate",
            kind=type(parent).__name__ if parent is not None else None,
        )
    other = parent.expression if parent.this is col else parent.this
    if not isinstance(other, exp.Literal) or not other.is_int:
        raise GuardRejection(
            message="company_id must be compared to an integer literal",
            reason="company_id_non_literal",
        )
    return other


def _resolve_company_id_filter(
    col: exp.Column, tenant_refs: list[tuple[str, str]], user_company_id: int
) -> str:
    col_table_ref = (col.table or "").lower()
    if col_table_ref:
        for alias, _name in tenant_refs:
            if alias == col_table_ref:
                return alias
        raise GuardRejection(
            message=(
                f"company_id filter references unknown alias {col_table_ref!r}; "
                "guard cannot resolve which tenant table it constrains"
            ),
            reason="company_id_unknown_alias",
            alias=col_table_ref,
        )
    if len(tenant_refs) == 1:
        return tenant_refs[0][0]
    raise GuardRejection(
        message=(
            "unqualified company_id filter is ambiguous when multiple tenant "
            "tables are joined; qualify with an alias"
        ),
        reason="company_id_ambiguous",
        tables=[t for _, t in tenant_refs],
    )


def _enforce_limit(root: exp.Select, *, default_limit: int, max_limit: int) -> int:
    limit_node = root.args.get("limit")
    if limit_node is None:
        root.limit(default_limit, copy=False)
        return default_limit
    value_node = limit_node.expression
    if isinstance(value_node, exp.Literal) and value_node.is_int:
        existing = int(value_node.name)
        if existing > max_limit:
            value_node.set("this", str(max_limit))
            return max_limit
        return existing
    root.limit(default_limit, copy=False)
    return default_limit
