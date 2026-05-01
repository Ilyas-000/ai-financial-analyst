"""SQLAlchemy 2.x models for the AI Financial Analyst (multi-tenant partner portal).

Tenancy posture
---------------
The deployment is multi-tenant: one instance hosted by the BaaS operator serves
many partner companies (tenants). Tables fall into two groups:

* **Tenancy-aware** — every row carries ``company_id``. The SQL guard injects
  ``WHERE company_id = <authenticated_user.company_id>`` automatically; the
  value is **never** taken from LLM input. Tables: ``employees``, ``cards``,
  ``transactions``, ``payouts``, ``payout_recipients``, ``limits``,
  ``tariffs``, ``tariff_rules``, ``report_snapshots``, ``audit_log``.
* **Shared** — operator-wide reference data, no ``company_id``. Tables:
  ``companies``, ``categories``, ``reason_codes``, ``currency_rates``,
  ``ingestion_state``.

Every ``company_id`` column is indexed to match the dominant access pattern
(filter-by-tenant + secondary attribute).

Column comments are in English so the schema retriever (in I-04) can ground
SQL generation on them directly.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Tenant-id column documentation — repeated identically on every tenancy-aware
# table so the schema retriever surfaces the contract uniformly.
TENANT_ID_COMMENT = (
    "tenant identifier; SQL guard injects from authenticated user, "
    "never from LLM input"
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# Shared (operator-wide) tables
# ---------------------------------------------------------------------------


class Company(Base):
    """Client-company (tenant). Shared table — referenced by every tenancy-aware row."""

    __tablename__ = "companies"
    __table_args__ = (
        CheckConstraint("tier in ('standard','premium')", name="ck_companies_tier"),
        {"comment": "Client-companies (tenants) onboarded by the BaaS operator."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="legal name")
    inn: Mapped[str] = mapped_column(
        String(12),
        nullable=False,
        unique=True,
        comment="Russian taxpayer identification number (10 or 12 digits)",
    )
    tier: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="commercial tier: standard | premium",
    )


class Category(Base):
    """Spending category (hierarchical via self-FK on ``parent``)."""

    __tablename__ = "categories"
    __table_args__ = {"comment": "Spending categories. Shared reference data."}

    code: Mapped[str] = mapped_column(String(50), primary_key=True, comment="machine code")
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="human-readable name")
    parent: Mapped[str | None] = mapped_column(
        String(50),
        ForeignKey("categories.code", ondelete="SET NULL"),
        nullable=True,
        comment="parent category code; NULL for top-level",
    )


class ReasonCode(Base):
    """Reason code attached to outgoing transactions / payouts."""

    __tablename__ = "reason_codes"
    __table_args__ = {"comment": "Reason codes (operational classifiers). Shared reference data."}

    code: Mapped[str] = mapped_column(String(50), primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(
        String(50),
        ForeignKey("categories.code", ondelete="SET NULL"),
        nullable=True,
        comment="default category this reason code maps to",
    )


class CurrencyRate(Base):
    """Daily FX rate snapshot (composite PK base+quote+date)."""

    __tablename__ = "currency_rates"
    __table_args__ = (
        UniqueConstraint("base", "quote", "date", name="uq_currency_rates_triple"),
        {"comment": "Daily FX rates. Shared reference data."},
    )

    base: Mapped[str] = mapped_column(
        String(3), primary_key=True, comment="base currency ISO 4217 code"
    )
    quote: Mapped[str] = mapped_column(
        String(3), primary_key=True, comment="quote currency ISO 4217 code"
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True, comment="rate snapshot date")
    rate: Mapped[Decimal] = mapped_column(
        Numeric(18, 8), nullable=False, comment="1 unit of base = rate units of quote"
    )
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="upstream provider id"
    )


class IngestionState(Base):
    """RAG ingestion bookkeeping — tracks which documents are indexed in Qdrant."""

    __tablename__ = "ingestion_state"
    __table_args__ = {"comment": "RAG ingestion bookkeeping. Shared (operator-wide)."}

    doc_id: Mapped[str] = mapped_column(
        String(200), primary_key=True, comment="stable document identifier (path or slug)"
    )
    hash: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="content hash for change detection"
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Tenancy-aware tables (every row carries company_id)
# ---------------------------------------------------------------------------


class Employee(Base):
    __tablename__ = "employees"
    __table_args__ = (
        Index("ix_employees_company_id", "company_id"),
        {"comment": "Employees of a client-company. Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    department: Mapped[str | None] = mapped_column(String(100), nullable=True)
    position: Mapped[str | None] = mapped_column(String(100), nullable=True)


class Card(Base):
    __tablename__ = "cards"
    __table_args__ = (
        Index("ix_cards_company_id", "company_id"),
        Index("ix_cards_employee_id", "employee_id"),
        CheckConstraint("status in ('active','blocked','expired')", name="ck_cards_status"),
        {"comment": "Corporate cards issued to employees. Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    employee_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("employees.id", ondelete="RESTRICT"),
        nullable=False,
        comment="cardholder",
    )
    last4: Mapped[str] = mapped_column(String(4), nullable=False, comment="last 4 digits of PAN")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="active | blocked | expired"
    )
    limit_monthly: Mapped[Decimal] = mapped_column(
        Numeric(15, 2), nullable=False, comment="monthly spending limit in card currency"
    )
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, comment="ISO 4217 code"
    )
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        Index("ix_transactions_company_id", "company_id"),
        Index("ix_transactions_card_id", "card_id"),
        Index("ix_transactions_company_created_at", "company_id", "created_at"),
        CheckConstraint(
            "status in ('approved','declined','reversed')", name="ck_transactions_status"
        ),
        {"comment": "Card transactions. Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    card_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("cards.id", ondelete="RESTRICT"), nullable=False
    )
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    amount: Mapped[Decimal] = mapped_column(
        Numeric(15, 2), nullable=False, comment="positive value in transaction currency"
    )
    fee_amount: Mapped[Decimal] = mapped_column(
        Numeric(15, 2),
        nullable=False,
        default=0,
        comment="operator fee charged for this transaction in the same currency",
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    mcc: Mapped[str | None] = mapped_column(
        String(4), nullable=True, comment="merchant category code (4 digits)"
    )
    category: Mapped[str | None] = mapped_column(
        String(50),
        ForeignKey("categories.code", ondelete="SET NULL"),
        nullable=True,
        comment="resolved category code",
    )
    merchant: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="approved | declined | reversed"
    )
    reason_code: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("reason_codes.code", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PayoutRecipient(Base):
    __tablename__ = "payout_recipients"
    __table_args__ = (
        Index("ix_payout_recipients_company_id", "company_id"),
        CheckConstraint("type in ('individual','company')", name="ck_recipients_type"),
        {"comment": "Counterparties that receive payouts from a tenant. Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    type: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="individual | company"
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    inn: Mapped[str | None] = mapped_column(
        String(12), nullable=True, comment="Russian taxpayer id; sensitive — restricted columns"
    )
    account: Mapped[str | None] = mapped_column(
        String(34), nullable=True, comment="bank account; sensitive — restricted columns"
    )


class Payout(Base):
    __tablename__ = "payouts"
    __table_args__ = (
        Index("ix_payouts_company_id", "company_id"),
        Index("ix_payouts_recipient_id", "recipient_id"),
        Index("ix_payouts_company_created_at", "company_id", "created_at"),
        CheckConstraint(
            "status in ('pending','approved','rejected','executed')", name="ck_payouts_status"
        ),
        {
            "comment": (
                "Outgoing payouts "
                "(vendor / contractor / employee reimbursements). Tenancy-aware."
            )
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    recipient_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("payout_recipients.id", ondelete="RESTRICT"), nullable=False
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    fee_amount: Mapped[Decimal] = mapped_column(
        Numeric(15, 2),
        nullable=False,
        default=0,
        comment="operator fee charged for this payout in the same currency",
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="pending | approved | rejected | executed"
    )
    reason_code: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("reason_codes.code", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Limit(Base):
    __tablename__ = "limits"
    __table_args__ = (
        Index("ix_limits_company_id", "company_id"),
        CheckConstraint(
            "subject_type in ('company','employee','card','category')",
            name="ck_limits_subject_type",
        ),
        CheckConstraint(
            "period in ('day','week','month','quarter','year')", name="ck_limits_period"
        ),
        {"comment": "Spending limits — per company / employee / card / category. Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    subject_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="company | employee | card | category",
    )
    subject_id: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="reference id matching subject_type (NULL when subject_type='company')",
    )
    period: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="day | week | month | quarter | year"
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)


class Tariff(Base):
    """Operator-side tariff plan a partner company is signed up to."""

    __tablename__ = "tariffs"
    __table_args__ = (
        Index("ix_tariffs_company_id", "company_id"),
        Index("ix_tariffs_company_status", "company_id", "status"),
        CheckConstraint(
            "status in ('active','inactive','draft')", name="ck_tariffs_status"
        ),
        {"comment": "Operator tariff plans assigned to partner companies. Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="human-readable plan name"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="active | inactive | draft"
    )
    effective_from: Mapped[date] = mapped_column(
        Date, nullable=False, comment="first day the tariff applies (inclusive)"
    )
    effective_to: Mapped[date | None] = mapped_column(
        Date, nullable=True, comment="last day the tariff applies (inclusive); NULL = open-ended"
    )
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, comment="default currency for fees in this tariff"
    )


class TariffRule(Base):
    """Single fee rule belonging to a tariff (per operation type)."""

    __tablename__ = "tariff_rules"
    __table_args__ = (
        Index("ix_tariff_rules_company_id", "company_id"),
        Index("ix_tariff_rules_tariff_id", "tariff_id"),
        CheckConstraint(
            "fee_type in ('fixed','percent','combined')", name="ck_tariff_rules_fee_type"
        ),
        {"comment": "Fee rules attached to a tariff (1 rule per operation_type). Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tariff_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("tariffs.id", ondelete="CASCADE"), nullable=False
    )
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    operation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="card_payment | payout_individual | payout_company | currency_conversion",
    )
    fee_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="fixed | percent | combined (fixed + percent)",
    )
    amount_fixed: Mapped[Decimal | None] = mapped_column(
        Numeric(15, 2),
        nullable=True,
        comment="fixed fee component (NULL for pure percent)",
    )
    amount_percent: Mapped[Decimal | None] = mapped_column(
        Numeric(7, 4),
        nullable=True,
        comment="percent fee component, e.g. 1.5000 = 1.5%",
    )
    min_fee: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    max_fee: Mapped[Decimal | None] = mapped_column(Numeric(15, 2), nullable=True)
    conditions_json: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="optional structured conditions (mcc whitelist, amount thresholds, etc.)",
    )


class ReportSnapshot(Base):
    """Snapshot of a periodic report produced for a partner company."""

    __tablename__ = "report_snapshots"
    __table_args__ = (
        Index("ix_report_snapshots_company_id", "company_id"),
        Index(
            "ix_report_snapshots_company_period",
            "company_id",
            "period_start",
            "period_end",
        ),
        CheckConstraint(
            "report_type in ('monthly','quarterly','annual','reconciliation')",
            name="ck_report_snapshots_type",
        ),
        {"comment": "Periodic report snapshots (closed periods). Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    report_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        comment="monthly | quarterly | annual | reconciliation",
    )
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    payload_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        comment="structured snapshot: totals by category, fees, anomalies, etc.",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AuditLog(Base):
    """Audit trail for every SQL request the agent issues + cross-tenant incidents."""

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_log_company_id", "company_id"),
        Index("ix_audit_log_created_at", "created_at"),
        CheckConstraint("severity in ('info','warning','high')", name="ck_audit_severity"),
        {"comment": "Audit trail for SQL requests + tenancy isolation incidents. Tenancy-aware."},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        comment=TENANT_ID_COMMENT,
    )
    user_role: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="role of the authenticated user that issued the action"
    )
    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="event kind: sql_executed | sql_rejected | tenancy_rewrite | tenancy_violation",
    )
    sql_text: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="SQL statement after guard rewrites (if any)"
    )
    rows_returned: Mapped[int | None] = mapped_column(Integer, nullable=True)
    severity: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="info",
        comment="info | warning | high (cross-tenant incidents = high)",
    )
    details: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="structured payload: violations, original SQL, errors"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
