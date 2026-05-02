"""Deterministic demo seed data."""

import asyncio
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from faker import Faker
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Card,
    Category,
    Company,
    CurrencyRate,
    Employee,
    Limit,
    Payout,
    PayoutRecipient,
    ReasonCode,
    ReportSnapshot,
    Tariff,
    TariffRule,
    Transaction,
)
from app.db.session import get_engine, get_sessionmaker

SEED = 42

# Explicit truncate order for readability.
TRUNCATE_ORDER = (
    "audit_log",
    "report_snapshots",
    "tariff_rules",
    "tariffs",
    "transactions",
    "payouts",
    "payout_recipients",
    "cards",
    "limits",
    "employees",
    "currency_rates",
    "ingestion_state",
    "reason_codes",
    "categories",
    "companies",
)


@dataclass(frozen=True)
class TenantSpec:
    name: str
    inn: str
    tier: str
    profile: str
    employees: int
    cards: int
    transactions: int
    payouts: int


TENANTS: tuple[TenantSpec, ...] = (
    TenantSpec("ACME LLC", "7700000001", "standard", "general", 70, 140, 2000, 80),
    TenantSpec("Ostrovok-mock", "7800000002", "premium", "travel_heavy", 50, 100, 1800, 70),
    TenantSpec("CheckScan-mock", "5000000003", "standard", "saas_heavy", 30, 60, 1200, 50),
)

CATEGORIES: tuple[tuple[str, str, str | None], ...] = (
    ("travel", "Travel", None),
    ("travel_lodging", "Lodging", "travel"),
    ("travel_transport", "Transport", "travel"),
    ("travel_meals", "Business meals on trips", "travel"),
    ("software", "Software & SaaS", None),
    ("software_saas", "SaaS subscriptions", "software"),
    ("software_licenses", "Perpetual licenses", "software"),
    ("services", "Professional services", None),
    ("services_legal", "Legal services", "services"),
    ("services_consulting", "Consulting services", "services"),
    ("office", "Office expenses", None),
    ("office_supplies", "Office supplies", "office"),
    ("office_rent", "Office rent", "office"),
    ("misc", "Other", None),
)


REASON_CODES: tuple[tuple[str, str, str | None], ...] = (
    ("BUSINESS_TRIP", "Business trip expenses", "travel"),
    ("CLIENT_MEETING", "Client meeting expenses", "travel_meals"),
    ("SOFTWARE_PURCHASE", "Software / SaaS purchase", "software"),
    ("LEGAL_FEE", "Legal services payment", "services_legal"),
    ("CONSULTING_FEE", "Consulting services payment", "services_consulting"),
    ("OFFICE_SUPPLIES", "Office supplies purchase", "office_supplies"),
    ("OFFICE_RENT", "Office rent payment", "office_rent"),
    ("EMPLOYEE_REIMB", "Employee reimbursement", None),
    ("VENDOR_PAYMENT", "Vendor invoice payment", None),
    ("CONTRACTOR_PAY", "Contractor payment", None),
    ("MARKETING", "Marketing campaign", None),
    ("EQUIPMENT", "Equipment purchase", None),
)

PROFILE_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "general": {
        "travel_lodging": 0.10,
        "travel_transport": 0.10,
        "travel_meals": 0.05,
        "software_saas": 0.15,
        "services_legal": 0.05,
        "services_consulting": 0.10,
        "office_supplies": 0.15,
        "office_rent": 0.10,
        "misc": 0.20,
    },
    "travel_heavy": {
        "travel_lodging": 0.30,
        "travel_transport": 0.25,
        "travel_meals": 0.15,
        "software_saas": 0.05,
        "services_consulting": 0.05,
        "office_supplies": 0.05,
        "office_rent": 0.05,
        "misc": 0.10,
    },
    "saas_heavy": {
        "software_saas": 0.30,
        "software_licenses": 0.10,
        "services_consulting": 0.20,
        "services_legal": 0.10,
        "travel_transport": 0.05,
        "travel_lodging": 0.05,
        "office_supplies": 0.05,
        "misc": 0.15,
    },
}

CATEGORY_AMOUNT_RANGE: dict[str, tuple[int, int]] = {
    "travel_lodging": (3_000, 25_000),
    "travel_transport": (500, 15_000),
    "travel_meals": (500, 5_000),
    "software_saas": (1_000, 30_000),
    "software_licenses": (5_000, 80_000),
    "services_legal": (10_000, 150_000),
    "services_consulting": (10_000, 200_000),
    "office_supplies": (500, 10_000),
    "office_rent": (50_000, 250_000),
    "misc": (300, 8_000),
}


CURRENCIES = ("RUB", "USD", "EUR")
CURRENCY_WEIGHTS = (0.85, 0.10, 0.05)
TX_STATUS_WEIGHTS = (("approved", 0.92), ("declined", 0.06), ("reversed", 0.02))
PAYOUT_STATUS_WEIGHTS = (
    ("executed", 0.65),
    ("approved", 0.15),
    ("pending", 0.15),
    ("rejected", 0.05),
)


def _weighted_choice(rng: random.Random, items: list[tuple[str, float]]) -> str:
    choice_names = [choice_name for choice_name, _ in items]
    choice_weights = [choice_weight for _, choice_weight in items]
    return rng.choices(choice_names, weights=choice_weights, k=1)[0]


def _weighted_category(rng: random.Random, profile: str) -> str:
    weights = PROFILE_CATEGORY_WEIGHTS[profile]
    return rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def _amount_for_category(rng: random.Random, category: str) -> Decimal:
    lower_bound, upper_bound = CATEGORY_AMOUNT_RANGE.get(category, (500, 10_000))
    cents = rng.randint(lower_bound * 100, upper_bound * 100)
    return (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))


def _pick_currency(rng: random.Random) -> str:
    return rng.choices(list(CURRENCIES), weights=list(CURRENCY_WEIGHTS), k=1)[0]


def _reason_for_category(category: str) -> str | None:
    mapping = {
        "travel_lodging": "BUSINESS_TRIP",
        "travel_transport": "BUSINESS_TRIP",
        "travel_meals": "CLIENT_MEETING",
        "software_saas": "SOFTWARE_PURCHASE",
        "software_licenses": "SOFTWARE_PURCHASE",
        "services_legal": "LEGAL_FEE",
        "services_consulting": "CONSULTING_FEE",
        "office_supplies": "OFFICE_SUPPLIES",
        "office_rent": "OFFICE_RENT",
    }
    return mapping.get(category)


# Per-tier fee bands in percent.
TX_FEE_PERCENT_RANGE: dict[str, tuple[float, float]] = {
    "standard": (1.8, 2.5),
    "premium": (1.0, 1.6),
}
PAYOUT_FEE_PERCENT_RANGE: dict[str, tuple[float, float]] = {
    "standard": (0.8, 1.5),
    "premium": (0.4, 0.9),
}


def _fee_for(amount: Decimal, status: str, tier: str, kind: str) -> Decimal:
    """Return a deterministic fee in the same currency."""
    if status in {"declined", "reversed", "rejected", "pending"}:
        return Decimal("0.00")
    band = (TX_FEE_PERCENT_RANGE if kind == "tx" else PAYOUT_FEE_PERCENT_RANGE)[tier]
    cents = int((amount * 100).to_integral_value()) % 1000
    span = band[1] - band[0]
    pct = Decimal(str(band[0] + (cents / 1000.0) * span))
    return (amount * pct / Decimal(100)).quantize(Decimal("0.01"))


async def _seed_reference(session: AsyncSession) -> None:
    """Seed shared reference data."""
    for category_code, category_name, parent_code in CATEGORIES:
        if parent_code is None:
            session.add(Category(code=category_code, name=category_name, parent=None))
    await session.flush()
    for category_code, category_name, parent_code in CATEGORIES:
        if parent_code is not None:
            session.add(Category(code=category_code, name=category_name, parent=parent_code))
    await session.flush()

    for reason_code, description, default_category in REASON_CODES:
        session.add(
            ReasonCode(
                code=reason_code,
                description=description,
                category=default_category,
            )
        )
    await session.flush()

    today = date.today()
    rng = random.Random(SEED)
    for offset in range(30):
        rate_date = today - timedelta(days=offset)
        usd = Decimal(rng.randint(8800, 9800)) / Decimal(100)  # ~88..98 RUB / USD
        eur = Decimal(rng.randint(9500, 10500)) / Decimal(100)  # ~95..105 RUB / EUR
        session.add(CurrencyRate(base="USD", quote="RUB", date=rate_date, rate=usd, source="seed"))
        session.add(CurrencyRate(base="EUR", quote="RUB", date=rate_date, rate=eur, source="seed"))
        session.add(
            CurrencyRate(
                base="RUB",
                quote="RUB",
                date=rate_date,
                rate=Decimal("1.00"),
                source="seed",
            )
        )
    await session.flush()


async def _seed_tenant(
    session: AsyncSession, spec: TenantSpec, fake: Faker, rng: random.Random
) -> None:
    company = Company(name=spec.name, inn=spec.inn, tier=spec.tier)
    session.add(company)
    await session.flush()

    employees: list[Employee] = []
    departments = ("Finance", "Engineering", "Sales", "Operations", "HR", "Legal")
    positions = ("Manager", "Specialist", "Analyst", "Director", "Lead", "Coordinator")
    for _ in range(spec.employees):
        emp = Employee(
            company_id=company.id,
            name=fake.name(),
            department=rng.choice(departments),
            position=rng.choice(positions),
        )
        session.add(emp)
        employees.append(emp)
    await session.flush()

    cards: list[Card] = []
    base_issued = datetime.now(UTC) - timedelta(days=365)
    monthly_limits = (50_000, 100_000) if spec.tier == "standard" else (100_000, 300_000)
    for card_index in range(spec.cards):
        owner = employees[card_index % len(employees)]
        card = Card(
            company_id=company.id,
            employee_id=owner.id,
            last4=f"{rng.randint(0, 9999):04d}",
            status=rng.choices(["active", "blocked", "expired"], weights=[0.92, 0.05, 0.03], k=1)[
                0
            ],
            limit_monthly=Decimal(rng.randint(*monthly_limits)),
            currency="RUB",
            issued_at=base_issued + timedelta(days=rng.randint(0, 300)),
        )
        session.add(card)
        cards.append(card)
    await session.flush()

    limit_count = rng.randint(3, 5)
    period_choices = ("month", "quarter", "year")
    base_limit_amount = Decimal(500_000) if spec.tier == "standard" else Decimal(2_000_000)
    for _ in range(limit_count):
        subject_type = rng.choice(["company", "category", "employee"])
        if subject_type == "company":
            subject_id = None
        elif subject_type == "category":
            subject_id = rng.choice(list(CATEGORY_AMOUNT_RANGE.keys()))
        else:
            subject_id = str(rng.choice(employees).id)
        session.add(
            Limit(
                company_id=company.id,
                subject_type=subject_type,
                subject_id=subject_id,
                period=rng.choice(period_choices),
                amount=base_limit_amount * Decimal(rng.randint(1, 5)),
                currency="RUB",
            )
        )
    await session.flush()

    recipient_count = max(5, spec.payouts // 3)
    recipients: list[PayoutRecipient] = []
    for _ in range(recipient_count):
        recipient_type = rng.choices(["company", "individual"], weights=[0.7, 0.3], k=1)[0]
        recipients.append(
            PayoutRecipient(
                company_id=company.id,
                type=recipient_type,
                name=fake.company() if recipient_type == "company" else fake.name(),
                inn=fake.numerify("##########") if recipient_type == "company" else None,
                account=fake.numerify("407028108########"),
            )
        )
    session.add_all(recipients)
    await session.flush()

    now = datetime.now(UTC)
    active_cards = [card for card in cards if card.status == "active"] or cards
    for _ in range(spec.transactions):
        category = _weighted_category(rng, spec.profile)
        amount = _amount_for_category(rng, category)
        currency = _pick_currency(rng)
        if currency != "RUB":
            amount = (amount / Decimal(90)).quantize(Decimal("0.01"))
        card = rng.choice(active_cards)
        status = _weighted_choice(rng, list(TX_STATUS_WEIGHTS))
        session.add(
            Transaction(
                card_id=card.id,
                company_id=company.id,
                amount=amount,
                fee_amount=_fee_for(amount, status, spec.tier, "tx"),
                currency=currency,
                mcc=f"{rng.randint(1000, 9999):04d}",
                category=category,
                merchant=fake.company(),
                status=status,
                reason_code=_reason_for_category(category),
                created_at=now - timedelta(minutes=rng.randint(0, 180 * 24 * 60)),
            )
        )
    await session.flush()

    for _ in range(spec.payouts):
        recipient = rng.choice(recipients)
        category = _weighted_category(rng, spec.profile)
        amount = _amount_for_category(rng, category) * Decimal(rng.randint(2, 8))
        status = _weighted_choice(rng, list(PAYOUT_STATUS_WEIGHTS))
        session.add(
            Payout(
                company_id=company.id,
                recipient_id=recipient.id,
                amount=amount,
                fee_amount=_fee_for(amount, status, spec.tier, "payout"),
                currency="RUB",
                status=status,
                reason_code=_reason_for_category(category) or "VENDOR_PAYMENT",
                created_at=now - timedelta(minutes=rng.randint(0, 180 * 24 * 60)),
            )
        )
    await session.flush()

    await _seed_tariffs(session, spec, company.id, rng)
    await _seed_report_snapshots(session, spec, company.id, now)


TARIFF_RULE_TEMPLATES: dict[str, list[dict]] = {
    "standard": [
        {
            "operation_type": "card_payment",
            "fee_type": "percent",
            "amount_percent": Decimal("2.00"),
            "min_fee": Decimal("10.00"),
        },
        {
            "operation_type": "payout_individual",
            "fee_type": "combined",
            "amount_fixed": Decimal("25.00"),
            "amount_percent": Decimal("1.20"),
            "min_fee": Decimal("30.00"),
            "max_fee": Decimal("3000.00"),
        },
        {
            "operation_type": "payout_company",
            "fee_type": "fixed",
            "amount_fixed": Decimal("50.00"),
        },
        {
            "operation_type": "currency_conversion",
            "fee_type": "percent",
            "amount_percent": Decimal("1.50"),
        },
    ],
    "premium": [
        {
            "operation_type": "card_payment",
            "fee_type": "percent",
            "amount_percent": Decimal("1.30"),
            "min_fee": Decimal("5.00"),
        },
        {
            "operation_type": "payout_individual",
            "fee_type": "combined",
            "amount_fixed": Decimal("15.00"),
            "amount_percent": Decimal("0.70"),
            "min_fee": Decimal("20.00"),
            "max_fee": Decimal("2000.00"),
        },
        {
            "operation_type": "payout_company",
            "fee_type": "fixed",
            "amount_fixed": Decimal("30.00"),
        },
        {
            "operation_type": "currency_conversion",
            "fee_type": "percent",
            "amount_percent": Decimal("0.80"),
        },
    ],
}


async def _seed_tariffs(
    session: AsyncSession, spec: TenantSpec, company_id: int, rng: random.Random
) -> None:
    today = date.today()
    historical = Tariff(
        company_id=company_id,
        name=f"{spec.name} {today.year - 1} plan",
        status="inactive",
        effective_from=today.replace(year=today.year - 1, month=1, day=1),
        effective_to=today.replace(year=today.year - 1, month=12, day=31),
        currency="RUB",
    )
    current = Tariff(
        company_id=company_id,
        name=f"{spec.name} {today.year} plan",
        status="active",
        effective_from=today.replace(month=1, day=1),
        effective_to=None,
        currency="RUB",
    )
    session.add_all([historical, current])
    await session.flush()

    template = TARIFF_RULE_TEMPLATES[spec.tier]
    for tariff in (historical, current):
        markup = Decimal("1.10") if tariff is historical else Decimal("1.00")
        for rule_template in template:
            session.add(
                TariffRule(
                    tariff_id=tariff.id,
                    company_id=company_id,
                    operation_type=rule_template["operation_type"],
                    fee_type=rule_template["fee_type"],
                    amount_fixed=(
                        (rule_template.get("amount_fixed") or Decimal(0)) * markup
                    ).quantize(Decimal("0.01"))
                    if rule_template.get("amount_fixed") is not None
                    else None,
                    amount_percent=(
                        (rule_template.get("amount_percent") or Decimal(0)) * markup
                    ).quantize(Decimal("0.0001"))
                    if rule_template.get("amount_percent") is not None
                    else None,
                    min_fee=rule_template.get("min_fee"),
                    max_fee=rule_template.get("max_fee"),
                    conditions_json=(
                        {"mcc_whitelist_min": 4000, "mcc_whitelist_max": 7999}
                        if rule_template["operation_type"] == "card_payment" and rng.random() < 0.5
                        else None
                    ),
                )
            )
    await session.flush()


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    start_month = 3 * (quarter - 1) + 1
    end_month = start_month + 2
    start = date(year, start_month, 1)
    end = date(year, end_month, 28)
    return start, end


async def _seed_report_snapshots(
    session: AsyncSession, spec: TenantSpec, company_id: int, now: datetime
) -> None:
    today = now.date()
    year = today.year

    for quarter in (1, 2):
        period_start, period_end = _quarter_bounds(year, quarter)
        session.add(
            ReportSnapshot(
                company_id=company_id,
                report_type="quarterly",
                period_start=period_start,
                period_end=period_end,
                currency="RUB",
                payload_json={
                    "tenant": spec.name,
                    "quarter": f"Q{quarter} {year}",
                    "totals": {
                        "transactions_count": spec.transactions // 4,
                        "payouts_count": spec.payouts // 4,
                    },
                    "anomalies": [],
                },
                created_at=now,
            )
        )

    for offset in (1, 2):
        ref_month = today.month - offset
        ref_year = year
        if ref_month <= 0:
            ref_month += 12
            ref_year -= 1
        period_start = date(ref_year, ref_month, 1)
        period_end = date(ref_year, ref_month, 28)
        session.add(
            ReportSnapshot(
                company_id=company_id,
                report_type="monthly",
                period_start=period_start,
                period_end=period_end,
                currency="RUB",
                payload_json={
                    "tenant": spec.name,
                    "month": f"{ref_year}-{ref_month:02d}",
                    "totals": {
                        "transactions_count": spec.transactions // 12,
                        "payouts_count": spec.payouts // 12,
                    },
                    "anomalies": [],
                },
                created_at=now,
            )
        )

    await session.flush()


async def _truncate_all(session: AsyncSession) -> None:
    tables = ", ".join(TRUNCATE_ORDER)
    await session.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))


async def main() -> None:
    fake = Faker(locale="ru_RU")
    Faker.seed(SEED)
    rng = random.Random(SEED)

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session, session.begin():
        await _truncate_all(session)
        await _seed_reference(session)
        for spec in TENANTS:
            await _seed_tenant(session, spec, fake, rng)

    await get_engine().dispose()
    print("seed: done")


if __name__ == "__main__":
    asyncio.run(main())
