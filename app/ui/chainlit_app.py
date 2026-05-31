"""Chainlit UI for the partner-portal assistant — demo layer.

UI-only. Handlers resolve a Chainlit Chat Profile into an immutable
(tenant, role) identity, call ``ChatService.astream_ask``, stream the answer
into a message, and render trailing ``Sources`` / ``Suggested action`` blocks.

Identity model: production passes ``(company_id, role)`` from the auth token,
fixed per session. We mirror that with Chat Profiles locked at chat start;
switching identity = "New Chat" → different profile.

Invariant (``rules.md`` §2.1): this is the ONLY module that may import
``chainlit``, and it must NOT import ``app.graph`` / ``app.rag`` / ``app.tools``
/ ``app.db`` — everything goes through ``app.services`` and ``app.config``. The
compiled graph is reached via ``fastapi_app.state.graph`` (set by the lifespan),
not a global.
"""

import uuid
from dataclasses import dataclass

import chainlit as cl
import structlog

from app.main import app as fastapi_app
from app.services.chat_service import ChatResult, ChatService
from app.services.tenants_view import TenantOption, list_tenants_for_ui
from app.services.trace_view import build_trace_steps

logger = structlog.get_logger(__name__)


_ROLES_DISPLAY: dict[str, str] = {
    "cfo": "CFO",
    "finance_manager": "Финансовый менеджер",
    "accountant": "Бухгалтер",
    "auditor": "Аудитор",
}


@dataclass(frozen=True)
class ChatProfileSpec:
    """One concrete identity available at chat start."""

    name: str  # shown in the Chainlit profile picker; must be unique
    tenant_slug: str  # resolved against TenantOption.slug from the DB
    role: str
    description: str  # markdown rendered in the profile picker

    def role_display(self) -> str:
        return _ROLES_DISPLAY.get(self.role, self.role)


# Demo profiles — chosen to cover RBAC, tenant-scoped docs and route variety.
# Not 12 (3 tenants × 4 roles): only the combinations that exercise something
# distinct in the system. Add new profiles when a scenario can't be reached
# from the current set.
_PROFILES: list[ChatProfileSpec] = [
    ChatProfileSpec(
        name="ACME LLC · CFO",
        tenant_slug="acme",
        role="cfo",
        description=(
            "Финансовый директор крупнейшего tenant'а (ACME LLC). "
            "Доступ ко всем таблицам своей компании и ко всем регламентам оператора. "
            "Может запрашивать draft-выгрузку отчётов и эскалации."
        ),
    ),
    ChatProfileSpec(
        name="ACME LLC · Финансовый менеджер",
        tenant_slug="acme",
        role="finance_manager",
        description=(
            "Финансовый менеджер ACME LLC. Видит транзакции, карты, payouts, "
            "лимиты, тарифы, отчётные срезы. Нет доступа к таблицам `employees` "
            "и `audit_log`; чувствительные поля (`ИНН`, `last4`) маскируются."
        ),
    ),
    ChatProfileSpec(
        name="Ostrovok-mock · Финансовый менеджер",
        tenant_slug="ostrovok",
        role="finance_manager",
        description=(
            "Premium-tenant с активным travel-профилем. Внутренние регламенты "
            "поверх общих: travel policy и лимиты на проживание (`tenant:ostrovok`)."
        ),
    ),
    ChatProfileSpec(
        name="CheckScan-mock · Бухгалтер",
        tenant_slug="checkscan",
        role="accountant",
        description=(
            "Бухгалтер небольшого SaaS-клиента. Внутренние регламенты `tenant:checkscan`: "
            "акты с самозанятыми (доступны бухгалтеру и выше), SaaS procurement."
        ),
    ),
    ChatProfileSpec(
        name="CheckScan-mock · Аудитор",
        tenant_slug="checkscan",
        role="auditor",
        description=(
            "Внутренний аудитор CheckScan-mock. Единственная роль с доступом к "
            "`audit_log` своего tenant'а; все документы своей компании + shared."
        ),
    ),
]


_SESSION_TENANT_KEY = "tenant_option"
_SESSION_ROLE_KEY = "user_role"
_SESSION_THREAD_KEY = "thread_id"


def _new_thread_id() -> str:
    return str(uuid.uuid4())


def _resolve_profile(name: str | None) -> ChatProfileSpec | None:
    if not name:
        return None
    return next((profile for profile in _PROFILES if profile.name == name), None)


def _resolve_tenant(catalogue: list[TenantOption], slug: str) -> TenantOption | None:
    return next((option for option in catalogue if option.slug == slug), None)


def _resolve_chat_service() -> ChatService:
    """Borrow the compiled graph from the FastAPI lifespan and wrap it."""
    graph = getattr(fastapi_app.state, "graph", None)
    if graph is None:
        raise RuntimeError(
            "agent graph not initialised — FastAPI lifespan must run before Chainlit handlers"
        )
    return ChatService(graph)


@cl.set_chat_profiles
async def chat_profiles() -> list[cl.ChatProfile]:
    """Profiles act as the demo's stand-in for the production auth flow.

    Chainlit shows them as a picker before the conversation starts; the
    selection is locked for that chat. To switch identity the user starts a
    new chat (= signs in as a different user in production).
    """
    return [
        cl.ChatProfile(name=profile.name, markdown_description=profile.description)
        for profile in _PROFILES
    ]


def _format_identity_banner(spec: ChatProfileSpec, tenant: TenantOption) -> str:
    return (
        f"**Идентичность:** {tenant.name} (company_id={tenant.company_id}) · "
        f"роль `{spec.role}` ({spec.role_display()}).\n\n"
        "Чтобы сменить идентичность — **New Chat** слева, "
        "выберите другой профиль."
    )


@cl.on_chat_start
async def on_chat_start() -> None:
    profile_name = cl.user_session.get("chat_profile")
    spec = _resolve_profile(profile_name)
    if spec is None:
        await cl.Message(
            content=(
                "Не удалось определить профиль. Выберите один в списке слева и нажмите "
                "**New Chat**."
            )
        ).send()
        return

    catalogue = await list_tenants_for_ui()
    tenant = _resolve_tenant(catalogue, spec.tenant_slug)
    if tenant is None:
        await cl.Message(
            content=(
                f"Tenant `{spec.tenant_slug}` отсутствует в базе. "
                "Запустите `make seed` и обновите страницу."
            )
        ).send()
        return

    cl.user_session.set(_SESSION_TENANT_KEY, tenant)
    cl.user_session.set(_SESSION_ROLE_KEY, spec.role)
    cl.user_session.set(_SESSION_THREAD_KEY, _new_thread_id())
    cl.user_session.set("chat_service", _resolve_chat_service())

    await cl.Message(content=_format_identity_banner(spec, tenant)).send()


def _format_source_line(index: int, source: dict) -> str:
    src_type = source.get("type")
    if src_type == "sql":
        sql = (source.get("sql") or "").strip()
        rows = source.get("rows_returned", 0)
        attempts = source.get("attempts")
        bits = [f"rows={rows}"]
        if attempts:
            bits.append(f"attempts={attempts}")
        meta = ", ".join(bits)
        return f"[{index}] SQL · {meta}\n```sql\n{sql}\n```"
    if src_type == "doc":
        title = source.get("doc_title") or source.get("doc_id") or "документ"
        section = source.get("section")
        score = source.get("score")
        suffix_parts = []
        if section:
            suffix_parts.append(f"§ {section}")
        if isinstance(score, (int, float)):
            suffix_parts.append(f"score={score:.3f}")
        suffix = (" · " + " · ".join(suffix_parts)) if suffix_parts else ""
        return f"[{index}] {title}{suffix}"
    return f"[{index}] {source!r}"


async def _send_sources(sources: list[dict]) -> None:
    if not sources:
        return
    lines = [_format_source_line(i, s) for i, s in enumerate(sources, start=1)]
    await cl.Message(content="**Источники**\n\n" + "\n\n".join(lines)).send()


def _format_payload(payload: dict) -> str:
    if not payload:
        return ""
    return "\n".join(f"- **{key}**: {value}" for key, value in payload.items())


async def _send_suggested_action(action: dict) -> None:
    kind = action.get("kind", "unknown")
    title = action.get("title") or kind
    payload = action.get("payload") or {}
    requires = action.get("requires_confirmation", True)
    payload_block = _format_payload(payload)
    suffix = "Требует подтверждения." if requires else "Авто-исполнение запрещено."
    body = (
        f"**Предлагаемое действие · `{kind}`** — {title}\n\n"
        f"{payload_block}\n\n"
        f"_{suffix} Ассистент не выполняет действие сам — это только draft._"
    )
    await cl.Message(content=body).send()


async def _send_trace(result: ChatResult) -> None:
    """Render the agent's post-hoc trace as collapsible Chainlit steps.

    Steps are reconstructed from the already-guarded ``ChatResult`` (see
    ``app.services.trace_view``) — they appear after the answer because the
    R4 streaming model buffers the run; this is a trace summary, not live
    thinking. A rendering hiccup here must never sink the answer, so failures
    are swallowed with a log.
    """
    try:
        for step in build_trace_steps(result):
            async with cl.Step(name=step.name, type=step.step_type) as cl_step:
                cl_step.output = step.content
    except Exception:
        logger.exception("trace_render_failed")


async def _flush_result(answer_msg: cl.Message, result: ChatResult) -> None:
    """Make sure the streamed message reflects the canonical result.

    Token streaming may be empty (LLM unavailable, cached path, etc.); fall
    back to ``result.answer`` so the user always sees something.
    """
    current = (answer_msg.content or "").strip()
    if current != result.answer.strip() and result.answer:
        answer_msg.content = result.answer
        await answer_msg.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    service: ChatService | None = cl.user_session.get("chat_service")
    tenant: TenantOption | None = cl.user_session.get(_SESSION_TENANT_KEY)
    role: str | None = cl.user_session.get(_SESSION_ROLE_KEY)
    thread_id: str | None = cl.user_session.get(_SESSION_THREAD_KEY)

    if service is None or tenant is None or role is None or thread_id is None:
        await cl.Message(
            content=("Сессия не инициализирована. Нажмите **New Chat** слева и выберите профиль.")
        ).send()
        return

    answer_msg = cl.Message(content="")
    await answer_msg.send()

    result: ChatResult | None = None
    try:
        async for chunk in service.astream_ask(
            question=message.content,
            user_role=role,  # type: ignore[arg-type]
            company_id=tenant.company_id,
            thread_id=thread_id,
        ):
            if chunk.kind == "reset":
                answer_msg.content = ""
                await answer_msg.update()
            elif chunk.kind == "token":
                await answer_msg.stream_token(chunk.token)
            elif chunk.kind == "result" and chunk.result is not None:
                result = chunk.result
    except Exception:
        logger.exception("chat_stream_failed")
        answer_msg.content = (
            "Не удалось обработать запрос. Попробуйте ещё раз — если повторится, "
            "проверьте, что Postgres, Qdrant и Ollama запущены."
        )
        await answer_msg.update()
        return

    if result is None:
        answer_msg.content = "Ответ не сформирован: терминальное событие не получено."
        await answer_msg.update()
        return

    await _flush_result(answer_msg, result)
    await _send_trace(result)
    await _send_sources(result.sources)
    if result.suggested_action:
        await _send_suggested_action(result.suggested_action)
