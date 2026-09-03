"""Productivity MCP surface: plain @mcp tools over connected accounts.

This is the reference PURE-MCP door: no named-service registration behind it.
Each tool declares which connected-account provider claims it needs (the
``ToolClaimPolicy`` shape under ``connections.delegated_to_kdcube.
connected_accounts``) and enforces them at execution time with
``enforce_tool_requirements`` - so a plain MCP tool answers with the SAME
demand ordering and consent envelopes as the named-services door:

- zero accounts on the backing provider -> connect-first denial (the guided
  connect plan, ending in the agent-grant hand-off);
- account present but unusable for the call -> the account-level consent
  (claim upgrade / agent grant / reconnect / account pick);
- everything resolves -> the wrapped provider tool runs.

Copy this module as the template for your own pure-MCP door: declare the
tool's requirements in ``PRODUCTIVITY_TOOLS``, call ``_prepare()`` +
``enforce_tool_requirements`` first in every tool body, then run the real
provider work.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Mapping

from pydantic import Field

from kdcube_ai_app.apps.chat.sdk.integrations.email.imap_smtp_tools import ImapSmtpMailTools
from kdcube_ai_app.apps.chat.sdk.integrations.google.gmail_tools import GmailTools
from kdcube_ai_app.apps.chat.sdk.integrations.mail.realm import (
    MailAccount,
    MailProviderSpec,
    connect_required_envelope,
    discover_mail_providers,
    list_mail_accounts,
    mail_requirement,
)
from kdcube_ai_app.apps.chat.sdk.integrations.slack.tools import SlackTools
from connection_hub.delegated_credentials.credential_view import (
    delegated_credential_view,
)
from connection_hub.delegated_to_kdcube.models import (
    ToolClaimPolicy,
)
from connection_hub.mcp_metadata import (
    kdcube_mcp_icons,
    kdcube_website_url,
    read_only_annotations,
    write_annotations,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.mcp_tool_enforcement import (
    bind_service_connector_apps_from_config,
    enforce_tool_requirements,
    resolve_tool_requirements,
)

from ...services.named_services.request_scope import set_public_base_url_from_request
from ...services.productivity.google_docs import GoogleDocsService
from .productivity_docs import (
    DOCS_PRODUCTIVITY_TOOLS,
    register_google_docs_tools,
)
from .productivity_linkedin import (
    LINKEDIN_PRODUCTIVITY_TOOLS,
    register_linkedin_tools,
)
from .productivity_sheets import (
    SHEETS_PRODUCTIVITY_TOOLS,
    register_google_sheets_tools,
)
from .productivity_web import (
    WEB_PRODUCTIVITY_TOOLS,
    register_web_tools,
)

ConfigFactory = Callable[[], Mapping[str, Any]]

PRODUCTIVITY_MCP_INSTRUCTIONS = """\
This MCP server exposes productivity tools that run on the approving user's
connected accounts (Slack, mail, Google Sheets, Google Docs, LinkedIn). Mail is
a realm across the mail providers this deployment configures (Gmail, IMAP/SMTP
mailboxes such as iCloud Mail): when the user may have several
mailboxes, list accounts first and pass the chosen account_id; a mail call
without account_id and several eligible accounts answers account_required with
labeled candidates, which means ask the user, never pick a provider for them.
For Sheets, use search when the spreadsheet id is unknown, describe before
structural changes, and pass the returned stable ids to read or write tools. For
Docs, use search when the document id is unknown, get before editing, and pass
the returned document id and text indices to the edit and comment tools. For
LinkedIn, list accounts first when several may be connected, then publish; pass
the returned post_urn to the comment tool. LinkedIn exposes no feed, message or
post-content reads here. Web search and web fetch run on no connected account:
search finds and deduplicates pages (use it to FIND), fetch dereferences URLs
you already know (use it to READ), and when the operator configures a domain
allowlist, hosts outside it are dropped or denied server-side with the reason
in the result. Each account tool names the account access it needs. When a call reports a consent requirement, relay the reason
and connection_hub_url to the user instead of retrying blindly:
connect_required, claim_upgrade_required, and reconnect_required are fixed by
the user at connection_hub_url; account_required is fixed by resending the
same call with account_id set from candidates.
"""

# Declarative per-tool requirements, in the SAME shape application tool
# configs use (ToolClaimPolicy.from_tool_config). ``claims`` speak the
# PROVIDER's claim vocabulary - what a connected account of that provider
# row can hold.
PRODUCTIVITY_TOOLS: dict[str, dict[str, Any]] = {
    "productivity_slack_search": {
        "label": "Search Slack",
        "description": "Search Slack messages through the user's connected Slack account.",
        "connections": {
            "delegated_to_kdcube": {
                "connected_accounts": [
                    {"provider_id": "slack", "claims": ["slack:search"]},
                ],
            },
        },
    },
    # Mail is a REALM: which provider a call needs is the USER's choice of
    # mailbox, and which mail providers exist is the DEPLOYMENT's choice (its
    # descriptor). Neither is known when this code is written, so the mail
    # tools declare no provider here. At call time the realm (mail/realm.py)
    # discovers the members from the hub catalog and the tool gates through
    # the same enforcement as every other tool, with an ``any_of`` group over
    # the members as its requirement; the enforcement answers which account
    # was chosen and the call routes to that provider's transport.
    "productivity_mail_accounts": {
        "label": "List mail accounts",
        "description": (
            "List the user's connected mail accounts across providers, with "
            "what each may do. Reads KDCube's own connection records; takes "
            "no account_id and needs no provider claim."
        ),
    },
    "productivity_mail_search": {
        "label": "Search mail",
        "description": "Search one of the user's connected mail accounts.",
    },
    "productivity_mail_get": {
        "label": "Read mail message",
        "description": "Read one message from one of the user's connected mail accounts.",
    },
    "productivity_mail_draft": {
        "label": "Draft mail message",
        "description": (
            "Create a DRAFT in one of the user's connected mail accounts "
            "without sending it."
        ),
    },
    **SHEETS_PRODUCTIVITY_TOOLS,
    **DOCS_PRODUCTIVITY_TOOLS,
    **LINKEDIN_PRODUCTIVITY_TOOLS,
    **WEB_PRODUCTIVITY_TOOLS,
}


def tool_requirements(tool_name: str) -> list[dict[str, Any]]:
    """The tool's declared connected-account requirements, parsed through the
    canonical ToolClaimPolicy reader (so the declaration and the enforcement
    can never drift on shape)."""
    policy = ToolClaimPolicy.from_tool_config(
        tool_name, PRODUCTIVITY_TOOLS.get(tool_name) or {}
    )
    return [item.to_dict() for item in policy.connected_accounts]


def build_productivity_mcp_app(
    *,
    name: str,
    config_factory: ConfigFactory,
    tenant_factory: Callable[[], str],
    project_factory: Callable[[], str],
    request: Any = None,
):
    """Build the managed productivity MCP surface (plain tools, stateless)."""

    try:
        from kdcube_ai_app.apps.chat.sdk.runtime.mcp.server import KDCubeMCPServer
        from mcp.types import Icon, ToolAnnotations
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise ImportError("mcp server SDK is not installed") from exc

    icons = kdcube_mcp_icons(Icon, request=request)
    mcp = KDCubeMCPServer(
        name,
        instructions=PRODUCTIVITY_MCP_INSTRUCTIONS,
        icons=icons,
        website_url=kdcube_website_url(request=request),
    )

    slack = SlackTools()
    gmail = GmailTools()
    imap_transports: dict[str, ImapSmtpMailTools] = {}

    def _mail_transport(spec: MailProviderSpec) -> Any:
        """The transport for a realm member: the shared Gmail tools, or one
        IMAP/SMTP transport per provider instance carrying its hosts."""
        if spec.transport == "gmail":
            return gmail
        transport = imap_transports.get(spec.provider_id)
        if transport is None:
            transport = ImapSmtpMailTools(
                provider_id=spec.provider_id,
                connector_app_id=spec.connector_app_id,
                settings=spec.settings,
                label=spec.label,
            )
            imap_transports[spec.provider_id] = transport
        return transport

    def _prepare() -> None:
        """Per-call request binding, mirroring the named-services bridge:

        - the public origin the client connected to (absolute out-of-band
          URLs);
        - the surface's connector-app declaration (which connector app serves
          each provider - never a user pick);
        - the calling client's delegated identity + per-account claim scope,
          so connected-account resolution is default-CLOSED for delegated
          callers (empty binding = nothing granted -> agent-grant consent).
        """
        set_public_base_url_from_request(request)
        bind_service_connector_apps_from_config(dict(config_factory() or {}))
        from connection_hub.agent_account_scope import (
            set_agent_account_scope,
            set_agent_identity,
        )

        view = delegated_credential_view(request)
        set_agent_account_scope(view.account_scope)
        set_agent_identity(
            client_id=view.client_id,
            resource=view.resources[0] if view.resources else "",
        )

    async def _enforce(
        tool_name: str,
        operation: str,
        account_id: str = "",
        requirements: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """``requirements`` overrides the tool's static declaration for calls
        that have already chosen an account: the realm passes the chosen
        account's own provider requirement, so an iCloud pick is enforced as
        iCloud and never as the Gmail default."""
        _prepare()
        return await enforce_tool_requirements(
            request,
            tool_name=tool_name,
            operation=operation,
            requirements=(
                requirements if requirements is not None else tool_requirements(tool_name)
            ),
            account_id=account_id,
            tenant=tenant_factory(),
            project=project_factory(),
        )

    async def _route_mail(
        tool_name: str, operation: str, need: str, account_id: str
    ) -> tuple[MailAccount | None, dict[str, Any] | None]:
        """One mail call's gate and routing.

        The gate is the declared shape every other tool uses, with the realm's
        ``any_of`` group as its requirement: the group's members are the mail
        providers this deployment configures (discovered from the hub catalog),
        each with the claim this verb needs there. Enforcement then decides
        exactly as for any tool: one eligible account proceeds, several across
        providers answer account_required with labeled candidates, none leads
        with the connect-first consent naming every member. What comes back is
        the account chosen, so the call routes to that provider's transport.

        Returns (account, denial); a denial is the tool's result."""
        _prepare()
        specs = await discover_mail_providers()
        if not specs:
            return None, connect_required_envelope(
                where=tool_name, need=need, specs=[],
                tenant=tenant_factory(), project=project_factory(),
            )
        accounts = await list_mail_accounts(specs=specs)
        wanted = str(account_id or "").strip()
        if wanted and not any(item.account_id == wanted for item in accounts):
            return None, {
                "ok": False,
                "error": {
                    "code": "account_not_found",
                    "message": f"No connected mail account has id {wanted!r}.",
                    "where": tool_name,
                    "retryable": True,
                },
                "ret": {
                    "reason": "account_not_found",
                    "account_id": wanted,
                    "candidates": [item.public_dict() for item in accounts if item.allows(need)],
                },
            }

        async def _accounts_of(provider_id: str) -> list[Any]:
            return [item for item in accounts if item.provider.provider_id == provider_id]

        resolution = await resolve_tool_requirements(
            request,
            tool_name=tool_name,
            operation=operation,
            requirements=[mail_requirement(specs, need)],
            account_id=wanted,
            tenant=tenant_factory(),
            project=project_factory(),
            accounts_lister=_accounts_of,
        )
        if resolution.denial is not None:
            return None, resolution.denial
        chosen = resolution.account_for(0)
        if chosen is None:
            return None, connect_required_envelope(
                where=tool_name, need=need, specs=specs,
                tenant=tenant_factory(), project=project_factory(),
            )
        account = next(
            (item for item in accounts if item.account_id == chosen.account_id), None
        )
        if account is None:
            spec = next((item for item in specs if item.provider_id == chosen.provider_id), specs[0])
            account = MailAccount(account_id=chosen.account_id, provider=spec)
        return account, None

    @mcp.tool(
        name="productivity_slack_search",
        title="Search Slack",
        description=(
            "Search Slack messages visible to the approving user's connected "
            "Slack account. Returns {ok, error, ret}; ret contains matching "
            "messages with channel, author, text, permalink."
        ),
        annotations=read_only_annotations(ToolAnnotations, title="Search Slack"),
        structured_output=False,
    )
    async def _productivity_slack_search(
        query: Annotated[str, Field(description="Slack search query.")],
        count: Annotated[
            int,
            Field(ge=1, le=20, description="Maximum results to return, 1-20."),
        ] = 10,
        account_id: Annotated[
            str,
            Field(
                description=(
                    "Optional connected account id when the user has several "
                    "Slack workspaces."
                )
            ),
        ] = "",
    ) -> dict[str, Any]:
        denial = await _enforce("productivity_slack_search", "search", account_id)
        if denial is not None:
            return denial
        return await slack.search_slack(query=query, count=count, account_id=account_id)

    @mcp.tool(
        name="productivity_mail_accounts",
        title="List mail accounts",
        description=(
            "List the approving user's connected mail accounts across the "
            "configured mail providers with account_id, address, and what "
            "each may do (read, draft, send). Call this first when the user "
            "has more than one mailbox, then pass the chosen account_id to the "
            "other mail tools. Returns {ok, error, ret}."
        ),
        annotations=read_only_annotations(ToolAnnotations, title="List mail accounts"),
        structured_output=False,
    )
    async def _productivity_mail_accounts() -> dict[str, Any]:
        _prepare()
        accounts = await list_mail_accounts()
        rows = [account.public_dict() for account in accounts]
        return {
            "ok": True,
            "error": None,
            "ret": {
                "accounts": rows,
                "count": len(rows),
                "providers": sorted({row["provider"] for row in rows}),
            },
        }

    @mcp.tool(
        name="productivity_mail_search",
        title="Search mail",
        description=(
            "Search the approving user's connected mail account. With several "
            "connected mail accounts and no account_id this returns "
            "account_required with the candidates: ask the user, then resend "
            "with account_id. Returns {ok, error, ret}; ret contains message "
            "ids, subjects, senders, dates, snippets."
        ),
        annotations=read_only_annotations(ToolAnnotations, title="Search mail"),
        structured_output=False,
    )
    async def _productivity_mail_search(
        query: Annotated[
            str,
            Field(
                description=(
                    "Mail search query, for example "
                    "'from:alice@example.com newer_than:7d'."
                )
            ),
        ] = "",
        max_results: Annotated[
            int,
            Field(ge=1, le=10, description="Maximum messages to return, 1-10."),
        ] = 5,
        account_id: Annotated[
            str,
            Field(
                description=(
                    "Optional connected account id when the user has several "
                    "mail accounts."
                )
            ),
        ] = "",
    ) -> dict[str, Any]:
        account, denial = await _route_mail(
            "productivity_mail_search", "search", "read", account_id
        )
        if denial is not None:
            return denial
        assert account is not None
        transport = _mail_transport(account.provider)
        if transport is gmail:
            return await gmail.search_gmail(
                query=query, max_results=max_results, account_id=account.account_id
            )
        return await transport.search(
            query=query, max_results=max_results, account_id=account.account_id
        )

    @mcp.tool(
        name="productivity_mail_get",
        title="Read mail message",
        description=(
            "Read one message body and attachment metadata from the approving "
            "user's connected mail account. Use productivity_mail_search first "
            "to get the message id. Returns {ok, error, ret}."
        ),
        annotations=read_only_annotations(ToolAnnotations, title="Read mail message"),
        structured_output=False,
    )
    async def _productivity_mail_get(
        message_id: Annotated[
            str,
            Field(description="Mail message id returned by productivity_mail_search."),
        ],
        include_html: Annotated[
            bool, Field(description="Include the HTML body in the result.")
        ] = False,
        account_id: Annotated[
            str,
            Field(
                description=(
                    "Optional connected account id when the user has several "
                    "mail accounts."
                )
            ),
        ] = "",
    ) -> dict[str, Any]:
        account, denial = await _route_mail(
            "productivity_mail_get", "get", "read", account_id
        )
        if denial is not None:
            return denial
        assert account is not None
        transport = _mail_transport(account.provider)
        if transport is gmail:
            return await gmail.read_gmail_message(
                message_id=message_id, include_html=include_html, account_id=account.account_id
            )
        return await transport.read_message(
            message_id=message_id, include_html=include_html, account_id=account.account_id
        )

    @mcp.tool(
        name="productivity_mail_draft",
        title="Draft mail message",
        description=(
            "Create a DRAFT email in the approving user's connected mail "
            "account without sending it. The person reviews and sends the "
            "draft in their own mail client; this surface has no send tool. "
            "Attachments ride inline as base64 entries. Returns {ok, error, "
            "ret}; ret carries the draft id."
        ),
        annotations=write_annotations(ToolAnnotations, title="Draft mail message"),
        structured_output=False,
    )
    async def _productivity_mail_draft(
        to: Annotated[
            str,
            Field(
                description=(
                    "Comma, semicolon, or newline separated recipient email "
                    "addresses. May be empty for a recipientless draft."
                )
            ),
        ] = "",
        subject: Annotated[str, Field(description="Email subject.")] = "",
        body_markdown: Annotated[
            str,
            Field(description="Markdown body stored as text and HTML."),
        ] = "",
        cc: Annotated[
            str,
            Field(description="Optional cc recipients, same separators as `to`."),
        ] = "",
        bcc: Annotated[
            str,
            Field(description="Optional bcc recipients, same separators as `to`."),
        ] = "",
        body_html: Annotated[
            str,
            Field(description="Optional complete HTML body. Leave empty when using body_markdown."),
        ] = "",
        attachments_base64: Annotated[
            str,
            Field(
                description=(
                    "Optional JSON list of {filename, content_base64, "
                    "mime_type?} attachments, up to 25MB decoded each."
                )
            ),
        ] = "",
        account_id: Annotated[
            str,
            Field(
                description=(
                    "Optional connected account id when the user has several "
                    "mail accounts."
                )
            ),
        ] = "",
    ) -> dict[str, Any]:
        account, denial = await _route_mail(
            "productivity_mail_draft", "draft", "draft", account_id
        )
        if denial is not None:
            return denial
        assert account is not None
        transport = _mail_transport(account.provider)
        if transport is not gmail:
            return await transport.create_draft(
                to=to,
                subject=subject,
                body_markdown=body_markdown,
                cc=cc,
                bcc=bcc,
                body_html=body_html,
                attachments_base64=attachments_base64,
                account_id=account.account_id,
            )
        return await gmail.create_gmail_draft(
            to=to,
            subject=subject,
            body_markdown=body_markdown,
            cc=cc,
            bcc=bcc,
            body_html=body_html,
            attachments_base64=attachments_base64,
            account_id=account.account_id,
        )

    register_google_sheets_tools(
        mcp=mcp,
        tool_annotations_type=ToolAnnotations,
        enforce=_enforce,
    )

    register_google_docs_tools(
        mcp,
        service=GoogleDocsService(),
        enforce_tool=_enforce,
    )

    register_linkedin_tools(
        mcp=mcp,
        tool_annotations_type=ToolAnnotations,
        enforce=_enforce,
    )

    register_web_tools(
        mcp=mcp,
        tool_annotations_type=ToolAnnotations,
        config_factory=config_factory,
    )

    return mcp
