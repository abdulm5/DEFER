from __future__ import annotations

from typing import Any

from defer.domains.extended_state import (
    AccessGrant,
    NotificationRecord,
    StoredFile,
    WebhookEndpoint,
)
from defer.domains.state import WorldState
from defer.domains.types import ToolExecutionResult, deterministic_id


def register_webhook(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    webhook_id = args.get("webhook_id", deterministic_id("wh"))
    endpoint = WebhookEndpoint(
        webhook_id=webhook_id,
        url=args["url"],
        event_type=args.get("event_type", "all"),
        active=True,
        propagated=False,
    )
    state.webhooks[webhook_id] = endpoint
    return ToolExecutionResult(
        ok=True,
        observation={"webhook_id": webhook_id, "active": True},
        state_diff={"webhooks": {webhook_id: endpoint.model_dump(mode="json")}},
        pending_fields=["webhook_propagation"],
        irreversible_commit=False,
    )


def upload_file(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    file_id = args.get("file_id", deterministic_id("file"))
    shared = bool(args.get("shared", False))
    stored = StoredFile(
        file_id=file_id,
        filename=args["filename"],
        size_bytes=int(args.get("size_bytes", 0)),
        shared=shared,
        replicated=False,
    )
    state.stored_files[file_id] = stored
    return ToolExecutionResult(
        ok=True,
        observation={"file_id": file_id, "shared": shared},
        state_diff={"stored_files": {file_id: stored.model_dump(mode="json")}},
        pending_fields=["file_replication"],
        irreversible_commit=shared,
    )


def modify_access(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    grant_id = args.get("grant_id", deterministic_id("grant"))
    grant = AccessGrant(
        grant_id=grant_id,
        principal=args["principal"],
        resource=args["resource"],
        permission=args.get("permission", "read"),
        propagated=False,
    )
    state.access_grants[grant_id] = grant
    return ToolExecutionResult(
        ok=True,
        observation={"grant_id": grant_id, "permission": grant.permission},
        state_diff={"access_grants": {grant_id: grant.model_dump(mode="json")}},
        pending_fields=["permission_propagation"],
        irreversible_commit=True,
    )


def send_notification(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    notification_id = args.get("notification_id", deterministic_id("notif"))
    delivered = bool(args.get("deliver_now", False))
    record = NotificationRecord(
        notification_id=notification_id,
        channel=args.get("channel", "email"),
        message=args["message"],
        recipients=args.get("recipients", []),
        delivered=delivered,
    )
    state.notifications[notification_id] = record
    return ToolExecutionResult(
        ok=True,
        observation={"notification_id": notification_id, "delivered": delivered},
        state_diff={"notifications": {notification_id: record.model_dump(mode="json")}},
        pending_fields=["notification_delivery"] if delivered else [],
        irreversible_commit=delivered,
    )


EXTENDED_TOOLS = {
    "register_webhook": register_webhook,
    "upload_file": upload_file,
    "modify_access": modify_access,
    "send_notification": send_notification,
}
