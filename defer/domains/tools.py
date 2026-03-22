from __future__ import annotations

from typing import Any

from defer.domains.state import ApiResource, CalendarEvent, EmailMessage, SqlRow, WorldState
from defer.domains.types import ToolExecutionResult, deterministic_id, reset_tool_call_counter


def create_calendar_event(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    event_id = args.get("event_id", deterministic_id("evt"))
    event = CalendarEvent(
        event_id=event_id,
        title=args["title"],
        start_time=args["start_time"],
        end_time=args["end_time"],
        attendees=args.get("attendees", []),
        status="tentative" if args.get("tentative", True) else "confirmed",
    )
    state.calendar_events[event_id] = event
    pending = ["calendar_confirmed"] if event.status == "tentative" else []
    return ToolExecutionResult(
        ok=True,
        observation={"event_id": event_id, "status": event.status},
        state_diff={"calendar_events": {event_id: event.model_dump(mode="json")}},
        pending_fields=pending,
        irreversible_commit=False,
    )


def send_email(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    message_id = args.get("message_id", deterministic_id("msg"))
    to_raw = args.get("to", [])
    if isinstance(to_raw, str):
        to_raw = [to_raw]
    email = EmailMessage(
        message_id=message_id,
        subject=args["subject"],
        body=args["body"],
        to=to_raw,
        sent=bool(args.get("send_now", False)),
    )
    state.emails[message_id] = email
    pending = ["email_delivery"] if email.sent else []
    return ToolExecutionResult(
        ok=True,
        observation={"message_id": message_id, "sent": email.sent},
        state_diff={"emails": {message_id: email.model_dump(mode="json")}},
        pending_fields=pending,
        irreversible_commit=email.sent,
    )


def upsert_api_resource(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    resource_id = args.get("resource_id", deterministic_id("res"))
    current = state.api_resources.get(resource_id)
    version = 1 if current is None else current.version + 1
    resource = ApiResource(resource_id=resource_id, payload=args.get("payload", {}), version=version)
    state.api_resources[resource_id] = resource
    return ToolExecutionResult(
        ok=True,
        observation={"resource_id": resource_id, "version": version},
        state_diff={"api_resources": {resource_id: resource.model_dump(mode="json")}},
        pending_fields=["api_replication"],
        irreversible_commit=False,
    )


def upsert_sql_row(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    table = args["table"]
    primary_key = args["primary_key"]
    row_id = f"{table}:{primary_key}"
    row = SqlRow(table=table, primary_key=primary_key, values=args.get("values", {}))
    state.sql_rows[row_id] = row
    return ToolExecutionResult(
        ok=True,
        observation={"row_id": row_id},
        state_diff={"sql_rows": {row_id: row.model_dump(mode="json")}},
        pending_fields=["replica_visibility"],
        irreversible_commit=False,
    )


def refresh_state(state: WorldState, args: dict[str, Any]) -> ToolExecutionResult:
    keys = args.get("keys", [])
    for key in keys:
        state.stale_cache.pop(key, None)
    return ToolExecutionResult(
        ok=True,
        observation={"refreshed_keys": keys},
        state_diff={"stale_cache": state.stale_cache.copy()},
        pending_fields=[],
        irreversible_commit=False,
    )


from defer.domains.extended_tools import EXTENDED_TOOLS

TOOLS = {
    "create_calendar_event": create_calendar_event,
    "send_email": send_email,
    "upsert_api_resource": upsert_api_resource,
    "upsert_sql_row": upsert_sql_row,
    "refresh_state": refresh_state,
    **EXTENDED_TOOLS,
}
