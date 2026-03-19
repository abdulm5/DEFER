from __future__ import annotations

from pydantic import BaseModel, Field


class CalendarEvent(BaseModel):
    event_id: str
    title: str
    start_time: str
    end_time: str
    attendees: list[str] = Field(default_factory=list)
    status: str = "tentative"


class EmailMessage(BaseModel):
    message_id: str
    subject: str
    body: str
    to: list[str] = Field(default_factory=list)
    sent: bool = False


class ApiResource(BaseModel):
    resource_id: str
    payload: dict
    version: int = 1


class SqlRow(BaseModel):
    table: str
    primary_key: str
    values: dict


class WorldState(BaseModel):
    step: int = 0
    user_authorized: bool = True
    calendar_events: dict[str, CalendarEvent] = Field(default_factory=dict)
    emails: dict[str, EmailMessage] = Field(default_factory=dict)
    api_resources: dict[str, ApiResource] = Field(default_factory=dict)
    sql_rows: dict[str, SqlRow] = Field(default_factory=dict)
    stale_cache: dict[str, str] = Field(default_factory=dict)
    pending_jobs: dict[str, dict] = Field(default_factory=dict)
    collaborator_edits: list[dict] = Field(default_factory=list)

    def flat(self) -> dict:
        return self.model_dump(mode="json")
