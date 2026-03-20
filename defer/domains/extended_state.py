from __future__ import annotations

from pydantic import BaseModel, Field


class WebhookEndpoint(BaseModel):
    webhook_id: str
    url: str
    event_type: str
    active: bool = True
    propagated: bool = False


class StoredFile(BaseModel):
    file_id: str
    filename: str
    size_bytes: int = 0
    shared: bool = False
    replicated: bool = False


class AccessGrant(BaseModel):
    grant_id: str
    principal: str
    resource: str
    permission: str = "read"
    propagated: bool = False


class NotificationRecord(BaseModel):
    notification_id: str
    channel: str
    message: str
    recipients: list[str] = Field(default_factory=list)
    delivered: bool = False
