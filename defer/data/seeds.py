from __future__ import annotations

import random
from typing import Iterable

from defer.data.schema import SeedTask


DOMAIN_TEMPLATES = {
    "calendar": [
        ("calendar_meeting", "Schedule a meeting with {name} from {start} to {end}.", "create_calendar_event"),
        ("calendar_followup", "Set up a follow-up call about {topic} at {start}.", "create_calendar_event"),
        ("calendar_review", "Arrange a project review session for {topic} tomorrow.", "create_calendar_event"),
    ],
    "email": [
        ("email_draft", "Draft an email to {name} about {topic}.", "send_email"),
        ("email_send", "Send an urgent message to {name} confirming {topic}.", "send_email"),
        ("email_update", "Email the team with status update for {topic}.", "send_email"),
    ],
    "rest": [
        ("api_upsert", "Update the API resource for {topic} with the latest payload.", "upsert_api_resource"),
        ("api_create", "Create a new API entry for {topic}.", "upsert_api_resource"),
        ("api_sync", "Sync external service data for {topic}.", "upsert_api_resource"),
    ],
    "sql": [
        ("sql_insert", "Insert {topic} record into database table {table}.", "upsert_sql_row"),
        ("sql_update", "Update database row for {topic} in {table}.", "upsert_sql_row"),
        ("sql_repair", "Repair missing values for {topic} in table {table}.", "upsert_sql_row"),
    ],
}

NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Riley", "Sam", "Casey"]
TOPICS = ["roadmap", "incident review", "hiring plan", "budget", "SLA", "migration", "launch"]
TABLES = ["accounts", "events", "tickets", "users", "audit_log"]
TIMES = [
    ("2026-04-01T09:00", "2026-04-01T10:00"),
    ("2026-04-02T11:00", "2026-04-02T11:30"),
    ("2026-04-03T14:00", "2026-04-03T15:00"),
    ("2026-04-04T16:00", "2026-04-04T16:45"),
]

PROMPT_STYLES = [
    "{core}",
    "{core} Keep all constraints exact.",
    "{core} Confirm any ambiguity before finalizing.",
    "User request: {core}",
    "{core} Use minimal tool calls.",
    "{core} Prioritize correctness over speed.",
    "{core} Treat names and dates as case-sensitive.",
    "{core} Do not assume missing fields.",
    "{core} Preserve user intent and policy compliance.",
    "{core} Ask clarification if authorization is unclear.",
    "{core} Avoid irreversible actions without verification.",
    "{core} Re-check stale context before commit.",
    "{core} Ensure argument grounding from available evidence.",
    "{core} Follow enterprise escalation defaults.",
    "{core} Avoid speculative tool arguments.",
    "{core} Prefer reversible workflow when uncertain.",
    "{core} Validate side effects after action.",
    "{core} Keep auditability of reasoning steps.",
    "{core} Respect delayed consistency constraints.",
    "{core} Cross-check if external state may have drifted.",
]


def build_template_split_map(base_template_ids: list[str]) -> dict[str, str]:
    expanded = sorted(
        f"{base_template_id}_style_{style_idx:02d}"
        for base_template_id in base_template_ids
        for style_idx in range(len(PROMPT_STYLES))
    )
    n = len(expanded)
    train_cut = int(n * 0.7)
    val_cut = train_cut + int(n * 0.1)
    mapping: dict[str, str] = {}
    for idx, template_id in enumerate(expanded):
        if idx < train_cut:
            mapping[template_id] = "train"
        elif idx < val_cut:
            mapping[template_id] = "val"
        else:
            mapping[template_id] = "test"
    return mapping


def generate_seed_tasks(tasks_per_domain: int, seed: int) -> list[SeedTask]:
    rng = random.Random(seed)
    tasks: list[SeedTask] = []
    for domain, templates in DOMAIN_TEMPLATES.items():
        split_map = build_template_split_map([base_id for base_id, _, _ in templates])
        for idx in range(tasks_per_domain):
            base_template_id, template, tool = templates[idx % len(templates)]
            style_idx = idx % len(PROMPT_STYLES)
            template_id = f"{base_template_id}_style_{style_idx:02d}"
            name = rng.choice(NAMES)
            topic = rng.choice(TOPICS)
            table = rng.choice(TABLES)
            start, end = rng.choice(TIMES)
            core = template.format(name=name, topic=topic, table=table, start=start, end=end)
            prompt = PROMPT_STYLES[style_idx].format(core=core)

            if domain == "calendar":
                tool_args = {
                    "title": f"{topic} with {name}",
                    "start_time": start,
                    "end_time": end,
                    "attendees": [f"{name.lower()}@example.com"],
                    "tentative": True,
                }
                expects_irreversible = False
            elif domain == "email":
                tool_args = {
                    "subject": f"{topic.title()} update",
                    "body": f"Please review the {topic}.",
                    "to": [f"{name.lower()}@example.com"],
                    "send_now": True,
                }
                expects_irreversible = True
            elif domain == "rest":
                tool_args = {
                    "resource_id": f"resource_{idx}",
                    "payload": {"topic": topic, "owner": name, "status": "active"},
                }
                expects_irreversible = False
            else:
                tool_args = {
                    "table": table,
                    "primary_key": f"{topic}_{idx}",
                    "values": {"topic": topic, "owner": name, "status": "open"},
                }
                expects_irreversible = False

            task_id = f"{domain}_{idx:04d}"
            split = split_map[template_id]
            requires_refresh = idx % 5 == 0
            tasks.append(
                SeedTask(
                    task_id=task_id,
                    domain=domain,
                    template_id=template_id,
                    split=split,
                    prompt=prompt,
                    required_tool=tool,
                    tool_args=tool_args,
                    expects_irreversible=expects_irreversible,
                    requires_refresh=requires_refresh,
                )
            )
    return tasks


def as_json_rows(tasks: Iterable[SeedTask]) -> list[dict]:
    return [task.model_dump(mode="json") for task in tasks]
