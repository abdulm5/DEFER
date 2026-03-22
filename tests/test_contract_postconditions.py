from __future__ import annotations

from defer.core.contracts import check_postconditions
from defer.domains.contracts import default_contracts
from defer.domains.state import EmailMessage, WorldState


def test_exists_postcondition_requires_non_empty_collection() -> None:
    contracts = default_contracts()
    send_email_contract = contracts["send_email"]

    empty_state = WorldState().flat()
    failures = check_postconditions(send_email_contract, empty_state)
    assert "postcondition_failed:emails:exists" in failures

    non_empty_state = WorldState()
    non_empty_state.emails["m1"] = EmailMessage(
        message_id="m1",
        subject="Status",
        body="Body",
        to=["alex@example.com"],
        sent=True,
    )
    failures_after = check_postconditions(send_email_contract, non_empty_state.flat())
    assert failures_after == []
