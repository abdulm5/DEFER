from defer.sim.events import EventLoop


def test_event_order_is_step_then_insertion_order() -> None:
    loop = EventLoop()
    emitted_names: list[str] = []

    loop.schedule(3, "e3_a", callback=lambda: {"name": "e3_a"}, description="A")
    loop.schedule(2, "e2_a", callback=lambda: {"name": "e2_a"}, description="B")
    loop.schedule(3, "e3_b", callback=lambda: {"name": "e3_b"}, description="C")
    loop.schedule(2, "e2_b", callback=lambda: {"name": "e2_b"}, description="D")

    for event in loop.advance_to(3):
        emitted_names.append(event["payload"]["name"])

    assert emitted_names == ["e2_a", "e2_b", "e3_a", "e3_b"]


def test_drain_all_emits_remaining_events() -> None:
    loop = EventLoop()
    emitted_names: list[str] = []

    loop.schedule(2, "e2", callback=lambda: {"name": "e2"}, description="E2")
    loop.schedule(5, "e5", callback=lambda: {"name": "e5"}, description="E5")
    loop.schedule(4, "e4", callback=lambda: {"name": "e4"}, description="E4")

    # Simulate partial progression first.
    for event in loop.advance_to(2):
        emitted_names.append(event["payload"]["name"])

    for event in loop.drain_all():
        emitted_names.append(event["payload"]["name"])

    assert emitted_names == ["e2", "e4", "e5"]
