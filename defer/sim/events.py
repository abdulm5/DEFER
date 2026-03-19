from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable


@dataclass(order=True)
class ScheduledEvent:
    step: int
    insertion_idx: int
    event_id: str = field(compare=False)
    callback: Callable[[], dict] = field(compare=False)
    description: str = field(compare=False, default="")


class EventLoop:
    def __init__(self) -> None:
        self._queue: list[ScheduledEvent] = []
        self._counter = 0
        self.current_step = 0

    def schedule(self, step: int, event_id: str, callback: Callable[[], dict], description: str) -> None:
        event = ScheduledEvent(
            step=step,
            insertion_idx=self._counter,
            event_id=event_id,
            callback=callback,
            description=description,
        )
        self._counter += 1
        heapq.heappush(self._queue, event)

    def advance_to(self, step: int) -> list[dict]:
        emitted: list[dict] = []
        if step < self.current_step:
            raise ValueError("Cannot move event loop backwards.")
        while self._queue and self._queue[0].step <= step:
            next_event = heapq.heappop(self._queue)
            self.current_step = next_event.step
            payload = next_event.callback()
            emitted.append(
                {
                    "event_id": next_event.event_id,
                    "time_step": next_event.step,
                    "description": next_event.description,
                    "payload": payload,
                }
            )
        self.current_step = step
        return emitted

    def drain_all(self) -> list[dict]:
        """
        Emit all queued events in deterministic time order.
        """
        emitted: list[dict] = []
        while self._queue:
            next_step = self._queue[0].step
            emitted.extend(self.advance_to(next_step))
        return emitted
