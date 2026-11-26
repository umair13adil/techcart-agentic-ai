"""In-process event bus for the admin dashboard's live feed (Server-Sent Events).
Only real events are published: orders placed, assistant turns, searches. Swap for Azure Web PubSub / Service Bus
when running more than one API instance."""
from __future__ import annotations

import asyncio
import json


class Broadcaster:
    def __init__(self):
        self.queues: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)

    def publish(self, event: dict) -> None:
        for q in list(self.queues):
            if not q.full():
                q.put_nowait(event)


BUS = Broadcaster()


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"
