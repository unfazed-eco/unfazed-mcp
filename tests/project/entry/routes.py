import typing as t

from event.mcp import event_app
from unfazed.route import Route, include, mount, path

patterns: t.List[Route] = [
    mount("/ping", app=event_app),
    path("/api/event", routes=include("event.routes")),
]
