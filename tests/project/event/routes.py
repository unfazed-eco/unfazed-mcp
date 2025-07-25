import typing as t

from event.endpoints import ping
from unfazed.route import Route, path

patterns: t.List[Route] = [
    path("/ping", endpoint=ping, methods=["POST"]),
]
