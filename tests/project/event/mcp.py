from contextlib import AbstractAsyncContextManager
from typing import Any, Mapping

from fastmcp.server.http import StarletteWithLifespan
from unfazed.core import Unfazed
from unfazed.lifespan import BaseLifeSpan

from unfazed_mcp.backends import UnfazedFastMCP

from .endpoints import ping

event_mcp = UnfazedFastMCP(
    name="event",
    version="1.0.0",
)
event_mcp.tool(ping)

event_app: StarletteWithLifespan = event_mcp.http_app(
    path="/mcp", transport="streamable-http"
)


class EventMCPLifespan(BaseLifeSpan):
    def __init__(self, app: Unfazed) -> None:
        super().__init__(app)
        self.mcp_lifespan_context: (
            AbstractAsyncContextManager[None, bool | None]
            | AbstractAsyncContextManager[Mapping[str, Any], bool | None]
            | None
        ) = None

    async def on_startup(self) -> None:
        self.mcp_lifespan_context = event_app.lifespan(self.unfazed)
        if self.mcp_lifespan_context is not None:
            await self.mcp_lifespan_context.__aenter__()

    async def on_shutdown(self) -> None:
        if self.mcp_lifespan_context is not None:
            await self.mcp_lifespan_context.__aexit__(None, None, None)
