# Unfazed MCP


## Installation

```bash
pip install unfazed-mcp
```

## Quick Start

### Use in your project

```python

# event.application.py

from unfazed.core import Unfazed
from unfazed.lifespan import BaseLifeSpan

from unfazed_mcp.backends import UnfazedFastMCP

from .endpoints import ping

event_mcp = UnfazedFastMCP(
    name="event",
)
event_mcp.tool(ping)

event_app = event_mcp.http_app(
    path="/mcp", transport="streamable-http"
)


class EventMCPLifespan(BaseLifeSpan):
    def __init__(self, app: Unfazed) -> None:
        super().__init__(app)
        self.mcp_lifespan_context = None

    async def on_startup(self) -> None:
        self.mcp_lifespan_context = event_app.lifespan(self.unfazed)
        if self.mcp_lifespan_context is not None:
            await self.mcp_lifespan_context.__aenter__()

    async def on_shutdown(self) -> None:
        if self.mcp_lifespan_context is not None:
            await self.mcp_lifespan_context.__aexit__(None, None, None)

```

### Mount MCP appliction at root route

```python

# entry.routes.py

from event.mcp import event_app
from unfazed.route import mount

patterns = [
    mount("/ping", app=event_app),
]

```

### Add settings to your project

```python

# settings.py

UNFAZED_SETTINGS = {
    "LIFESPAN": ["event.mcp.EventMCPLifespan"]
}

```
