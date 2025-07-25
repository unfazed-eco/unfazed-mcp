import typing as t

from event.schema import BaseResponse, PingRequest, PingRequest2
from unfazed.http import HttpRequest, JsonResponse
from unfazed.route import params as p


async def ping(
    request: HttpRequest,
    ctx: t.Annotated[PingRequest, p.Json()],
    ctx2: t.Annotated[PingRequest2, p.Json()],
) -> t.Annotated[JsonResponse, p.ResponseSpec(model=BaseResponse)]:
    """
    FastMCP compatible ping function.
    """

    result: t.Dict[str, t.Any] = {
        "ctx": ctx.model_dump(),
        "ctx2": ctx2.model_dump(),
        "message": "Ping successful",
        "request_id": -1,
    }
    if hasattr(request, "mcp_context"):
        result["request_id"] = request.mcp_context.request_id

    return JsonResponse(result)
