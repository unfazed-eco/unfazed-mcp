"""
Unfazed FastMCP implementation with Request parameter injection.
"""

import inspect
import json
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    get_args,
    overload,
)

import pydantic_core
from fastmcp import Context, FastMCP
from fastmcp.server.dependencies import get_context, get_http_request
from fastmcp.tools.tool import (
    FunctionTool,
    ParsedFunction,
    ToolAnnotations,
    ToolResult,
    _convert_to_content,
)
from fastmcp.utilities.types import NotSet, NotSetT, find_kwarg_by_type
from mcp.types import (
    AnyFunction,
    AudioContent,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)
from pydantic import BaseModel, ConfigDict, create_model
from unfazed.http import HttpRequest, JsonResponse
from unfazed.route import params as p
from unfazed.route.endpoint import SUPPOTED_REQUEST_TYPE


class UnfazedParsedFunction(ParsedFunction):
    """
    Unfazed ParsedFunction that excludes Request parameters from schema.
    """

    @classmethod
    def _convert_args_to_params(cls, fn: Callable[..., Any]) -> Dict[str, tuple]:
        sig: inspect.Signature = inspect.signature(fn)
        fn_name: str = getattr(fn, "__name__", None) or fn.__class__.__name__
        body_params: Dict[str, tuple] = {}
        for _, param in sig.parameters.items():
            # skip var positional and var keyword
            if param.kind in [
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ]:
                continue

            # skip unfazed request
            mro: list[type] = getattr(param.annotation, "__mro__", [])
            if HttpRequest in mro:
                continue

            if Context in mro:
                continue

            # mcp fucntion only support Json type hint
            annotation: type = param.annotation
            if hasattr(annotation, "_name") and annotation._name == "Annotated":
                metadata: list[Any] = getattr(annotation, "__metadata__", [])

                model_or_field = metadata[0]
                if not isinstance(model_or_field, p.Json):
                    continue

            # mcp function should not have default value
            if param.default != inspect.Parameter.empty:
                raise ValueError(
                    f"mcp function {fn_name} should not have default value, use Annotated instead"
                    "change `{param.name}: {param.annotation} = {param.default}` "
                    "to `{param.name}: Annotated[{param.annotation}, Field(default={param.default})]`"
                )

            # get the origin class of the annotation
            origin_cls: type = param.annotation
            if (
                hasattr(param.annotation, "_name")
                and param.annotation._name == "Annotated"
            ):
                origin_cls = param.annotation.__origin__

            if not issubclass(origin_cls, SUPPOTED_REQUEST_TYPE):
                raise ValueError(
                    f"unsupported type hint for `{param.name}` in mcp function: {fn_name}"
                    "supported types are: str, int, float, list, BaseModel"
                )

            body_params[param.name] = (origin_cls, model_or_field)

        return body_params

    @classmethod
    def create_param_model(
        cls, fn: Callable[..., Any], body_params: Dict[str, Any]
    ) -> Optional[BaseModel]:
        fields: List[str] = []
        bases: List[Type[BaseModel]] = []
        field_difinitions: Dict[str, Annotated[Any, "p.Parms"]] = {}
        model_name: str = fn.__name__
        for name, define in body_params.items():
            annotation, fieldinfo = define
            model_name += f"_{name}"
            if issubclass(annotation, BaseModel):
                for field_name, field in annotation.model_fields.items():
                    if field_name in fields:
                        raise ValueError(f"field {field_name} already exists")

                    fields.append(field_name)

                    if not field.alias:
                        field.alias = field_name

                    if not field.title:
                        field.title = field_name

                bases.append(annotation)
            # field info
            else:
                if name in fields:
                    raise ValueError(
                        f"Error for {fn.__name__}: field {name} already exists"
                    )

                fields.append(name)

                if not fieldinfo.alias:
                    fieldinfo.alias = name

                if not fieldinfo.title:
                    fieldinfo.title = name

                field_difinitions[name] = (annotation, fieldinfo)

        config_dict = ConfigDict(arbitrary_types_allowed=True)
        for base in bases:
            config_dict.update(base.model_config)

        model_cls: BaseModel | None = None

        if bases:
            base_one: type[BaseModel] = bases[0]
            base_one.model_config = config_dict

            model_cls = create_model(
                model_name,
                __base__=tuple(bases) or None,
                **field_difinitions,
            )  # type: ignore

        return model_cls

    @classmethod
    def check_response_valid(cls, fn: Callable[..., Any]) -> bool:
        response_model: type = inspect.signature(fn).return_annotation
        # Check if return type is JsonResponse directly
        if response_model == JsonResponse:
            return True
        # Check if return type is Annotated with JsonResponse
        args: tuple[type, ...] = get_args(response_model)
        if args and args[0] == JsonResponse:
            return True
        return False

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        exclude_args: list[str] | None = None,
        validate: bool = True,
        wrap_non_object_output_schema: bool = True,
    ) -> ParsedFunction:
        """Create an UnfazedParsedFunction from a function."""

        if find_kwarg_by_type(fn, kwarg_type=HttpRequest):
            if not cls.check_response_valid(fn):
                raise ValueError(
                    f"""mcp function {fn.__name__} should return JsonResponse, if won't return JsonResponse, please use this function as a tool,
                    unfazed_mcp = UnfazedMCP()
                    @unfazed_mcp.tool()
                    def fn(1:int) -> dict:
                        return {"message": "Hello, World!"}
                    """
                )

            fn_name: str = getattr(fn, "__name__", None) or fn.__class__.__name__
            fn_doc: str | None = inspect.getdoc(fn)

            if not inspect.isroutine(fn):
                fn = fn.__call__  # type: ignore
            # if the fn is a staticmethod, we need to work with the underlying function
            if isinstance(fn, staticmethod):
                fn = fn.__func__

            body_params: Dict[str, tuple] = cls._convert_args_to_params(fn)
            model_cls: Optional[BaseModel] = cls.create_param_model(fn, body_params)
            input_schema: dict[str, Any] = {}
            if model_cls is not None:
                input_schema = model_cls.model_json_schema()
            # Allow functions with only HttpRequest parameter (empty input_schema)
            # This is valid for functions that only need the request context

            return cls(
                fn=fn,
                name=fn_name,
                description=fn_doc,
                input_schema=input_schema,
                output_schema=None,
            )
        else:
            return super().from_function(
                fn=fn,
                exclude_args=exclude_args,
                validate=validate,
                wrap_non_object_output_schema=wrap_non_object_output_schema,
            )


class UnfazedFunctionTool(FunctionTool):
    """
    Unfazed FunctionTool that supports Request parameter injection.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._has_request_param = False
        self._request_param_name: str | None = None

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        annotations: Any = None,
        exclude_args: list[str] | None = None,
        output_schema: Any = None,
        serializer: Callable[[Any], str] | None = None,
        enabled: bool | None = None,
    ) -> "UnfazedFunctionTool":
        """Create an UnfazedFunctionTool from a function."""

        # check if the function has Request param
        request_kwarg = find_kwarg_by_type(fn, kwarg_type=HttpRequest)

        # use UnfazedParsedFunction to ensure Request param is excluded
        parsed_fn = UnfazedParsedFunction.from_function(fn, exclude_args=exclude_args)

        if name is None and parsed_fn.name == "<lambda>":
            raise ValueError("You must provide a name for lambda functions")

        if isinstance(output_schema, NotSetT):
            output_schema = parsed_fn.output_schema
        elif output_schema is False:
            output_schema = None
        # Note: explicit schemas (dict) are used as-is without auto-wrapping

        # Validate that explicit schemas are object type for structured content
        if output_schema is not None and isinstance(output_schema, dict):
            if output_schema.get("type") != "object":
                raise ValueError(
                    f'Output schemas must have "type" set to "object" due to MCP spec limitations. Received: {output_schema!r}'
                )

        # Handle functions with only HttpRequest parameter (empty input_schema)
        # These functions are valid and should be allowed
        input_schema: dict[str, Any] = parsed_fn.input_schema
        if not input_schema and request_kwarg:
            # For functions with only HttpRequest parameter, provide an empty object schema
            input_schema = {"type": "object", "properties": {}}

        # create tool instance
        tool: UnfazedFunctionTool = cls(
            fn=fn,
            name=name or parsed_fn.name,
            title=title,
            description=description or parsed_fn.description,
            parameters=input_schema,
            output_schema=output_schema,
            annotations=annotations,
            tags=tags or set(),
            serializer=serializer,
            enabled=enabled if enabled is not None else True,
        )

        # mark this mcp tool has request param
        tool._has_request_param = request_kwarg is not None
        tool._request_param_name = request_kwarg or None

        return tool

    def _inject_request_param(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Inject Request parameter if needed."""
        if not self._has_request_param:
            return arguments

        request_param_name: str | None = self._request_param_name
        if request_param_name and request_param_name not in arguments:
            try:
                request: HttpRequest = get_http_request()
                context: Context = get_context()
                setattr(request, "mcp_context", context)
                arguments[request_param_name] = request
            except RuntimeError:
                pass

        return arguments

    def _process_special_response_types(self, result: Any) -> dict[str, Any]:
        if hasattr(result, "__class__") and result.__class__.__name__ == "JsonResponse":
            if hasattr(result, "body"):
                try:
                    if isinstance(result.body, bytes):
                        body_str: str = result.body.decode("utf-8")
                        return json.loads(body_str)
                    else:
                        return result.body
                except Exception:
                    return result.body
            elif hasattr(result, "content"):
                return result.content
        return result

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        body_params: dict[str, Any] = {}

        body_param_models: dict[str, tuple] = (
            UnfazedParsedFunction._convert_args_to_params(self.fn)
        )
        for name, define in body_param_models.items():
            model, _ = define
            if issubclass(model, BaseModel):
                body_params[name] = model(**arguments)

        context_kwarg: str | None = find_kwarg_by_type(self.fn, kwarg_type=Context)
        if context_kwarg and context_kwarg not in body_params:
            body_params[context_kwarg] = get_context()

        body_params = self._inject_request_param(body_params)

        result: Any = self.fn(**body_params)
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, ToolResult):
            return result

        result = self._process_special_response_types(result)

        unstructured_result: List[
            TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource
        ] = _convert_to_content(result, serializer=self.serializer)

        structured_output = None
        # First handle structured content based on output schema, if any
        if self.output_schema is not None:
            if self.output_schema.get("x-fastmcp-wrap-result"):
                # Schema says wrap - always wrap in result key
                structured_output = {"result": result}
            else:
                structured_output = result
        # If no output schema, try to serialize the result. If it is a dict, use
        # it as structured content. If it is not a dict, ignore it.
        if structured_output is None:
            try:
                structured_output = pydantic_core.to_jsonable_python(result)
                if not isinstance(structured_output, dict):
                    structured_output = None
            except Exception:
                pass

        return ToolResult(
            content=unstructured_result,
            structured_content=structured_output,
        )


class UnfazedFastMCP(FastMCP):
    """
    Unfazed FastMCP with Request parameter injection support.

    This class extends FastMCP to automatically inject Request objects
    into tool functions that have Request type annotations.
    """

    @overload
    def tool(
        self,
        name_or_fn: AnyFunction,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        output_schema: dict[str, Any] | None | NotSetT = NotSet,
        annotations: ToolAnnotations | dict[str, Any] | None = None,
        exclude_args: list[str] | None = None,
        enabled: bool | None = None,
    ) -> FunctionTool: ...

    @overload
    def tool(
        self,
        name_or_fn: str | None = None,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        output_schema: dict[str, Any] | None | NotSetT = NotSet,
        annotations: ToolAnnotations | dict[str, Any] | None = None,
        exclude_args: list[str] | None = None,
        enabled: bool | None = None,
    ) -> Callable[[AnyFunction], FunctionTool]: ...

    def tool(
        self,
        name_or_fn: str | AnyFunction | None = None,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        tags: set[str] | None = None,
        output_schema: dict[str, Any] | None | NotSetT = NotSet,
        annotations: ToolAnnotations | dict[str, Any] | None = None,
        exclude_args: list[str] | None = None,
        enabled: bool | None = None,
    ) -> Callable[[AnyFunction], FunctionTool] | FunctionTool:
        """
        Unfazed tool decorator that supports Request parameter injection.

        This decorator works exactly like FastMCP's tool decorator, but
        also supports automatic injection of Request objects.
        """
        tool_name: Optional[str] = None

        if isinstance(annotations, dict):
            annotations = ToolAnnotations(**annotations)

        if isinstance(name_or_fn, classmethod):
            raise ValueError(
                inspect.cleandoc(
                    """
                    To decorate a classmethod, first define the method and then call
                    tool() directly on the method instead of using it as a
                    decorator. See https://gofastmcp.com/patterns/decorating-methods
                    for examples and more information.
                    """
                )
            )

        if inspect.isroutine(name_or_fn):
            fn: Any = name_or_fn
            tool_name = name

            tool: UnfazedFunctionTool = UnfazedFunctionTool.from_function(
                fn,
                name=tool_name,
                title=title,
                description=description,
                tags=tags,
                output_schema=output_schema,
                annotations=annotations,
                exclude_args=exclude_args,
                serializer=self._tool_serializer,
                enabled=enabled,
            )
            self.add_tool(tool)
            return tool

        elif isinstance(name_or_fn, str):
            if name is not None:
                raise TypeError(
                    "Cannot specify both a name as first argument and as keyword argument. "
                    f"Use either @tool('{name_or_fn}') or @tool(name='{name}'), not both."
                )
            tool_name = name_or_fn
        elif name_or_fn is None:
            tool_name = name
        else:
            raise TypeError(
                f"First argument to @tool must be a function, string, or None, got {type(name_or_fn)}"
            )

        return partial(
            self.tool,
            name=tool_name,
            title=title,
            description=description,
            tags=tags,
            output_schema=output_schema,
            annotations=annotations,
            exclude_args=exclude_args,
            enabled=enabled,
        )  # type: ignore
