"""Tool extensions for the hook system."""

from typing import Any, Dict, Optional, TypeVar

from traitlets import HasTraits, Instance
from pydantic_ai import Tool
from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from .base import Hook
from .tool_state import ToolStateManager

T = TypeVar('T')  # Return type for hooks

class ToolHooks(HasTraits):
    """A container for tool-specific hooks."""
    # Tool-specific hooks
    before = Instance(Hook, allow_none=True)
    after = Instance(Hook, allow_none=True)
    error = Instance(Hook, allow_none=True)

    def __init__(self) -> None:
        """Initialize empty hooks."""
        super().__init__()
        # Initialize all hooks to None
        self.before = None
        self.after = None
        self.error = None

    async def run_before(self, context: Dict[str, Any]) -> Optional[Any]:
        """Run before hook.
        
        Args:
            context: Context containing:
                    - tool: The tool being run
                    - args: Tool arguments
                    - kwargs: Tool keyword arguments
        
        Returns:
            The hook's result if one exists
        """
        if self.before:
            return await self.before(context)
        return None

    async def run_after(self, context: Dict[str, Any]) -> Optional[Any]:
        """Run after hook.
        
        Args:
            context: Context containing:
                    - tool: The tool that ran
                    - args: Tool arguments
                    - kwargs: Tool keyword arguments
                    - result: Tool result
        
        Returns:
            The hook's result if one exists
        """
        if self.after:
            return await self.after(context)
        return None

    async def run_error(self, context: Dict[str, Any]) -> Optional[Any]:
        """Run error hook.
        
        Args:
            context: Context containing:
                    - tool: The tool that failed
                    - args: Tool arguments
                    - kwargs: Tool keyword arguments
                    - error: The error that occurred
        
        Returns:
            The hook's result if one exists
        """
        if self.error:
            return await self.error(context)
        return None

class ReactiveTool(Tool):
    """Tool with hook support."""

    def __init__(
        self,
        name: str,
        function: Any,
        *,
        takes_ctx: bool | None = None,
        max_retries: int | None = None,
        description: str | None = None,
        **kwargs
    ):
        """Create a new reactive tool instance.

        Args:
            name: Name of the tool
            function: The function to call
            takes_ctx: Whether the function takes a RunContext
            max_retries: Maximum number of retries
            description: Tool description
            **kwargs: Additional arguments for Tool
        """
        super().__init__(
            function=function,
            name=name,
            takes_ctx=takes_ctx,
            max_retries=max_retries,
            description=description,
            **kwargs
        )
        self.on = ToolHooks()  # Our discoverable hook API
        self.state = ToolStateManager()  # State management

    async def __call__(self, *args, **kwargs) -> Any:
        """Call the tool with hook support.

        This extends the base __call__ method to add hook calls:
        - before running
        - after completion
        - on failure
        """
        # Create context
        context = {
            'tool': self,
            'name': self.name,
            'args': args,
            'kwargs': kwargs
        }

        # Update state to running
        self.state.mark_running(context)

        # Run before hook
        await self.on.run_before(context)

        try:
            # Run tool
            result = await self.function(*args, **kwargs)

            # Update state to completed
            self.state.mark_completed(result)

            # Run after hook
            context['result'] = result
            await self.on.run_after(context)

            return result

        except Exception as e:
            # Update state to error
            self.state.mark_error(e)

            # Run error hook
            context['error'] = e
            await self.on.run_error(context)
            raise

class ReactiveToolset(AbstractToolset):
    """A toolset that adds hook support to tools."""

    def __init__(self, tools: list[ReactiveTool]) -> None:
        """Initialize with a list of reactive tools."""
        self.tools = tools

    @property
    def id(self) -> str:
        """Get the toolset ID."""
        return "reactive"

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Get the tools for this run step."""
        tools = {}
        for tool in self.tools:
            tools[tool.name] = ReactiveToolsetTool(self, tool, tool.tool_def)
        return tools

    async def call_tool(self, name: str, args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]) -> Any:
        """Call a tool with the given arguments."""
        if not isinstance(tool, ReactiveToolsetTool):
            raise TypeError(f"Expected ReactiveToolsetTool, got {type(tool)}")
        # Create context for hooks
        context = {
            'tool': tool.tool,
            'name': tool.tool.name,
            'args': args,
            'kwargs': {}
        }
        # Update state to running
        tool.tool.state.mark_running(context)
        # Run before hook
        await tool.tool.on.run_before(context)
        try:
            # Run tool
            result = await tool.tool.function(**args)
            # Update state to completed
            tool.tool.state.mark_completed(result)
            # Run after hook
            context['result'] = result
            await tool.tool.on.run_after(context)
            return result
        except Exception as e:
            # Update state to error
            tool.tool.state.mark_error(e)
            # Run error hook
            context['error'] = e
            await tool.tool.on.run_error(context)
            raise

class ReactiveToolsetTool(ToolsetTool[Any]):
    """A tool that adds hook support."""

    def __init__(self, toolset: AbstractToolset[Any], tool: ReactiveTool, tool_def: ToolDefinition) -> None:
        """Initialize with a reactive tool."""
        super().__init__(
            toolset=toolset,
            tool_def=tool_def,
            max_retries=1,  # Default to 1 retry
            args_validator=tool.function_schema.validator
        )
        self.tool = tool


