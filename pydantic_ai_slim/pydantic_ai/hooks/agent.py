"""Agent hook system for Pydantic-AI.

This module provides hook integration for Pydantic-AI's Agent class,
enabling lifecycle, prompt, and global tool hooks.
"""

from typing import Any, Optional, TypeVar

from pydantic_ai import Agent, Tool
from pydantic_ai._agent_graph import CallToolsNode
from pydantic_ai import messages as _messages
from traitlets import HasTraits, Instance

from .base import Hook, HookContainer
from .tool import ReactiveTool

T = TypeVar('T')  # Return type for hooks

class AgentHooks(HasTraits):
    """Hook container for Agent-level hooks.
    
    This class provides hooks for:
    - Lifecycle events (init, start, end)
    - Prompts (instructions, model)
    - Global tool events (before_any_tool, after_any_tool, any_tool_error)
    """
    
    # Lifecycle hooks
    init = Instance(Hook, allow_none=True)
    start = Instance(Hook, allow_none=True)
    end = Instance(Hook, allow_none=True)
    
    # Prompt hooks
    instructions = Instance(HookContainer, allow_none=True)
    model = Instance(HookContainer, allow_none=True)
    
    # Global tool hooks
    before_any_tool = Instance(Hook, allow_none=True)
    after_any_tool = Instance(Hook, allow_none=True)
    any_tool_error = Instance(Hook, allow_none=True)
    
    def __init__(self) -> None:
        """Initialize empty hooks."""
        super().__init__()
        # Initialize all hooks to None
        self.init = None
        self.start = None
        self.end = None
        self.instructions = HookContainer()
        self.model = HookContainer()
        self.before_any_tool = None
        self.after_any_tool = None
        self.any_tool_error = None
    
    async def before_tool(self, context: dict[str, Any]) -> Optional[Any]:
        """Run before_any_tool hook.
        
        Args:
            context: Context containing:
                    - tool: The tool being run
                    - args: Tool arguments
                    - kwargs: Tool keyword arguments
        
        Returns:
            The hook's result if one exists
        """
        return await self.run_before_any_tool(context)
    
    async def after_tool(self, context: dict[str, Any]) -> Optional[Any]:
        """Run after_any_tool hook.
        
        Args:
            context: Context containing:
                    - tool: The tool that ran
                    - args: Tool arguments
                    - kwargs: Tool keyword arguments
                    - result: Tool result
        
        Returns:
            The hook's result if one exists
        """
        return await self.run_after_any_tool(context)
    
    async def on_tool_error(self, context: dict[str, Any]) -> Optional[Any]:
        """Run any_tool_error hook.
        
        Args:
            context: Context containing:
                    - tool: The tool that failed
                    - args: Tool arguments
                    - kwargs: Tool keyword arguments
                    - error: The error that occurred
        
        Returns:
            The hook's result if one exists
        """
        return await self.run_any_tool_error(context)
    
    async def run_init(self, context: dict[str, Any]) -> Optional[Any]:
        """Run init hook."""
        if self.init:
            return await self.init(context)
        return None

    async def run_start(self, context: dict[str, Any]) -> Optional[Any]:
        """Run start hook."""
        if self.start:
            return await self.start(context)
        return None

    async def run_end(self, context: dict[str, Any]) -> Optional[Any]:
        """Run end hook."""
        if self.end:
            return await self.end(context)
        return None

    async def run_instructions(self, context: dict[str, Any]) -> Optional[Any]:
        """Run instructions hook."""
        return await self.instructions.run_hook(context)

    async def run_model(self, context: dict[str, Any]) -> Optional[Any]:
        """Run model hook."""
        return await self.model.run_hook(context)

    async def run_before_any_tool(self, context: dict[str, Any]) -> Optional[Any]:
        """Run before_any_tool hook."""
        if self.before_any_tool:
            return await self.before_any_tool(context)
        return None

    async def run_after_any_tool(self, context: dict[str, Any]) -> Optional[Any]:
        """Run after_any_tool hook."""
        if self.after_any_tool:
            return await self.after_any_tool(context)
        return None

    async def run_any_tool_error(self, context: dict[str, Any]) -> Optional[Any]:
        """Run any_tool_error hook."""
        if self.any_tool_error:
            return await self.any_tool_error(context)
        return None

class ReactiveAgent(Agent):
    """Agent with hook support.
    
    This class extends Pydantic-AI's Agent with a hook system that enables:
    - Lifecycle hooks (init, start, end)
    - Prompt customization (instructions, model)
    - Global tool hooks (before_any_tool, after_any_tool, any_tool_error)
    """
    
    def __init__(self, model, toolsets=None, instructions=None, **kwargs) -> None:
        """Initialize agent with hooks."""
        self._hooks = AgentHooks()  # Create hooks first
        super().__init__(model=model, toolsets=toolsets, instructions=instructions, **kwargs)  # Then initialize base class
        # Now store initial values
        self._hooks.model.value = super().model
        self._hooks.instructions.value = self._instructions  # Use private field
    
    @property
    def model(self):
        """Get the current model."""
        if not hasattr(self, '_hooks'):
            return super().model
        return self._hooks.model.value
    
    @model.setter
    def model(self, value):
        """Set a new model value."""
        if not hasattr(self, '_hooks'):
            super().__setattr__('_model', value)  # Set private field
        else:
            self._hooks.model.value = value
            super().__setattr__('_model', value)  # Keep base class in sync
    
    @property
    def instructions(self):
        """Get the current instructions."""
        if not hasattr(self, '_hooks'):
            return self._instructions  # Use private field
        return self._hooks.instructions.value
    
    @instructions.setter
    def instructions(self, value):
        """Set new instructions value."""
        if not hasattr(self, '_hooks'):
            super().__setattr__('_instructions', value)  # Set private field
        else:
            self._hooks.instructions.value = value
            super().__setattr__('_instructions', value)  # Keep base class in sync
    
    @property
    def on(self) -> AgentHooks:
        """Access to the agent's hooks.
        
        This property provides access to all available hooks:
        
        Lifecycle hooks:
        - init: Called during initialization
        - start: Called when agent starts running
        - end: Called when agent finishes running
        
        Prompt hooks:
        - instructions: Customize instructions
        - model: Adjust model settings
        
        Global tool hooks:
        - before_any_tool: Called before any tool runs
        - after_any_tool: Called after any tool completes
        - any_tool_error: Called if any tool raises an error
        
        Returns:
            The agent's hook container
        """
        return self._hooks
    
    async def run(self, *args, **kwargs) -> Any:
        """Run the agent with hook support.
        
        This extends the base run method to add hook calls at key points:
        - start hook before running
        - end hook after completion
        - error hooks on failure
        """
        # Create context
        context = {
            'agent': self,
            'args': args,
            'kwargs': kwargs
        }
        
        # Run start hook
        await self.on.run_start(context)
        
        try:
            # Run agent using iter() to get access to all events
            async with self.iter(*args, **kwargs) as agent_run:
                async for node in agent_run:
                    # Debug output for all nodes
                    print(f"Node type: {type(node).__name__}")
                    
                    # If it's a tool call node, run our hooks
                    if isinstance(node, CallToolsNode):
                        print("Found CallToolsNode, response parts:", [type(p).__name__ for p in node.model_response.parts])
                        # Process each tool call in the response
                        for part in node.model_response.parts:
                            print("Processing part:", type(part).__name__)
                            if isinstance(part, _messages.ToolCallPart):
                                tool_def = agent_run.ctx.deps.tool_manager.get_tool_def(part.tool_name)
                                if tool_def and agent_run.ctx.deps.tool_manager.tools:
                                    tool = agent_run.ctx.deps.tool_manager.tools[part.tool_name]
                                    tool_context = {
                                        'agent': self,
                                        'tool': tool,
                                        'name': tool_def.name,
                                        'args': part.args,
                                        'kwargs': {}
                                    }
                                    
                                    # Run before hook
                                    await self.on.run_before_any_tool(tool_context)
                                    
                                    try:
                                        # Let the tool execute through tool_manager
                                        result = await agent_run.ctx.deps.tool_manager.handle_call(part)
                                        
                                        # Run after hook
                                        tool_context['result'] = result
                                        await self.on.run_after_any_tool(tool_context)
                                        
                                        return result
                                    except Exception as e:
                                        # Run error hook
                                        tool_context['error'] = e
                                        await self.on.run_any_tool_error(tool_context)
                                        raise
                
                return agent_run.result
        finally:
            # Always run end hook
            await self.on.run_end(context)
    
    async def call_tool(self, tool: Any, *args, **kwargs) -> Any:
        """Call a tool directly with hook support.
        
        This method allows direct tool calls with hooks:
        - before_any_tool before running
        - after_any_tool after completion
        - any_tool_error on failure
        """
        # Create context
        context = {
            'agent': self,
            'tool': tool,
            'name': tool.name,
            'args': args,
            'kwargs': kwargs
        }
        
        # Run before hook
        await self.on.run_before_any_tool(context)
        
        try:
            # Run tool through its __call__ to trigger its hooks
            result = await tool(*args, **kwargs)
            
            # Run after hook
            context['result'] = result
            await self.on.run_after_any_tool(context)
            
            return result
        except Exception as e:
            # Run error hook
            context['error'] = e
            await self.on.run_any_tool_error(context)
            raise