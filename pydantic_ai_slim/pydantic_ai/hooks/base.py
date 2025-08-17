"""Core hook system for Pydantic-AI.

This module provides the base classes and protocols for the hook system,
enabling type-safe, observable hooks throughout the framework.
"""

from typing import Any, Protocol, TypeVar, runtime_checkable

from traitlets import HasTraits, Instance, observe

R = TypeVar('R')  # Return type for hooks

@runtime_checkable
class Hook(Protocol[R]):
    """Protocol defining the interface for hooks.
    
    All hooks must be async callables that take a context dict and return a value.
    The context dict contains relevant information for the hook execution.
    """
    async def __call__(self, context: dict[str, Any]) -> R:
        """Execute the hook with the given context.
        
        Args:
            context: Dictionary containing context for hook execution.
                    Common keys include:
                    - 'agent': The agent instance
                    - 'tool': The tool instance
                    - 'args': Tool arguments
                    - 'error': Any error that occurred
        
        Returns:
            The hook's result of type R.
        """
        ...

class HookContainer(HasTraits):
    """Base class for objects that can have hooks attached.
    
    This class provides the core functionality for hook management,
    including value storage and observation capabilities.
    """
    
    # The actual hook function
    hook = Instance(Hook, allow_none=True)
    # The stored value that can be observed
    value = Instance(object, allow_none=True)
    
    def __init__(self) -> None:
        """Initialize an empty hook container."""
        super().__init__()
        self.hook = None
        self.value = None
    
    @observe('hook')
    def _hook_changed(self, change: dict[str, Any]) -> None:
        """Called when the hook value changes.
        
        Args:
            change: Dictionary containing:
                   - 'old': Previous hook value
                   - 'new': New hook value
                   - 'name': Name of the trait ('hook')
                   - 'type': Type of change ('change')
        """
        # Implement any necessary validation or side effects
        pass
    
    async def run_hook(self, context: dict[str, Any]) -> Any:
        """Run the hook if one is set.
        
        Args:
            context: Context dictionary to pass to the hook.
        
        Returns:
            The hook's result if a hook is set, None otherwise.
        """
        if self.hook is not None:
            return await self.hook(context)
        return None