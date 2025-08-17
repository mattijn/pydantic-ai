"""Tool state management for the hook system."""

from enum import Enum
from typing import Any, Dict, Optional
from traitlets import HasTraits, Enum as TraitEnum, Instance, Dict as TraitDict

class ToolState(str, Enum):
    """Possible states for a tool."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class ToolStateManager(HasTraits):
    """Manages state for a tool."""
    # Current state
    state = TraitEnum(
        ToolState,
        default_value=ToolState.IDLE,
        help="Current state of the tool"
    )

    # Last result
    result = Instance(
        klass=object,
        allow_none=True,
        help="Result from the last execution"
    )

    # Last error
    error = Instance(
        klass=Exception,
        allow_none=True,
        help="Error from the last execution"
    )

    # Call context
    context = TraitDict(
        key_trait=Instance(str),
        value_trait=Instance(object),
        default_value={},
        help="Context from the current/last execution"
    )

    def __init__(self) -> None:
        """Initialize empty state."""
        super().__init__()
        self.state = ToolState.IDLE
        self.result = None
        self.error = None
        self.context = {}

    def mark_running(self, context: Dict[str, Any]) -> None:
        """Mark the tool as running.
        
        Args:
            context: The execution context
        """
        self.state = ToolState.RUNNING
        self.context = context
        self.result = None
        self.error = None

    def mark_completed(self, result: Any) -> None:
        """Mark the tool as completed.
        
        Args:
            result: The execution result
        """
        self.state = ToolState.COMPLETED
        self.result = result
        self.error = None
        # Reset to idle after completion
        self.state = ToolState.IDLE

    def mark_error(self, error: Exception) -> None:
        """Mark the tool as errored.
        
        Args:
            error: The execution error
        """
        self.state = ToolState.ERROR
        self.error = error
        self.result = None
        # Reset to idle after error
        self.state = ToolState.IDLE

    def reset(self) -> None:
        """Reset to idle state."""
        self.state = ToolState.IDLE
        self.result = None
        self.error = None
        self.context = {}

