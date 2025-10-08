"""
Task loader for VMEvalKit reasoning tasks.
"""

from typing import Dict, Any, Optional
from pathlib import Path


class Task:
    """Base class for reasoning tasks."""
    
    def __init__(self, name: str, difficulty: str = "medium"):
        self.name = name
        self.difficulty = difficulty
        self.problem_image = None
        
    def load_problem(self, image_path: str):
        """Load problem image."""
        self.problem_image = image_path


class TaskLoader:
    """
    Loader for various reasoning tasks.
    
    Tasks require models that support text+image→video generation.
    """
    
    AVAILABLE_TASKS = [
        "maze_solving",
        "mental_rotation",
        "chess",
        "ravens_matrices"
    ]
    
    @classmethod
    def load_task(cls, task_name: str, difficulty: str = "medium", **kwargs) -> Task:
        """
        Load a specific reasoning task.
        
        Args:
            task_name: Name of the task
            difficulty: Task difficulty level
            **kwargs: Additional task parameters
            
        Returns:
            Task instance
        """
        if task_name not in cls.AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {cls.AVAILABLE_TASKS}"
            )
        
        task = Task(name=task_name, difficulty=difficulty)
        
        # Load task-specific data (placeholder)
        if task_name == "maze_solving":
            task.problem_image = "maze_example.png"
        elif task_name == "chess":
            task.problem_image = "chess_position.png"
        # etc...
        
        return task
    
    @classmethod
    def list_tasks(cls) -> list:
        """List all available tasks."""
        return cls.AVAILABLE_TASKS.copy()
    
    @classmethod
    def register_task(cls, name: str, task_class: type):
        """Register a custom task."""
        # Placeholder for custom task registration
        pass
