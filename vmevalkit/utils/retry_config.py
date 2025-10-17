"""
Smart retry configuration for handling problematic tasks.
"""
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

class RetryConfig:
    """Manages retry logic and problematic task tracking."""
    
    def __init__(self, config_file: str = "data/outputs/retry_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load existing configuration or create new one."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {
            "max_retries_per_task": 2,
            "timeout_limits": {
                "default": 900,      # 15 minutes default
                "extended": 1800,    # 30 minutes for first retry
                "max": 1800         # Never go beyond 30 minutes
            },
            "problematic_tasks": {},  # Track tasks with multiple failures
            "skip_tasks": []          # Tasks to skip after multiple failures
        }
    
    def save_config(self):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def should_skip_task(self, task_id: str) -> bool:
        """Check if a task should be skipped."""
        return task_id in self.config["skip_tasks"]
    
    def record_failure(self, task_id: str, error_type: str, duration: float):
        """Record a task failure."""
        if task_id not in self.config["problematic_tasks"]:
            self.config["problematic_tasks"][task_id] = {
                "failures": [],
                "total_attempts": 0
            }
        
        self.config["problematic_tasks"][task_id]["failures"].append({
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "duration": duration
        })
        self.config["problematic_tasks"][task_id]["total_attempts"] += 1
        
        # Auto-skip after max retries
        if self.config["problematic_tasks"][task_id]["total_attempts"] >= self.config["max_retries_per_task"]:
            if task_id not in self.config["skip_tasks"]:
                self.config["skip_tasks"].append(task_id)
                print(f"⚠️ Task {task_id} added to skip list after {self.config['max_retries_per_task']} failures")
        
        self.save_config()
    
    def get_timeout_for_task(self, task_id: str) -> int:
        """Get appropriate timeout for a task based on its history."""
        if task_id in self.config["problematic_tasks"]:
            attempts = self.config["problematic_tasks"][task_id]["total_attempts"]
            if attempts == 0:
                return self.config["timeout_limits"]["default"]
            elif attempts == 1:
                return self.config["timeout_limits"]["extended"]
            else:
                # Don't retry after 2 attempts
                return self.config["timeout_limits"]["max"]
        return self.config["timeout_limits"]["default"]
    
    def get_retry_summary(self) -> Dict:
        """Get summary of problematic tasks."""
        summary = {
            "total_problematic": len(self.config["problematic_tasks"]),
            "auto_skipped": len(self.config["skip_tasks"]),
            "timeout_patterns": {},
            "recommendations": []
        }
        
        # Analyze patterns
        for task_id, data in self.config["problematic_tasks"].items():
            for failure in data["failures"]:
                error = failure["error_type"]
                if error not in summary["timeout_patterns"]:
                    summary["timeout_patterns"][error] = []
                summary["timeout_patterns"][error].append(task_id)
        
        # Generate recommendations
        if summary["total_problematic"] > 0:
            summary["recommendations"].append(
                f"Consider skipping {summary['auto_skipped']} tasks that have failed multiple times"
            )
        
        for error_type, tasks in summary["timeout_patterns"].items():
            if "timeout" in error_type.lower() and len(tasks) > 3:
                model = tasks[0].split('_')[0] if tasks else "unknown"
                summary["recommendations"].append(
                    f"Model {model} has consistent timeout issues - consider using alternative model"
                )
        
        return summary


# Utility functions for experiment integration
def should_attempt_task(task_id: str, config: Optional[RetryConfig] = None) -> bool:
    """Check if a task should be attempted."""
    if config is None:
        config = RetryConfig()
    
    if config.should_skip_task(task_id):
        print(f"⏭️ Skipping {task_id} (marked in skip list)")
        return False
    return True

def get_dynamic_timeout(task_id: str, config: Optional[RetryConfig] = None) -> int:
    """Get dynamic timeout based on task history."""
    if config is None:
        config = RetryConfig()
    
    timeout = config.get_timeout_for_task(task_id)
    print(f"⏱️ Using {timeout}s timeout for {task_id}")
    return timeout
