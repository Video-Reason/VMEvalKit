"""
Simple inference runner for video generation.

Handles running different models with a clean interface.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json

from ..models import LumaInference


# Available models and their configurations
AVAILABLE_MODELS = {
    "luma-ray-2": {
        "class": LumaInference,
        "model": "ray-2",
        "description": "Luma Ray 2 - Latest model with best quality"
    },
    "luma-ray-flash-2": {
        "class": LumaInference,
        "model": "ray-flash-2", 
        "description": "Luma Ray Flash 2 - Faster generation"
    }
}


def run_inference(
    model_name: str,
    image_path: Union[str, Path],
    text_prompt: str,
    output_dir: str = "./outputs",
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference with specified model.
    
    Args:
        model_name: Name of model to use (e.g., "luma-ray-2", "luma-ray-flash-2")
        image_path: Path to input image
        text_prompt: Text instructions for video generation
        output_dir: Directory to save outputs
        api_key: Optional API key (uses env var if not provided)
        **kwargs: Additional model-specific parameters
        
    Returns:
        Dictionary with inference results
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    
    model_config = AVAILABLE_MODELS[model_name]
    model_class = model_config["class"]
    
    # Create model instance with specific configuration
    model = model_class(
        api_key=api_key,
        model=model_config["model"],
        output_dir=output_dir,
        **kwargs
    )
    
    # Run inference
    return model.generate(image_path, text_prompt)


class InferenceRunner:
    """
    Simple inference runner for managing video generation.
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        """
        Initialize runner.
        
        Args:
            output_dir: Directory to save generated videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Simple logging to track runs
        self.log_file = self.output_dir / "inference_log.json"
        self.runs = self._load_log()
    
    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        run_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference and log results.
        
        Args:
            model_name: Model to use
            image_path: Input image
            text_prompt: Text instructions
            run_id: Optional run identifier
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        start_time = datetime.now()
        
        # Generate run ID if not provided
        if not run_id:
            run_id = f"{model_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Run inference
            result = run_inference(
                model_name=model_name,
                image_path=image_path,
                text_prompt=text_prompt,
                output_dir=self.output_dir,
                **kwargs
            )
            
            # Add metadata
            result["run_id"] = run_id
            result["timestamp"] = start_time.isoformat()
            
            # Log the run
            self._log_run(run_id, result)
            
            return result
            
        except Exception as e:
            # Log failure
            error_result = {
                "run_id": run_id,
                "status": "failed",
                "error": str(e),
                "model": model_name,
                "timestamp": start_time.isoformat()
            }
            self._log_run(run_id, error_result)
            return error_result
    
    def list_models(self) -> Dict[str, str]:
        """List available models and their descriptions."""
        return {
            name: config["description"]
            for name, config in AVAILABLE_MODELS.items()
        }
    
    def _load_log(self) -> list:
        """Load existing run log."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def _log_run(self, run_id: str, result: Dict[str, Any]):
        """Log a run to the log file."""
        self.runs.append({
            "run_id": run_id,
            **result
        })
        
        with open(self.log_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
