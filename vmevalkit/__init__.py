"""
VMEvalKit - A framework for video generation with text+image inputs.

This framework enables video generation models to process both text prompts 
and image inputs to generate videos.
"""

__version__ = "0.1.0"

# Import runner functionality
from .runner import run_inference, InferenceRunner

# Import models
from .models import LumaInference, luma_generate

# Loader and task system removed as unused

__all__ = [
    # Runner
    "run_inference",
    "InferenceRunner",
    # Models
    "LumaInference",
    "luma_generate",
    "__version__"
]