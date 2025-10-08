"""
Base interface for video generation models in VMEvalKit.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
from PIL import Image


class BaseVideoModel(ABC):
    """
    Abstract base class for all video generation models.
    
    All models must accept both text prompts and image inputs to generate videos.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the video model.
        
        Args:
            name: Model identifier
            **kwargs: Model-specific configuration
        """
        self.name = name
        self.config = kwargs
        self._validate_capabilities()
    
    def _validate_capabilities(self):
        """Validate that the model supports required text+image input."""
        if not self.supports_text_image_input():
            raise ValueError(
                f"Model {self.name} does not support text+image→video generation. "
                "VMEvalKit requires models that accept BOTH text prompts and images."
            )
    
    @abstractmethod
    def supports_text_image_input(self) -> bool:
        """
        Check if model supports both text and image inputs simultaneously.
        
        Returns:
            bool: True if model accepts text+image, False otherwise
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        text_prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        resolution: tuple = (1280, 720),
        **kwargs
    ) -> Union[str, np.ndarray]:
        """
        Generate a video from text prompt and image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            text_prompt: Text instructions for video generation
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Output resolution (width, height)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Path to generated video file or video array
        """
        pass
    
    def preprocess_image(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        Preprocess input image to required format.
        
        Args:
            image: Input image in various formats
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information and capabilities.
        
        Returns:
            Dictionary containing model details
        """
        return {
            "name": self.name,
            "supports_text_image": self.supports_text_image_input(),
            "config": self.config
        }
