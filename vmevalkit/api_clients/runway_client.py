"""
Runway API client implementation.

IMPORTANT: Based on Runway API documentation (https://docs.dev.runwayml.com/guides/models/),
current Runway models do NOT support the text+image→video capability required by VMEvalKit.

Available Runway models and their limitations:
- gen4_turbo: Image→Video only (no text prompt)
- gen4_aleph: Video+Text/Image→Video (requires video input)
- act_two: Image/Video→Video (no text prompt)
- veo3: Text OR Image→Video (not both simultaneously)

This implementation is provided for reference but will not work for VMEvalKit's
reasoning tasks which require BOTH text prompts AND image inputs.
"""

import os
import time
import requests
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.base import BaseVideoModel


class RunwayAPIError(Exception):
    """Custom exception for Runway API errors."""
    pass


class RunwayModel(BaseVideoModel):
    """
    Runway API model implementation.
    
    WARNING: Current Runway API models do NOT support text+image→video generation
    required for VMEvalKit reasoning tasks.
    """
    
    BASE_URL = "https://api.runwayml.com/v1"
    
    def __init__(
        self,
        model_name: str = "gen4_turbo",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Runway API client.
        
        Args:
            model_name: One of 'gen4_turbo', 'gen4_aleph', 'act_two', 'veo3'
            api_key: Runway API key (or set RUNWAY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("RUNWAY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Runway API key required. Set RUNWAY_API_KEY env var or pass api_key parameter."
            )
        
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Model-specific endpoints and capabilities
        self.model_config = self._get_model_config()
        
        super().__init__(name=f"runway_{model_name}", **kwargs)
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get configuration for specific Runway model."""
        configs = {
            "gen4_turbo": {
                "endpoint": "/image_to_video",
                "accepts_text": False,
                "accepts_image": True,
                "requires_video": False,
                "pricing": 5  # credits per second
            },
            "gen4_aleph": {
                "endpoint": "/video_to_video",
                "accepts_text": True,  # But requires video input
                "accepts_image": True,  # But requires video input
                "requires_video": True,  # PRIMARY input must be video
                "pricing": 15
            },
            "act_two": {
                "endpoint": "/character_performance",
                "accepts_text": False,
                "accepts_image": True,
                "requires_video": False,
                "pricing": 5
            },
            "veo3": {
                "endpoint": "/text_or_image_to_video",  # Hypothetical endpoint
                "accepts_text": True,  # OR operation, not AND
                "accepts_image": True,  # OR operation, not AND
                "requires_video": False,
                "pricing": 40
            }
        }
        
        if self.model_name not in configs:
            raise ValueError(
                f"Unknown Runway model: {self.model_name}. "
                f"Available: {list(configs.keys())}"
            )
        
        return configs[self.model_name]
    
    def supports_text_image_input(self) -> bool:
        """
        Check if model supports both text AND image inputs simultaneously.
        
        Returns:
            False for all current Runway models
        """
        # Based on documentation analysis:
        # - gen4_turbo: Image only
        # - gen4_aleph: Requires video as primary input
        # - act_two: Image/video only, no text
        # - veo3: Text OR image, not both
        
        if self.model_name == "gen4_turbo":
            print(f"❌ {self.model_name}: Only accepts image input, no text prompt support")
            return False
        elif self.model_name == "gen4_aleph":
            print(f"❌ {self.model_name}: Requires VIDEO as primary input (not suitable for image+text)")
            return False
        elif self.model_name == "act_two":
            print(f"❌ {self.model_name}: No text prompt support mentioned")
            return False
        elif self.model_name == "veo3":
            print(f"❌ {self.model_name}: Accepts text OR image, not both simultaneously")
            return False
        
        return False
    
    def generate(
        self,
        image: Union[str, Path],
        text_prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        resolution: tuple = (1280, 720),
        **kwargs
    ):
        """
        Attempt to generate video (will fail due to API limitations).
        
        This method is implemented for completeness but will raise an error
        because Runway models don't support the required text+image input.
        """
        raise NotImplementedError(
            f"Runway model '{self.model_name}' does not support text+image→video generation.\n"
            f"Based on official documentation:\n"
            f"- gen4_turbo: Image→Video only (no text prompt)\n"
            f"- gen4_aleph: Requires VIDEO input (not image)\n" 
            f"- act_two: Image/Video→Video (no text prompt)\n"
            f"- veo3: Text OR Image (not both)\n\n"
            f"VMEvalKit requires models that accept BOTH text prompts AND images simultaneously."
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _make_api_call(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API call to Runway (for reference only).
        
        This is implemented to show how the API would be called if it supported
        the required functionality.
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RunwayAPIError(f"Runway API request failed: {e}")
    
    def check_pricing(self, duration: float) -> float:
        """Calculate estimated cost in credits."""
        credits_per_sec = self.model_config["pricing"]
        total_credits = credits_per_sec * duration
        cost_usd = total_credits * 0.01  # 1 credit = $0.01
        return cost_usd
