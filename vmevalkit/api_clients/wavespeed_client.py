"""
WaveSpeed AI (Wan 2.2) API client implementation.

Based on https://wavespeed.ai/collections/wan-2-2, Wan 2.2 offers:
- Text-to-video (t2v) models
- Image-to-video (i2v) models
- Potentially combined text+image capabilities

This needs verification with actual API testing.
"""

import os
import time
import base64
import requests
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
from io import BytesIO
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.base import BaseVideoModel


class WaveSpeedAPIError(Exception):
    """Custom exception for WaveSpeed API errors."""
    pass


class WaveSpeedWan22(BaseVideoModel):
    """
    WaveSpeed Wan 2.2 API implementation.
    
    This model family includes both i2v and t2v capabilities.
    We need to verify if i2v models also accept text prompts for guidance.
    """
    
    BASE_URL = "https://api.wavespeed.ai/v1"  # Hypothetical - needs verification
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_variant: str = "i2v-480p",
        **kwargs
    ):
        """
        Initialize WaveSpeed Wan 2.2 client.
        
        Args:
            api_key: WaveSpeed API key (or set WAVESPEED_API_KEY env var)
            model_variant: Which Wan 2.2 model to use (i2v-480p, t2v-720p, etc.)
        """
        self.api_key = api_key or os.getenv("WAVESPEED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "WaveSpeed API key required. Set WAVESPEED_API_KEY env var or pass api_key parameter."
            )
        
        self.model_variant = model_variant
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        super().__init__(name=f"wavespeed_wan22_{model_variant}", **kwargs)
    
    def supports_text_image_input(self) -> bool:
        """
        Check if model supports both text and image inputs.
        
        Based on the website, they have separate i2v and t2v models.
        The question is: do i2v models also accept text prompts?
        
        Returns:
            Unknown - needs testing with actual API
        """
        # This needs to be verified with actual API testing
        print("⚠️  WaveSpeed Wan 2.2 capability needs verification")
        print("   The model has i2v and t2v modes, but combined support is unclear")
        print("   Testing required to confirm if i2v accepts text guidance")
        
        # For now, return True to allow testing
        return True  # Optimistic - needs verification
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        text_prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        resolution: tuple = (854, 480),  # 480p default
        **kwargs
    ) -> str:
        """
        Generate video using WaveSpeed Wan 2.2.
        
        This implementation needs to be tested with actual API to determine:
        1. Correct endpoint structure
        2. Whether i2v models accept text prompts
        3. Parameter names and formats
        """
        # Preprocess image
        pil_image = self.preprocess_image(image)
        
        # Encode image to base64
        image_base64 = self._encode_image(pil_image)
        
        # Determine endpoint based on model variant
        if "i2v" in self.model_variant:
            endpoint = f"/wan-2.2/{self.model_variant}"
            
            # Try different payload structures to see what works
            payloads_to_test = [
                # Option 1: Both image and prompt
                {
                    "image": image_base64,
                    "prompt": text_prompt,
                    "duration": duration,
                    "resolution": f"{resolution[0]}x{resolution[1]}"
                },
                # Option 2: Image with text guidance
                {
                    "input_image": image_base64,
                    "text_guidance": text_prompt,
                    "duration": duration
                },
                # Option 3: Image with description
                {
                    "image": image_base64,
                    "description": text_prompt,
                    "duration": duration
                }
            ]
            
            # This would need actual testing
            print(f"Would test WaveSpeed {self.model_variant} with various payload structures")
            
        elif "t2v" in self.model_variant:
            # Text-to-video endpoint
            endpoint = f"/wan-2.2/{self.model_variant}"
            # Can t2v also accept an optional reference image?
            payload = {
                "prompt": text_prompt,
                "reference_image": image_base64,  # Test if this is accepted
                "duration": duration
            }
        
        # Placeholder for actual API call
        raise NotImplementedError(
            f"WaveSpeed Wan 2.2 implementation needs testing with actual API.\n"
            f"Model variant: {self.model_variant}\n"
            f"Capabilities to verify:\n"
            f"- Do i2v models accept text prompts for guidance?\n"
            f"- What are the actual parameter names?\n"
            f"- What is the correct API endpoint structure?"
        )
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def test_capabilities(self) -> Dict[str, Any]:
        """
        Test what inputs this model actually accepts.
        
        Returns:
            Dictionary with test results
        """
        test_results = {
            "model": self.model_variant,
            "tests_to_run": []
        }
        
        if "i2v" in self.model_variant:
            test_results["tests_to_run"] = [
                "Test 1: Image only (baseline)",
                "Test 2: Image + prompt parameter",
                "Test 3: Image + text_guidance parameter",
                "Test 4: Image + description parameter",
                "Test 5: Image + caption parameter"
            ]
        elif "t2v" in self.model_variant:
            test_results["tests_to_run"] = [
                "Test 1: Text only (baseline)",
                "Test 2: Text + reference_image parameter",
                "Test 3: Text + input_image parameter",
                "Test 4: Text + style_reference parameter"
            ]
        
        return test_results
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "api": "WaveSpeed Wan 2.2",
            "model_variant": self.model_variant,
            "status": "Needs verification",
            "capabilities": {
                "i2v_models": [
                    "wan-2.2/i2v-480p",
                    "wan-2.2/i2v-720p",
                    "wan-2.2/i2v-1080p"
                ],
                "t2v_models": [
                    "wan-2.2/t2v-480p",
                    "wan-2.2/t2v-720p", 
                    "wan-2.2/t2v-1080p"
                ],
                "special_models": [
                    "wan-2.2/video-edit (text-based video editing)",
                    "wan-2.2/fun-control (multi-modal conditional inputs)",
                    "wan-2.2/animate (character animation)"
                ],
                "text_image_support": "Unknown - needs testing"
            },
            "pricing": {
                "480p": "$0.15-0.20 per 5 seconds",
                "720p": "$0.30-0.40 per 5 seconds",
                "1080p": "$0.80 per 5 seconds"
            }
        }
