"""
Google Veo 3 video generation API client implementation.

Veo 3 supports both text and image inputs for video generation through Google Cloud Vertex AI,
making it suitable for VMEvalKit's reasoning tasks.

Documentation: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation
"""

import os
import time
import base64
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..models.base import BaseVideoModel


class VeoAPIError(Exception):
    """Custom exception for Google Veo API errors."""
    pass


class GoogleVeo(BaseVideoModel):
    """
    Google Veo 3 video generation model via Vertex AI.
    
    This model supports text+image→video generation required for reasoning tasks.
    Veo creates imaginative, high-quality videos from text and/or image prompts.
    """
    
    # Available models and their capabilities
    VEO_MODELS = {
        "veo-001": {
            "endpoint": "veo-001",
            "supports_text": True,
            "supports_image": True,
            "supports_both": True,  # Can handle text+image simultaneously
            "max_duration": 8,  # seconds
            "resolutions": ["720p", "1080p"],
            "fps": 24
        },
        "veo-002": {
            "endpoint": "veo-002",
            "supports_text": True,
            "supports_image": True,
            "supports_both": True,
            "max_duration": 8,
            "resolutions": ["720p", "1080p"],
            "fps": 24
        }
    }
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model_version: str = "veo-002",  # Default to latest version
        api_key: Optional[str] = None,
        storage_bucket: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Google Veo client.
        
        Args:
            project_id: Google Cloud project ID (or set GOOGLE_CLOUD_PROJECT env var)
            location: Google Cloud region (default: us-central1)
            model_version: Veo model version (veo-001 or veo-002)
            api_key: Google Cloud API key or access token (or use gcloud auth)
            storage_bucket: GCS bucket for storing videos (optional)
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError(
                "Google Cloud project ID required. Set GOOGLE_CLOUD_PROJECT env var or pass project_id parameter."
            )
        
        self.location = location
        self.model_version = model_version
        
        if model_version not in self.VEO_MODELS:
            raise ValueError(
                f"Unknown Veo model: {model_version}. Available: {list(self.VEO_MODELS.keys())}"
            )
        
        self.model_config = self.VEO_MODELS[model_version]
        
        # Try to get auth token - support multiple methods
        self.access_token = self._get_access_token(api_key)
        
        # Base URL for Vertex AI
        self.base_url = f"https://{location}-aiplatform.googleapis.com/v1"
        self.model_path = f"projects/{project_id}/locations/{location}/publishers/google/models/{self.model_config['endpoint']}"
        
        # Optional GCS bucket for video storage
        self.storage_bucket = storage_bucket or os.getenv("GCS_STORAGE_BUCKET")
        
        # Request headers
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        super().__init__(name=f"google_veo_{model_version}", **kwargs)
    
    def _get_access_token(self, api_key: Optional[str] = None) -> str:
        """
        Get Google Cloud access token.
        
        Tries multiple methods:
        1. Provided API key/token
        2. Environment variable GOOGLE_API_KEY
        3. gcloud auth token (if available)
        """
        # Try provided API key
        if api_key:
            return api_key
        
        # Try environment variable
        env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_ACCESS_TOKEN")
        if env_key:
            return env_key
        
        # Try gcloud auth
        try:
            import subprocess
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True
            )
            token = result.stdout.strip()
            if token:
                return token
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        raise ValueError(
            "Google Cloud authentication required. Please provide one of:\n"
            "1. api_key parameter\n"
            "2. GOOGLE_API_KEY or GOOGLE_ACCESS_TOKEN env var\n"
            "3. Authenticate via 'gcloud auth login'"
        )
    
    def supports_text_image_input(self) -> bool:
        """
        Veo supports both text and image inputs simultaneously.
        
        Returns:
            True - Veo accepts text prompts with image references
        """
        return self.model_config["supports_both"]
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        text_prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        resolution: tuple = (1280, 720),
        num_videos: int = 4,
        **kwargs
    ) -> str:
        """
        Generate video from text prompt and image using Google Veo.
        
        Args:
            image: Input image for reference
            text_prompt: Text instructions for video generation
            duration: Video duration in seconds (max 8 seconds)
            fps: Frames per second (Veo uses 24 fps)
            resolution: Output resolution
            num_videos: Number of video variations to generate (1-4)
            **kwargs: Additional parameters
            
        Returns:
            Path to first generated video file (or list if num_videos > 1)
        """
        # Validate duration
        max_duration = self.model_config["max_duration"]
        if duration > max_duration:
            print(f"Warning: Duration {duration}s exceeds max {max_duration}s. Capping to {max_duration}s.")
            duration = max_duration
        
        # Validate number of videos
        if num_videos < 1 or num_videos > 4:
            raise ValueError("num_videos must be between 1 and 4")
        
        # Preprocess image
        pil_image = self.preprocess_image(image)
        
        # Encode image to base64
        image_base64 = self._encode_image(pil_image)
        
        # Start generation
        operation_name = self._start_generation(
            image_base64=image_base64,
            prompt=text_prompt,
            duration=duration,
            resolution=resolution,
            num_videos=num_videos
        )
        
        # Poll for completion
        video_urls = self._poll_for_completion(operation_name)
        
        # Download videos
        video_paths = []
        for i, video_url in enumerate(video_urls):
            video_path = self._download_video(video_url, f"{operation_name.split('/')[-1]}_{i}")
            video_paths.append(video_path)
        
        # Return first video or all if requested
        if num_videos == 1:
            return video_paths[0]
        else:
            return video_paths
    
    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffered = BytesIO()
        # Veo prefers JPEG for efficiency
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _start_generation(
        self,
        image_base64: str,
        prompt: str,
        duration: float,
        resolution: tuple,
        num_videos: int = 4
    ) -> str:
        """
        Start video generation job.
        
        Returns:
            Operation name for polling
        """
        endpoint = f"{self.base_url}/{self.model_path}:predict"
        
        # Map resolution to Veo's format
        width, height = resolution
        if height >= 1080:
            resolution_str = "1080p"
        else:
            resolution_str = "720p"
        
        # Build request payload
        payload = {
            "instances": [{
                "prompt": prompt,
                "image": {
                    "bytesBase64Encoded": image_base64
                }
            }],
            "parameters": {
                "sampleCount": num_videos,
                "videoDuration": f"{int(duration)}s",
                "resolution": resolution_str,
                "fps": self.model_config["fps"]
            }
        }
        
        # Add storage URI if configured
        if self.storage_bucket:
            # Create timestamped folder
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_uri = f"gs://{self.storage_bucket}/veo_outputs/{timestamp}/"
            payload["parameters"]["storageUri"] = storage_uri
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if "name" not in data:
                raise VeoAPIError(f"No operation name in response: {data}")
            
            return data["name"]
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to start Veo generation: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg += f"\nAPI Error: {error_data}"
                except:
                    error_msg += f"\nResponse: {e.response.text}"
            raise VeoAPIError(error_msg)
    
    def _poll_for_completion(
        self,
        operation_name: str,
        max_wait: int = 600,
        poll_interval: int = 10
    ) -> list:
        """
        Poll for generation completion using fetchPredictOperation.
        
        Returns:
            List of URLs or GCS URIs of generated videos
        """
        # Use fetchPredictOperation endpoint
        endpoint = f"{self.base_url}/{self.model_path}:fetchPredictOperation"
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                # Request body for fetching operation status
                payload = {
                    "operationName": operation_name
                }
                
                response = requests.post(endpoint, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Check if operation is done
                if data.get("done", False):
                    # Check for errors
                    if "error" in data:
                        raise VeoAPIError(f"Generation failed: {data['error']}")
                    
                    # Extract video URLs
                    response_data = data.get("response", {})
                    videos = response_data.get("videos", [])
                    
                    if not videos:
                        # Check if videos were filtered
                        filtered_count = response_data.get("raiMediaFilteredCount", 0)
                        if filtered_count > 0:
                            reasons = response_data.get("raiMediaFilteredReasons", [])
                            raise VeoAPIError(
                                f"All {filtered_count} videos were filtered by safety policies. "
                                f"Reasons: {reasons}"
                            )
                        raise VeoAPIError("No videos generated")
                    
                    # Extract URLs or GCS URIs
                    video_urls = []
                    for video in videos:
                        if "gcsUri" in video:
                            video_urls.append(video["gcsUri"])
                        elif "bytesBase64Encoded" in video:
                            # Store base64 for later processing
                            video_urls.append({"base64": video["bytesBase64Encoded"]})
                        else:
                            raise VeoAPIError(f"Unknown video format: {video}")
                    
                    return video_urls
                
                # Still processing
                print(f"Generation in progress... ({int(time.time() - start_time)}s elapsed)")
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Failed to check operation status: {e}"
                if hasattr(e, 'response') and e.response is not None:
                    error_msg += f"\nResponse: {e.response.text}"
                raise VeoAPIError(error_msg)
        
        raise VeoAPIError(f"Generation timed out after {max_wait} seconds")
    
    def _download_video(self, video_source: Union[str, Dict], video_id: str) -> str:
        """
        Download or save generated video.
        
        Args:
            video_source: GCS URI, URL, or dict with base64 data
            video_id: Identifier for the video file
            
        Returns:
            Path to saved video file
        """
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        
        video_path = output_dir / f"veo_{video_id}.mp4"
        
        if isinstance(video_source, dict) and "base64" in video_source:
            # Decode base64 video
            video_data = base64.b64decode(video_source["base64"])
            with open(video_path, 'wb') as f:
                f.write(video_data)
            
        elif video_source.startswith("gs://"):
            # Download from GCS
            # For production, use gsutil or google-cloud-storage library
            # Here we'll use a simple approach
            try:
                import subprocess
                subprocess.run(
                    ["gsutil", "cp", video_source, str(video_path)],
                    check=True,
                    capture_output=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise VeoAPIError(
                    f"Failed to download from GCS: {video_source}\n"
                    "Please install gsutil or use google-cloud-storage library"
                )
        
        elif video_source.startswith("http"):
            # Download from URL
            try:
                response = requests.get(video_source, stream=True)
                response.raise_for_status()
                
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                raise VeoAPIError(f"Failed to download video: {e}")
        
        else:
            raise VeoAPIError(f"Unknown video source format: {video_source}")
        
        return str(video_path)
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.name,
            "supports_text_image": True,
            "api": "Google Veo (Vertex AI)",
            "model_version": self.model_version,
            "project_id": self.project_id,
            "location": self.location,
            "capabilities": {
                "text_prompt": True,
                "image_reference": True,
                "text_and_image": True,
                "max_duration": self.model_config["max_duration"],
                "resolutions": self.model_config["resolutions"],
                "fps": self.model_config["fps"],
                "max_videos_per_request": 4
            }
        }
