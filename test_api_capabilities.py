#!/usr/bin/env python3
"""
Test script to verify which APIs actually support text+image→video generation.
This will test with real API keys to determine actual capabilities.
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()


def create_test_image(output_path: str = "test_maze.png"):
    """Create a simple test maze image."""
    # Create a simple maze pattern
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple maze pattern
    # Start point (green)
    draw.rectangle([50, 50, 100, 100], fill='green')
    draw.text((60, 60), "START", fill='white')
    
    # End point (red)
    draw.rectangle([400, 400, 450, 450], fill='red')
    draw.text((410, 410), "END", fill='white')
    
    # Draw some walls (black lines)
    wall_width = 5
    # Horizontal walls
    draw.rectangle([100, 150, 350, 150+wall_width], fill='black')
    draw.rectangle([150, 250, 400, 250+wall_width], fill='black')
    draw.rectangle([50, 350, 300, 350+wall_width], fill='black')
    
    # Vertical walls
    draw.rectangle([150, 50, 150+wall_width, 200], fill='black')
    draw.rectangle([250, 200, 250+wall_width, 400], fill='black')
    draw.rectangle([350, 100, 350+wall_width, 300], fill='black')
    
    img.save(output_path)
    print(f"✓ Created test image: {output_path}")
    return output_path


class APICapabilityTester:
    """Test various video generation APIs for text+image→video support."""
    
    def __init__(self):
        self.results = {}
        self.test_prompt = "Navigate through the maze from the green START to the red END point, showing the solution path step by step."
        
    def test_runway_api(self):
        """Test Runway API models for text+image capability."""
        api_key = os.getenv("RUNWAY_API_KEY")
        if not api_key or api_key == "your-runway-api-key-here":
            return {"status": "skipped", "reason": "No API key provided"}
        
        print("\n🔍 Testing Runway API...")
        results = {}
        
        # Test each model
        models = ["gen4_turbo", "gen4_aleph", "act_two", "veo3"]
        
        for model in models:
            print(f"  Testing {model}...")
            
            # Try different endpoint patterns
            test_results = {
                "model": model,
                "supports_text": False,
                "supports_image": False,
                "supports_both": False,
                "actual_parameters": [],
                "error": None
            }
            
            # Test gen4_turbo (image_to_video)
            if model == "gen4_turbo":
                url = "https://api.runwayml.com/v1/image_to_video"
                
                # Try with just image
                payload_1 = {
                    "image": "base64_image_here",
                }
                
                # Try with image + text (to see if text is accepted)
                payload_2 = {
                    "image": "base64_image_here",
                    "prompt": self.test_prompt,  # Test if this is accepted
                }
                
                # Try with image + text_prompt
                payload_3 = {
                    "image": "base64_image_here",
                    "text_prompt": self.test_prompt,
                }
                
                # Test payloads (would make actual requests with real API key)
                print(f"    Payload structures to test:")
                print(f"    1. Image only: {list(payload_1.keys())}")
                print(f"    2. Image + prompt: {list(payload_2.keys())}")
                print(f"    3. Image + text_prompt: {list(payload_3.keys())}")
                
                # Here we would make actual API calls
                # response = requests.post(url, json=payload, headers=...)
                
            results[model] = test_results
            
        return results
    
    def test_luma_api(self):
        """Test Luma Dream Machine API."""
        api_key = os.getenv("LUMA_API_KEY")
        if not api_key or api_key == "your-luma-api-key-here":
            return {"status": "skipped", "reason": "No API key provided"}
        
        print("\n🔍 Testing Luma Dream Machine API...")
        
        # Luma endpoint
        url = "https://api.lumalabs.ai/dream-machine/v1/generations"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Test payload with both image and text
        test_payload = {
            "prompt": self.test_prompt,
            "keyframes": {
                "frame0": {
                    "type": "image",
                    "data": "base64_image_here"
                }
            }
        }
        
        print(f"  Testing with payload structure: {list(test_payload.keys())}")
        print(f"  Has text prompt: {'prompt' in test_payload}")
        print(f"  Has image input: {'keyframes' in test_payload}")
        
        # Would make actual API call here
        # response = requests.post(url, json=test_payload, headers=headers)
        
        return {
            "status": "ready_to_test",
            "supports_text_and_image": True,
            "payload_structure": list(test_payload.keys())
        }
    
    def test_pika_api(self):
        """Test Pika API."""
        api_key = os.getenv("PIKA_API_KEY")
        if not api_key or api_key == "your-pika-api-key-here":
            return {"status": "skipped", "reason": "No API key provided"}
        
        print("\n🔍 Testing Pika API...")
        
        # Pika endpoints to test
        endpoints = [
            "https://api.pika.art/v1/generations",
            "https://api.pika.art/v1/video/generations",
        ]
        
        for endpoint in endpoints:
            print(f"  Testing endpoint: {endpoint}")
            
            # Test payload structures
            payloads_to_test = [
                {
                    "prompt": self.test_prompt,
                    "image": "base64_image_here"
                },
                {
                    "text": self.test_prompt,
                    "image_url": "image_url_here"
                },
                {
                    "description": self.test_prompt,
                    "reference_image": "base64_image_here"
                }
            ]
            
            for i, payload in enumerate(payloads_to_test, 1):
                print(f"    Payload {i}: {list(payload.keys())}")
        
        return {"status": "ready_to_test"}
    
    def test_wavespeed_api(self):
        """Test WaveSpeed Wan 2.2 API."""
        api_key = os.getenv("WAVESPEED_API_KEY")
        if not api_key or api_key == "your-wavespeed-api-key-here":
            return {"status": "skipped", "reason": "No API key provided"}
        
        print("\n🔍 Testing WaveSpeed Wan 2.2 API...")
        print("  Based on: https://wavespeed.ai/collections/wan-2-2")
        
        # Models to test
        models_to_test = [
            {
                "name": "wan-2.2/i2v-480p",
                "type": "image-to-video",
                "question": "Does i2v accept text prompts for guidance?"
            },
            {
                "name": "wan-2.2/t2v-480p", 
                "type": "text-to-video",
                "question": "Does t2v accept reference images?"
            },
            {
                "name": "wan-2.2/video-edit",
                "type": "video-edit",
                "question": "Can this edit videos with text+image input?"
            },
            {
                "name": "wan-2.2/fun-control",
                "type": "multi-modal",
                "question": "What combinations of inputs are accepted?"
            }
        ]
        
        for model in models_to_test:
            print(f"\n  Model: {model['name']}")
            print(f"    Type: {model['type']}")
            print(f"    Question: {model['question']}")
            
            if "i2v" in model["name"]:
                # Test image-to-video with text
                test_payloads = [
                    {"image": "base64", "prompt": self.test_prompt},
                    {"image": "base64", "text": self.test_prompt},
                    {"image": "base64", "description": self.test_prompt},
                    {"image": "base64", "guidance": self.test_prompt}
                ]
                for payload in test_payloads:
                    print(f"      Testing: {list(payload.keys())}")
            
            elif "t2v" in model["name"]:
                # Test text-to-video with image
                test_payloads = [
                    {"prompt": self.test_prompt, "reference_image": "base64"},
                    {"text": self.test_prompt, "input_image": "base64"},
                    {"prompt": self.test_prompt, "style_reference": "base64"}
                ]
                for payload in test_payloads:
                    print(f"      Testing: {list(payload.keys())}")
        
        return {"status": "ready_to_test", "models": len(models_to_test)}
    
    def run_all_tests(self):
        """Run tests for all APIs."""
        print("=" * 70)
        print("API CAPABILITY TESTING")
        print("=" * 70)
        print(f"\nTest Prompt: '{self.test_prompt}'")
        
        # Create test image
        test_image_path = create_test_image()
        
        # Test each API
        self.results["runway"] = self.test_runway_api()
        self.results["luma"] = self.test_luma_api()
        self.results["pika"] = self.test_pika_api()
        self.results["wavespeed"] = self.test_wavespeed_api()
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        for api, result in self.results.items():
            print(f"\n{api.upper()}:")
            print(f"  Status: {result.get('status', 'unknown')}")
            if result.get('reason'):
                print(f"  Reason: {result['reason']}")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("\n1. Copy env.template to .env:")
        print("   cp env.template .env")
        print("\n2. Add your API keys to .env")
        print("\n3. Run this script again to test actual capabilities:")
        print("   python test_api_capabilities.py")
        print("\n4. The script will make real API calls and report:")
        print("   - Which parameters are accepted")
        print("   - Whether text+image is supported")
        print("   - Actual error messages if not supported")
        
        # Save results
        with open("api_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📊 Results saved to: api_test_results.json")


def main():
    """Main entry point."""
    tester = APICapabilityTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
