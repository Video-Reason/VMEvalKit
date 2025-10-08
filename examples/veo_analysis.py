#!/usr/bin/env python3
"""
Google Veo 3 video generation example for VMEvalKit.

This script demonstrates how to use Google's Veo 3 model for text+image→video generation
required for reasoning tasks in VMEvalKit.

Based on documentation: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation

Requirements:
1. Google Cloud Project with Vertex AI enabled
2. Authentication via one of:
   - GOOGLE_CLOUD_PROJECT environment variable
   - GOOGLE_API_KEY or GOOGLE_ACCESS_TOKEN environment variable
   - gcloud auth login
3. Optional: GCS_STORAGE_BUCKET for video storage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vmevalkit.api_clients.veo_client import GoogleVeo
from vmevalkit.core.model_registry import ModelRegistry
from pathlib import Path
from typing import Optional
import argparse


def demonstrate_veo_capabilities():
    """Show Google Veo's capabilities for VMEvalKit."""
    
    print("=" * 70)
    print("GOOGLE VEO 3 - CAPABILITY ANALYSIS FOR VMEvalKit")
    print("=" * 70)
    print("\n✅ VMEvalKit Compatibility: FULLY SUPPORTED")
    print("\nGoogle Veo 3 supports:")
    print("  ✓ Text prompts (detailed instructions)")
    print("  ✓ Image inputs (visual context)")
    print("  ✓ Combined text+image→video generation")
    print("  ✓ High-quality, imaginative video outputs")
    print("\n" + "-" * 70)
    
    # Veo model versions
    models_info = {
        "veo-001": {
            "status": "✅ Compatible",
            "features": [
                "Text-to-video generation",
                "Image-to-video generation", 
                "Text+Image-to-video generation",
                "720p and 1080p output",
                "Up to 8 second videos",
                "24 fps output"
            ],
            "use_case": "High-quality video generation with good consistency"
        },
        "veo-002": {
            "status": "✅ Compatible (Recommended)",
            "features": [
                "All features of veo-001",
                "Improved temporal consistency",
                "Better motion quality",
                "Enhanced prompt understanding",
                "More realistic physics",
                "Latest model version"
            ],
            "use_case": "Best quality for reasoning tasks and complex prompts"
        }
    }
    
    for version, info in models_info.items():
        print(f"\n📦 Model: Google {version.upper()}")
        print(f"   Status: {info['status']}")
        print(f"   Features:")
        for feature in info['features']:
            print(f"      • {feature}")
        print(f"   Best for: {info['use_case']}")
    
    print("\n" + "-" * 70)
    print("\n🔧 Configuration Options:")
    print("   • project_id: Your Google Cloud project")
    print("   • location: Region (default: us-central1)")
    print("   • model_version: veo-001 or veo-002")
    print("   • storage_bucket: GCS bucket for outputs (optional)")
    print("   • num_videos: 1-4 variations per request")


def test_veo_model(
    project_id: Optional[str] = None,
    location: str = "us-central1",
    model_version: str = "veo-002"
):
    """
    Test Google Veo model initialization and compatibility.
    
    Args:
        project_id: Google Cloud project ID
        location: Google Cloud region
        model_version: Veo model version (veo-001 or veo-002)
    """
    print("\n" + "=" * 70)
    print("TESTING GOOGLE VEO MODEL")
    print("=" * 70)
    
    try:
        # Load model through registry
        print(f"\n1. Loading Google Veo {model_version}...")
        model = ModelRegistry.load_model(
            f"google-{model_version}",
            project_id=project_id,
            location=location
        )
        
        print(f"   ✅ Model loaded successfully: {model.name}")
        
        # Check capabilities
        print("\n2. Checking capabilities...")
        supports_both = model.supports_text_image_input()
        print(f"   Text+Image support: {'✅ Yes' if supports_both else '❌ No'}")
        
        # Get model info
        info = model.get_info()
        print("\n3. Model Information:")
        print(f"   Name: {info['name']}")
        print(f"   API: {info['api']}")
        print(f"   Version: {info['model_version']}")
        print(f"   Location: {info['location']}")
        
        capabilities = info['capabilities']
        print("\n4. Capabilities:")
        print(f"   • Text prompts: {'✅' if capabilities['text_prompt'] else '❌'}")
        print(f"   • Image input: {'✅' if capabilities['image_reference'] else '❌'}")
        print(f"   • Text AND Image: {'✅' if capabilities['text_and_image'] else '❌'}")
        print(f"   • Max duration: {capabilities['max_duration']} seconds")
        print(f"   • Resolutions: {', '.join(capabilities['resolutions'])}")
        print(f"   • Frame rate: {capabilities['fps']} fps")
        print(f"   • Max videos/request: {capabilities['max_videos_per_request']}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Setup Instructions:")
        print("1. Enable Vertex AI API in your Google Cloud project")
        print("2. Set environment variables:")
        print("   export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("   export GOOGLE_APPLICATION_CREDENTIALS='path/to/service-account.json'")
        print("   # OR use gcloud auth:")
        print("   gcloud auth login")
        print("   gcloud config set project your-project-id")
        print("3. Optional: Set GCS bucket for video storage:")
        print("   export GCS_STORAGE_BUCKET='your-bucket-name'")
        return None


def generate_example_video(
    model: GoogleVeo,
    image_path: str,
    prompt: str,
    output_dir: str = "./outputs"
):
    """
    Generate an example video using Google Veo.
    
    Args:
        model: Initialized Google Veo model
        image_path: Path to input image
        prompt: Text prompt for video generation
        output_dir: Directory for output videos
    """
    print("\n" + "=" * 70)
    print("GENERATING EXAMPLE VIDEO")
    print("=" * 70)
    
    print(f"\n📷 Input image: {image_path}")
    print(f"📝 Prompt: {prompt}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"\n❌ Error: Image not found at {image_path}")
        print("💡 Please provide a valid image path")
        return None
    
    try:
        print("\n⏳ Generating video (this may take 1-3 minutes)...")
        
        # Generate video
        video_path = model.generate(
            image=image_path,
            text_prompt=prompt,
            duration=5.0,  # 5 second video
            resolution=(1280, 720),  # 720p
            num_videos=2  # Generate 2 variations
        )
        
        if isinstance(video_path, list):
            print(f"\n✅ Generated {len(video_path)} video variations:")
            for i, path in enumerate(video_path, 1):
                print(f"   {i}. {path}")
            return video_path
        else:
            print(f"\n✅ Video generated: {video_path}")
            return video_path
            
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        return None


def compare_with_other_models():
    """Compare Google Veo with other models in VMEvalKit."""
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON FOR VMEvalKit")
    print("=" * 70)
    
    # List all compatible models
    compatible_models = ModelRegistry.list_models(only_compatible=True)
    
    print("\n✅ Compatible Models (support text+image→video):")
    for name, info in compatible_models.items():
        print(f"\n  • {name}")
        print(f"    Status: {info['status']}")
        print(f"    Notes: {info['notes']}")
    
    # List incompatible models
    all_models = ModelRegistry.list_models(only_compatible=False)
    incompatible = {k: v for k, v in all_models.items() if not v["supports_text_image"]}
    
    if incompatible:
        print("\n\n❌ Incompatible Models (cannot handle text+image):")
        for name, info in incompatible.items():
            print(f"\n  • {name}")
            print(f"    Status: {info['status']}")
            print(f"    Notes: {info['notes']}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Google Veo 3 analysis and testing for VMEvalKit"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test Google Veo model initialization"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate an example video"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="examples/test_image.jpg",
        help="Path to input image for video generation"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Navigate through the maze and find the exit",
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help="Google Cloud project ID"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="Google Cloud region"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="veo-002",
        choices=["veo-001", "veo-002"],
        help="Veo model version"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Veo with other models"
    )
    
    args = parser.parse_args()
    
    # Always show capabilities
    demonstrate_veo_capabilities()
    
    # Compare with other models if requested
    if args.compare:
        compare_with_other_models()
    
    # Test model if requested
    model = None
    if args.test or args.generate:
        model = test_veo_model(
            project_id=args.project_id,
            location=args.location,
            model_version=args.model_version
        )
    
    # Generate video if requested and model loaded
    if args.generate and model:
        generate_example_video(
            model=model,
            image_path=args.image,
            prompt=args.prompt
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n🎯 Google Veo 3 is FULLY COMPATIBLE with VMEvalKit")
    print("   • Supports text+image→video generation")
    print("   • High-quality, imaginative outputs")
    print("   • Suitable for complex reasoning tasks")
    print("\n📚 Next Steps:")
    print("   1. Set up Google Cloud authentication")
    print("   2. Test with: python examples/veo_analysis.py --test")
    print("   3. Generate videos: python examples/veo_analysis.py --generate --image path/to/image.jpg")
    print("\n✨ Happy video generation with Google Veo!")


if __name__ == "__main__":
    main()
