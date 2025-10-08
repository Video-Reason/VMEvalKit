#!/usr/bin/env python3
"""
Example: Evaluating compatible models with VMEvalKit.

This script shows how to use models that actually support text+image→video.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vmevalkit import VMEvaluator, TaskLoader, ModelRegistry


def main():
    """Run evaluation with compatible models."""
    
    print("\n" + "=" * 70)
    print("VMEVALK IT - EVALUATING COMPATIBLE MODELS")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = VMEvaluator(output_dir="./results")
    
    # Show compatible models
    print("\n✅ COMPATIBLE MODELS (Support Text+Image→Video):")
    compatible_models = ModelRegistry.list_models(only_compatible=True)
    for model_name, info in compatible_models.items():
        print(f"   • {model_name}: {info['notes']}")
    
    print("\n❌ INCOMPATIBLE MODELS (Do NOT Support Text+Image→Video):")
    all_models = ModelRegistry.list_models(only_compatible=False)
    for model_name, info in all_models.items():
        if not info["supports_text_image"]:
            print(f"   • {model_name}: {info['notes']}")
    
    print("\n" + "-" * 70)
    
    # Example: Try to load a Runway model (will show warning)
    print("\n1. Attempting to load Runway model (incompatible):")
    try:
        runway_model = ModelRegistry.load_model("runway-gen4-turbo", api_key="dummy")
        # This would fail if we tried to use it for evaluation
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "-" * 70)
    
    # Example: Load a compatible model (Luma)
    print("\n2. Loading compatible model (Luma Dream Machine):")
    try:
        # Note: Would need actual API key to work
        luma_model = ModelRegistry.load_model(
            "luma-dream-machine",
            api_key=os.getenv("LUMA_API_KEY", "your-api-key-here")
        )
        print(f"   ✅ Model loaded: {luma_model.name}")
        print(f"   Supports text+image: {luma_model.supports_text_image_input()}")
    except Exception as e:
        print(f"   Note: {e}")
    
    print("\n" + "-" * 70)
    
    # Example evaluation workflow (demonstration only)
    print("\n3. Example Evaluation Workflow:")
    print("   Step 1: Load a reasoning task")
    task = TaskLoader.load_task("maze_solving", difficulty="medium")
    print(f"   ✓ Loaded task: {task.name} (difficulty: {task.difficulty})")
    
    print("\n   Step 2: Prepare inputs")
    print("   ✓ Problem image: maze_layout.png")
    print("   ✓ Text prompt: 'Navigate through the maze from start to end'")
    
    print("\n   Step 3: Generate solution video")
    print("   With compatible model (e.g., Luma):")
    print("     - Model receives BOTH image AND text")
    print("     - Generates video showing solution path")
    print("     - Video is evaluated for correctness")
    
    print("\n   With Runway models (would fail):")
    print("     - gen4_turbo: Can't accept text instructions")
    print("     - gen4_aleph: Needs video input (not image)")
    print("     - act_two: No text prompt support")
    print("     - veo3: Can't combine text AND image")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print("\nFor VMEvalKit reasoning tasks, use models that support text+image→video:")
    print("  ✅ Luma Dream Machine")
    print("  ✅ Pika 2.0+")
    print("  ✅ Genmo Mochi")
    print("  ✅ Google Imagen Video")
    print("  ✅ Stability AI Video")
    print("\nAvoid Runway API models which lack this capability:")
    print("  ❌ gen4_turbo, gen4_aleph, act_two, veo3")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
