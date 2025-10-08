"""
Core evaluation engine for VMEvalKit.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import time
from datetime import datetime
from dataclasses import dataclass

from ..models.base import BaseVideoModel


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    reasoning_score: float
    video_quality: float
    solution_accuracy: float
    task_name: str
    model_name: str
    duration: float
    metadata: Dict[str, Any]


class VMEvaluator:
    """
    Main evaluation orchestrator for video reasoning models.
    
    Evaluates models that support text+image→video generation on reasoning tasks.
    """
    
    def __init__(self, output_dir: str = "./results"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = []
    
    def evaluate(
        self,
        model: BaseVideoModel,
        task: Any,  # Task object from TaskLoader
        input_image: Union[str, Path],
        text_prompt: str,
        num_samples: int = 1,
        verify_text_image_support: bool = True,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a model on a single task.
        
        Args:
            model: Video generation model
            task: Task instance with problem and evaluation logic
            input_image: Path to problem image
            text_prompt: Instructions for solving the task
            num_samples: Number of video samples to generate
            verify_text_image_support: Check model capabilities
            **kwargs: Additional parameters
            
        Returns:
            Evaluation results
        """
        # Verify model capabilities
        if verify_text_image_support and not model.supports_text_image_input():
            raise ValueError(
                f"Model {model.name} does not support text+image→video generation. "
                f"This capability is required for VMEvalKit reasoning tasks."
            )
        
        print(f"\n🔍 Evaluating {model.name} on {task.name}")
        print(f"   Input: {input_image}")
        print(f"   Prompt: {text_prompt}")
        
        start_time = time.time()
        
        # Generate videos
        generated_videos = []
        for i in range(num_samples):
            print(f"   Generating sample {i+1}/{num_samples}...")
            try:
                video_path = model.generate(
                    image=input_image,
                    text_prompt=text_prompt,
                    **kwargs
                )
                generated_videos.append(video_path)
            except Exception as e:
                print(f"   ⚠️ Generation failed: {e}")
                continue
        
        # Evaluate generated videos
        if generated_videos:
            scores = self._evaluate_videos(generated_videos, task)
        else:
            # No successful generations
            scores = {
                "reasoning_score": 0.0,
                "video_quality": 0.0,
                "solution_accuracy": 0.0
            }
        
        duration = time.time() - start_time
        
        # Create result
        result = EvaluationResult(
            reasoning_score=scores["reasoning_score"],
            video_quality=scores["video_quality"],
            solution_accuracy=scores["solution_accuracy"],
            task_name=task.name,
            model_name=model.name,
            duration=duration,
            metadata={
                "num_samples": num_samples,
                "successful_generations": len(generated_videos),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Save result
        self._save_result(result)
        self.results.append(result)
        
        print(f"\n✅ Evaluation complete:")
        print(f"   Reasoning Score: {result.reasoning_score:.2f}")
        print(f"   Video Quality: {result.video_quality:.2f}")
        print(f"   Solution Accuracy: {result.solution_accuracy:.2f}")
        print(f"   Time: {duration:.2f}s")
        
        return result
    
    def run_benchmark(
        self,
        models: List[str],
        tasks: List[str],
        api_keys: Optional[Dict[str, str]] = None,
        output_dir: str = "./results",
        generate_report: bool = True,
        strict_mode: bool = True,
        test_inputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across multiple models and tasks.
        
        Args:
            models: List of model names to evaluate
            tasks: List of task names to evaluate
            api_keys: API keys for each model
            output_dir: Output directory for results
            generate_report: Whether to generate HTML report
            strict_mode: Fail if model doesn't support text+image
            test_inputs: Test inputs for verification
            **kwargs: Additional parameters
            
        Returns:
            Benchmark results dictionary
        """
        print("\n" + "=" * 70)
        print("VMEVALK IT BENCHMARK")
        print("=" * 70)
        print(f"Models: {models}")
        print(f"Tasks: {tasks}")
        print(f"Strict Mode: {strict_mode}")
        
        results = {
            "models": {},
            "tasks": {},
            "summary": {}
        }
        
        api_keys = api_keys or {}
        
        # Evaluate each model-task combination
        for model_name in models:
            print(f"\n📦 Loading model: {model_name}")
            
            # Check for Runway models and warn
            if "runway" in model_name.lower() or "gen4" in model_name.lower():
                print(f"⚠️  WARNING: Runway models do not support text+image→video")
                print(f"   Runway's gen4_turbo, gen4_aleph, act_two, and veo3")
                print(f"   do not accept both text AND image inputs simultaneously.")
                if strict_mode:
                    print(f"   Skipping {model_name} due to lack of support...")
                    continue
            
            model_results = []
            
            for task_name in tasks:
                print(f"   Task: {task_name}")
                
                # Here you would load the actual model and task
                # For now, this is a placeholder
                print(f"   ✓ Would evaluate {model_name} on {task_name}")
                
                # Store placeholder results
                model_results.append({
                    "task": task_name,
                    "status": "pending"
                })
            
            results["models"][model_name] = model_results
        
        # Generate report if requested
        if generate_report:
            self._generate_report(results, output_dir)
        
        return results
    
    def _evaluate_videos(
        self,
        video_paths: List[str],
        task: Any
    ) -> Dict[str, float]:
        """
        Evaluate generated videos for task performance.
        
        This is a placeholder - actual implementation would include:
        - Video analysis
        - Solution verification
        - Quality metrics
        """
        # Placeholder scores
        return {
            "reasoning_score": 0.75,
            "video_quality": 0.80,
            "solution_accuracy": 0.70
        }
    
    def _save_result(self, result: EvaluationResult):
        """Save evaluation result to file."""
        result_file = self.output_dir / f"{result.model_name}_{result.task_name}_{int(time.time())}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                "reasoning_score": result.reasoning_score,
                "video_quality": result.video_quality,
                "solution_accuracy": result.solution_accuracy,
                "task_name": result.task_name,
                "model_name": result.model_name,
                "duration": result.duration,
                "metadata": result.metadata
            }, f, indent=2)
    
    def _generate_report(self, results: Dict[str, Any], output_dir: str):
        """Generate HTML report of benchmark results."""
        report_path = Path(output_dir) / "benchmark_report.html"
        
        # Simple HTML report
        html = """
        <html>
        <head>
            <title>VMEvalKit Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .warning { color: #ff6600; background: #fff3cd; padding: 10px; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>VMEvalKit Benchmark Report</h1>
            <div class="warning">
                <strong>⚠️ Note on Runway Models:</strong><br>
                Based on official documentation, Runway's API models (gen4_turbo, gen4_aleph, act_two, veo3)
                do NOT support the text+image→video capability required for reasoning tasks.
            </div>
            <h2>Results</h2>
            <pre>{}</pre>
        </body>
        </html>
        """.format(json.dumps(results, indent=2))
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"\n📊 Report saved to: {report_path}")
