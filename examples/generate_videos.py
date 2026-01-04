#!/usr/bin/env python3
"""
VMEvalKit Video Generation

Generate videos using various models on your question datasets.

Key Features:
- Auto-discovers all tasks from questions directory
- Run specific models or use defaults
- Sequential execution with progress tracking and resume capability

Requirements:
- Questions directory with tasks in format: domain_task/task_id/{first_frame.png, prompt.txt}
- API keys configured for commercial models

Use --help for usage examples.
"""

import sys
import json
import shutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from PIL import Image
from dotenv import load_dotenv
import multiprocessing as mp
from functools import partial

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.runner.inference import InferenceRunner
from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS, get_model_family



# Paths are now configurable via CLI arguments - no hardcoded defaults

# Discover domains by scanning questions directory
def get_available_domains(questions_dir_path):
    """Discover available domains from questions directory."""
    questions_dir = Path(questions_dir_path)
    if not questions_dir.exists():
        return []
    
    domains = []
    for item in questions_dir.iterdir():
        if item.is_dir() and item.name.endswith('_task'):
            domain = item.name.replace('_task', '')
            domains.append(domain)
    return sorted(domains)

def discover_all_tasks_from_folders(questions_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover all tasks by scanning questions directory.
    
    Args:
        questions_dir: Path to questions directory
        
    Returns:
        Dictionary mapping domain to list of task dictionaries
    """
    print(f"🔍 Discovering tasks from: {questions_dir}")
    
    tasks_by_domain = {}
    total_tasks = 0
    
    # Scan each domain folder
    for domain_dir in sorted(questions_dir.glob("*_task")):
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name.replace("_task", "")
        domain_tasks = []
        
        print(f"   📁 Scanning {domain_dir.name}/")
        
        # Scan each task folder
        for task_dir in sorted(domain_dir.iterdir()):
            if not task_dir.is_dir():
                continue
                
            task_id = task_dir.name
            
            # Check for required files
            prompt_file = task_dir / "prompt.txt"
            first_image = task_dir / "first_frame.png"
            final_image = task_dir / "final_frame.png"
            ground_truth = task_dir / "ground_truth.mp4"
            
            if not prompt_file.exists():
                print(f"      ⚠️  Skipping {task_id}: Missing prompt.txt")
                continue
                
            if not first_image.exists():
                print(f"      ⚠️  Skipping {task_id}: Missing first_frame.png")
                continue
            
            # Load prompt
            prompt_text = prompt_file.read_text().strip()
            
            # Create task dictionary
            task = {
                "id": task_id,
                "domain": domain,
                "prompt": prompt_text,
                "first_image_path": str(first_image.absolute()),
                "final_image_path": str(final_image.absolute()) if final_image.exists() else None,
                "ground_truth_video": str(ground_truth.absolute()) if ground_truth.exists() else None
            }
            
            domain_tasks.append(task)
            
        print(f"      ✅ Found {len(domain_tasks)} tasks in {domain}")
        tasks_by_domain[domain] = domain_tasks
        total_tasks += len(domain_tasks)
    
    print(f"\n📊 Discovery Summary: {total_tasks} total tasks")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain}: {len(tasks)} tasks")
    
    return tasks_by_domain


def create_output_structure(base_dir: Path) -> None:
    """Create organized output directory structure for the new system."""
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # With the new structured output, each inference creates its own folder
    # Logs are disabled; no top-level logs directory
    
    print(f"📁 Output directory structure ready at: {base_dir}")
    print(f"   Each inference will create a self-contained folder with:")
    print(f"   - video/: Generated video file")
    print(f"   - question/: Input images and prompt")
    print(f"   - metadata.json: Complete inference metadata")


def create_model_directories(base_dir: Path, models: Dict[str, str], questions_dir: Path) -> None:
    """Create a subfolder per model and mirror questions tree for visibility."""
    # Create per-model root
    for model_name in models.keys():
        model_root = base_dir / model_name
        model_root.mkdir(exist_ok=True, parents=True)
        # Mirror questions directory structure with empty domain/task folders
        for domain_dir in sorted(questions_dir.glob("*_task")):
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name  # e.g., rotation_task
            # Create domain folder under model
            model_domain_dir = model_root / domain_name
            model_domain_dir.mkdir(exist_ok=True, parents=True)
            # Create each task folder (without runs yet)
            for task_dir in sorted(domain_dir.glob("*") ):
                if task_dir.is_dir():
                    (model_domain_dir / task_dir.name).mkdir(exist_ok=True, parents=True)
    print(f"📁 Created per-model folders under: {base_dir} and mirrored question tasks")




def _ensure_real_png(image_path: str) -> bool:
    """If file is SVG mislabeled as .png, convert to real PNG in-place using CairoSVG."""
    try:
        # Quick check by trying to open as PNG
        Image.open(image_path).verify()
        return True
    except Exception:
        # Fallback: detect SVG text and convert
        with open(image_path, 'rb') as f:
            head = f.read(1024)
        # Heuristic: look for '<svg' in the head bytes
        if b"<svg" in head.lower():
            import cairosvg
            with open(image_path, 'rb') as f:
                svg_bytes = f.read()
            cairosvg.svg2png(bytestring=svg_bytes, write_to=image_path)
            # Validate conversion
            Image.open(image_path).verify()
            print(f"   🔧 Converted SVG→PNG in-place: {image_path}")
            return True
    return False


def run_single_inference(
    model_name: str,
    task: Dict[str, Any],
    category: str,
    output_dir: Path,
    runner: Optional[InferenceRunner] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference for a single task-model pair using the new structured output system.
    
    Args:
        model_name: Name of the model to use
        task: Task dictionary from dataset
        category: Task category
        output_dir: Base output directory
        runner: Optional InferenceRunner instance (created if not provided)
        **kwargs: Additional model parameters
        
    Returns:
        Result dictionary with metadata
    """
    task_id = task["id"]
    image_path = task["first_image_path"]
    prompt = task["prompt"]

    run_id = f"{model_name}_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n  🎬 Generating: {task_id} with {model_name}")
    print(f"     Image: {image_path}")
    print(f"     Prompt: {prompt[:80]}...")
    
    start_time = datetime.now()

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Validate image is a real PNG (auto-fix if it's actually SVG)
    if not _ensure_real_png(image_path):
        raise ValueError(f"Input image invalid or corrupt: {image_path}")
    

    if runner is None:
        runner = InferenceRunner(output_dir=str(output_dir))
    

    result = runner.run(
        model_name=model_name,
        image_path=image_path,
        text_prompt=prompt,
        run_id=run_id,
        question_data=task,  # Pass full task data for structured output
        **kwargs  # Clean! No API key filtering needed
    )
    
    result.update({
        "task_id": task_id,
        "category": category,
        "model_name": model_name,
        "model_family": get_model_family(model_name),
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "success": result.get("status") != "failed"
    })
    
    if result.get("status") != "failed":
        print(f"     ✅ Success! Structured output saved to: {result.get('inference_dir', 'N/A')}")
    else:
        print(f"     ❌ Failed: {result.get('error', 'Unknown error')}")
    
    return result


def process_tasks_on_gpu(
    gpu_id: int,
    model_name: str,
    tasks: List[Dict[str, Any]],
    model_output_dir: Path,
    domain: str,
    skip_existing: bool,
    total_jobs: int,
    job_offset: int
) -> List[Dict[str, Any]]:
    """
    Process a subset of tasks on a specific GPU.
    
    Args:
        gpu_id: GPU device ID to use
        model_name: Name of the model
        tasks: List of tasks to process on this GPU
        model_output_dir: Output directory for this model
        domain: Domain name
        skip_existing: Whether to skip existing outputs
        total_jobs: Total number of jobs across all GPUs
        job_offset: Starting job number for this GPU
        
    Returns:
        List of result dictionaries
    """
    # Set CUDA_VISIBLE_DEVICES for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"      [GPU {gpu_id}] 🎮 Starting processing on GPU {gpu_id} with {len(tasks)} tasks")
    
    # Create runner for this GPU
    runner = InferenceRunner(output_dir=str(model_output_dir))
    
    results = []
    local_job_counter = 0
    
    for task in tasks:
        local_job_counter += 1
        global_job_id = job_offset + local_job_counter
        task_id = task["id"]
        
        print(f"      [GPU {gpu_id}] [{global_job_id}/{total_jobs}] Processing: {task_id}")
        
        # Check if inference folder already exists WITH actual video file
        run_id_pattern = f"{model_name}_{task_id}_*"
        domain_dir_name = f"{domain}_task"
        task_folder = model_output_dir / domain_dir_name / task_id
        existing_dirs = list(task_folder.glob(run_id_pattern))
        
        # Verify the run folder actually contains a video file
        has_valid_output = False
        if existing_dirs:
            for run_dir in existing_dirs:
                video_dir = run_dir / "video"
                if video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
                    if video_files:
                        has_valid_output = True
                        break
        
        if skip_existing and has_valid_output:
            print(f"      [GPU {gpu_id}] ⏭️  Skipped (existing output: {existing_dirs[0].name})")
            result = {
                "task_id": task_id,
                "model_name": model_name,
                "domain": domain,
                "success": True,
                "skipped": True,
                "gpu_id": gpu_id
            }
            results.append(result)
            continue
        
        result = run_single_inference(
            model_name=model_name,
            task=task,
            category=domain,
            output_dir=model_output_dir,
            runner=runner
        )
        
        result["gpu_id"] = gpu_id
        result["skipped"] = False
        results.append(result)
        
        if result["success"]:
            print(f"      [GPU {gpu_id}] ✅ Completed successfully")
        else:
            print(f"      [GPU {gpu_id}] ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"      [GPU {gpu_id}] 🏁 GPU {gpu_id} finished processing {len(tasks)} tasks")
    return results


def run_pilot_experiment(
    tasks_by_domain: Dict[str, List[Dict[str, Any]]],
    models: Dict[str, str],
    output_dir: Path,
    questions_dir: Path,
    skip_existing: bool = True,
    gpu_list: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run full pilot experiment with MULTI-GPU PARALLEL execution on ALL human-approved tasks.
    
    Processes one model at a time. For each model and domain, splits tasks across GPUs.
    
    Args:
        tasks_by_domain: Dictionary mapping domain to task lists
        models: Dictionary of model names to test
        output_dir: Base output directory
        questions_dir: Questions directory path
        skip_existing: Skip tasks that already have outputs
        gpu_list: List of GPU IDs to use (e.g., [0, 1, 2, 3])
        
    Returns:
        Dictionary with all results and statistics
    """
    num_gpus = len(gpu_list) if gpu_list else 1
    execution_mode = f"MULTI-GPU PARALLEL ({num_gpus} GPUs)" if num_gpus > 1 else "SEQUENTIAL (1 GPU)"
    
    print("🚀 VMEVAL KIT EXPERIMENT - MULTI-GPU EXECUTION")
    print(f"\n📊 Experiment Configuration:")
    print(f"   Models to run: {len(models)} - {', '.join(models.keys())}")
    print(f"   Domains: {len(tasks_by_domain)}")
    print(f"   🔄 Execution Mode: {execution_mode}")
    if gpu_list and num_gpus > 1:
        print(f"   🎮 GPUs: {gpu_list}")
    

    total_tasks = sum(len(tasks) for tasks in tasks_by_domain.values())
    total_generations = total_tasks * len(models)
    
    print(f"\n📈 Task Distribution:")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain.title()}: {len(tasks)} approved tasks")
    print(f"   Total approved tasks: {total_tasks}")
    print(f"   Total generations: {total_generations}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Skip existing: {skip_existing}\n")
    
    create_output_structure(output_dir)
    create_model_directories(output_dir, models, questions_dir)
    
    all_results = []
    
    statistics = {
        "total_tasks": total_tasks,
        "total_generations": total_generations,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "by_model": {},
        "by_domain": {}
    }
    
    for model in models.keys():
        statistics["by_model"][model] = {"completed": 0, "failed": 0, "skipped": 0}
    for domain in tasks_by_domain.keys():
        statistics["by_domain"][domain] = {"completed": 0, "failed": 0, "skipped": 0}
    
    experiment_start = datetime.now()
    
    total_jobs = sum(len(tasks) for tasks in tasks_by_domain.values()) * len(models)
    print(f"📋 Total inference jobs to run: {total_jobs}")
    if num_gpus > 1:
        print(f"🚀 Starting multi-GPU parallel execution...\n")
        print(f"   Processing order: Model by model, domain by domain, tasks split across {num_gpus} GPUs\n")
    else:
        print("🚀 Starting sequential execution...\n")
        print("   Processing order: Model by model, task by task\n")
    
    job_counter = 0
    
    for model_name, model_display in models.items():
        print(f"🤖 Processing Model: {model_display} ({model_name})")
        model_output_dir = output_dir / model_name
        
        model_start_time = datetime.now()
        model_completed = 0
        model_failed = 0
        model_skipped = 0
        
        # Process all tasks for this model
        for domain, tasks in tasks_by_domain.items():
            print(f"\n  📚 Domain: {domain.title()} ({len(tasks)} tasks)")
            
            if num_gpus > 1 and len(tasks) > 0:
                # Multi-GPU parallel execution: split tasks across GPUs
                print(f"    🔀 Splitting {len(tasks)} tasks across {num_gpus} GPUs...")
                
                # Split tasks into chunks for each GPU
                tasks_per_gpu = []
                chunk_size = (len(tasks) + num_gpus - 1) // num_gpus  # Ceiling division
                
                for i in range(num_gpus):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(tasks))
                    if start_idx < len(tasks):
                        gpu_tasks = tasks[start_idx:end_idx]
                        tasks_per_gpu.append((gpu_list[i], gpu_tasks, start_idx))
                        print(f"    📦 GPU {gpu_list[i]}: {len(gpu_tasks)} tasks (indices {start_idx}-{end_idx-1})")
                
                # Create processes for each GPU
                processes = []
                manager = mp.Manager()
                return_dict = manager.dict()
                
                domain_jobs = len(tasks)
                
                for idx, (gpu_id, gpu_tasks, start_idx) in enumerate(tasks_per_gpu):
                    job_offset = job_counter + start_idx
                    
                    def worker_wrapper(gpu_id, tasks, return_dict, idx):
                        try:
                            results = process_tasks_on_gpu(
                                gpu_id=gpu_id,
                                model_name=model_name,
                                tasks=tasks,
                                model_output_dir=model_output_dir,
                                domain=domain,
                                skip_existing=skip_existing,
                                total_jobs=domain_jobs,
                                job_offset=job_offset
                            )
                            return_dict[idx] = results
                        except Exception as e:
                            print(f"      [GPU {gpu_id}] ❌ Error: {str(e)}")
                            return_dict[idx] = []
                    
                    p = mp.Process(
                        target=worker_wrapper,
                        args=(gpu_id, gpu_tasks, return_dict, idx)
                    )
                    processes.append(p)
                    p.start()
                
                # Wait for all processes to complete
                for p in processes:
                    p.join()
                
                # Collect results from all GPUs
                for idx in range(len(tasks_per_gpu)):
                    if idx in return_dict:
                        gpu_results = return_dict[idx]
                        for result in gpu_results:
                            all_results.append(result)
                            
                            if result.get("skipped", False):
                                statistics["skipped"] += 1
                                statistics["by_model"][model_name]["skipped"] += 1
                                statistics["by_domain"][domain]["skipped"] += 1
                                model_skipped += 1
                            elif result["success"]:
                                statistics["completed"] += 1
                                statistics["by_model"][model_name]["completed"] += 1
                                statistics["by_domain"][domain]["completed"] += 1
                                model_completed += 1
                            else:
                                statistics["failed"] += 1
                                statistics["by_model"][model_name]["failed"] += 1
                                statistics["by_domain"][domain]["failed"] += 1
                                model_failed += 1
                
                job_counter += len(tasks)
                print(f"    ✅ Domain {domain.title()} completed on {num_gpus} GPUs")
                
            else:
                # Sequential execution (single GPU or backward compatibility)
                runner = InferenceRunner(output_dir=str(model_output_dir))
                
                for task in tasks:
                    job_counter += 1
                    task_id = task["id"]
                    
                    print(f"    [{job_counter}/{total_jobs}] Processing: {task_id}")
                    
                    # Check if inference folder already exists WITH actual video file
                    run_id_pattern = f"{model_name}_{task_id}_*"
                    domain_dir_name = f"{domain}_task"
                    task_folder = model_output_dir / domain_dir_name / task_id
                    existing_dirs = list(task_folder.glob(run_id_pattern))
                    
                    # Verify the run folder actually contains a video file
                    has_valid_output = False
                    if existing_dirs:
                        for run_dir in existing_dirs:
                            video_dir = run_dir / "video"
                            if video_dir.exists():
                                video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
                                if video_files:
                                    has_valid_output = True
                                    break
                    
                    if skip_existing and has_valid_output:
                        statistics["skipped"] += 1
                        statistics["by_model"][model_name]["skipped"] += 1
                        statistics["by_domain"][domain]["skipped"] += 1
                        model_skipped += 1
                        print(f"      ⏭️  Skipped (existing output: {existing_dirs[0].name})")
                        continue
                    
                    result = run_single_inference(
                        model_name=model_name,
                        task=task,
                        category=domain,
                        output_dir=model_output_dir,
                        runner=runner
                    )
                    
                    all_results.append(result)
                    
                    if result["success"]:
                        statistics["completed"] += 1
                        statistics["by_model"][model_name]["completed"] += 1
                        statistics["by_domain"][domain]["completed"] += 1
                        model_completed += 1
                        print(f"      ✅ Completed successfully")
                    else:
                        statistics["failed"] += 1
                        statistics["by_model"][model_name]["failed"] += 1
                        statistics["by_domain"][domain]["failed"] += 1
                        model_failed += 1
                        print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
        
        model_duration = (datetime.now() - model_start_time).total_seconds()
        print(f"\n  📊 Model {model_display} Summary: {model_completed} completed, {model_failed} failed, {model_skipped} skipped in {format_duration(model_duration)}")
    
    experiment_end = datetime.now()
    duration = (experiment_end - experiment_start).total_seconds()
    
    statistics["experiment_start"] = experiment_start.isoformat()
    statistics["experiment_end"] = experiment_end.isoformat()
    statistics["duration_seconds"] = duration
    statistics["duration_formatted"] = format_duration(duration)
    
    print(f"\n⏱️  Sequential execution completed in {format_duration(duration)}")
    
    return {
        "results": all_results,
        "statistics": statistics,
    }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def main():
    """Main execution function with resume support."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="VMEvalKit Video Generation - Flexible model and task selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run all tasks with specific models
            python generate_videos.py --questions-dir ./questions --output-dir ./outputs --model luma-ray-2 openai-sora-2
            
            # Run with a single model
            python generate_videos.py --questions-dir ./questions --output-dir ./outputs --model veo-3.0-generate
            
            # Run specific task IDs with models
            python generate_videos.py --questions-dir ./questions --output-dir ./outputs --model runway-gen4-turbo --task-id chess_0001 maze_0005
        """
    )
    
    parser.add_argument(
        "--model", 
        nargs="+", 
        required=True,
        help=f"Model(s) to run (REQUIRED). Available: {', '.join(list(AVAILABLE_MODELS.keys())[:10])}... (see --list-models for all)"
    )
    
    parser.add_argument(
        "--questions-dir",
        type=str,
        default="./questions",
        help="Path to questions directory with domain_task/task_id/{first_frame.png, prompt.txt} structure"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Path for inference outputs (default: ./outputs)"
    )
    
    
    parser.add_argument(
        "--task-id",
        nargs="+", 
        default=None,
        help="Specific task ID(s) to run (e.g., chess_0001 maze_0005). Overrides other task selection."
    )
    
    
    
    parser.add_argument("--list-models", action="store_true", help="List all available models and exit")
    
    parser.add_argument(
        "--override",
        dest="override",
        action="store_true",
        help="Delete output directory before running (override existing outputs)"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Single GPU device ID to use (e.g., --gpu 1 for GPU 1). Use --gpus for multiple GPUs."
    )
    
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="Multiple GPU device IDs to use for parallel processing (e.g., --gpus 0 1 2 3). Tasks will be split across GPUs at domain level."
    )
    
    # Legacy parameter for compatibility
    parser.add_argument(
        "--only-model",
        nargs="*",
        default=None,
        help="(Legacy) Same as --model"
    )
    
    args = parser.parse_args()
    
    # Handle GPU configuration
    gpu_list = None
    if args.gpus:
        # Multiple GPUs for parallel processing
        gpu_list = args.gpus
        print(f"🎮 Using {len(gpu_list)} GPUs for parallel processing: {gpu_list}")
    elif args.gpu is not None:
        # Single GPU (legacy mode)
        gpu_list = [args.gpu]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🎮 Using GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")
    else:
        # No GPU specified, use GPU 0 as default
        gpu_list = [0]
        print(f"🎮 No GPU specified, using default GPU 0")
    
    if args.list_models:
        print("🎬 Available Models:")
        print("=" * 60)
        families = {}
        for model_name, model_info in AVAILABLE_MODELS.items():
            family = model_info.get('family', 'Unknown')
            if family not in families:
                families[family] = []
            families[family].append((model_name, model_info.get('description', '')))
        
        for family, models in sorted(families.items()):
            print(f"\n📁 {family}:")
            for model_name, description in sorted(models):
                print(f"   • {model_name:25} - {description}")
        
        print(f"\nTotal: {len(AVAILABLE_MODELS)} models across {len(families)} families")
        return 
    
    questions_dir = Path(args.questions_dir)
    output_dir = Path(args.output_dir)
    
    if args.override:
        if output_dir.exists():
            print(f"🗑️  Override mode: Deleting {output_dir}...")
            shutil.rmtree(output_dir)
            print(f"   ✅ Deleted {output_dir}")
        else:
            print(f"   ℹ️  Output directory does not exist: {output_dir}")
    
    print("🔍 Discovering human-approved tasks from folder structure...")
    
    if not questions_dir.exists():
        raise ValueError(f"Questions directory not found at: {questions_dir}. Please ensure the questions directory exists with task folders.")

    all_tasks_by_domain = discover_all_tasks_from_folders(questions_dir)
    
    # Select tasks based on arguments
    if args.task_id:
        # Specific task IDs requested
        print(f"   🎯 Running specific task IDs: {', '.join(args.task_id)}")
        tasks_by_domain = {}
        for task_id in args.task_id:
            # Find which domain this task belongs to
            found = False
            for domain, tasks in all_tasks_by_domain.items():
                for task in tasks:
                    if task['id'] == task_id:
                        if domain not in tasks_by_domain:
                            tasks_by_domain[domain] = []
                        tasks_by_domain[domain].append(task)
                        found = True
                        break
            if not found:
                print(f"   ⚠️  Task ID '{task_id}' not found")
    else:
        # Run all discovered tasks
        tasks_by_domain = all_tasks_by_domain
        print(f"   🎯 Running all discovered tasks")
    
    if not args.model:
        raise ValueError("Model selection required. Use --model to specify one or more models, or --list-models to see available options.")
    
    model_names = args.model
    
    selected_models = {}
    unavailable_models = []
    for model_name in model_names:
        if model_name in AVAILABLE_MODELS:
            selected_models[model_name] = AVAILABLE_MODELS[model_name].get('family', 'Unknown')
        else:
            unavailable_models.append(model_name)
    
    if unavailable_models:
        print(f"⚠️  Models not available: {', '.join(unavailable_models)}. Use --list-models to see all available models")
    
    if not selected_models:
        raise ValueError("No valid models selected")
        
    print(f"\n🎯 Selected {len(selected_models)} model(s): {', '.join(selected_models.keys())}")

    print(f"\n🔍 Verifying {len(selected_models)} model(s) for testing...")
    for model_name, family in selected_models.items():
        print(f"   ✅ {model_name}: {family}")
    
    
    if not tasks_by_domain or sum(len(tasks) for tasks in tasks_by_domain.values()) == 0:
        raise ValueError("No approved tasks found. Please check the questions directory structure.")
    
    experiment_results = run_pilot_experiment(
        tasks_by_domain=tasks_by_domain,
        models=selected_models,
        output_dir=output_dir,
        questions_dir=questions_dir,
        skip_existing=True,
        gpu_list=gpu_list,
    )
    
    print("🎉 VIDEO GENERATION COMPLETE!")
    stats = experiment_results["statistics"]
    
    actual_total_attempted = stats['completed'] + stats['failed'] + stats['skipped']
    
    print(f"\n📊 Final Statistics:")
    print(f"   Models tested: {len(selected_models)}")
    print(f"   Tasks per model: {stats['total_tasks']}")
    print(f"   Total possible generations: {stats['total_generations']}")
    print(f"   Total attempted: {actual_total_attempted}")
    print(f"   Completed: {stats['completed']} ({stats['completed']/max(actual_total_attempted,1)*100:.1f}%)")
    print(f"   Failed: {stats['failed']} ({stats['failed']/max(actual_total_attempted,1)*100:.1f}%)")
    print(f"   Skipped: {stats['skipped']} ({stats['skipped']/max(actual_total_attempted,1)*100:.1f}%)")
    print(f"   ⏱️ Duration: {stats['duration_formatted']}")
    
    print(f"\n🎯 Results by Domain:")
    for domain, domain_stats in stats['by_domain'].items():
        domain_total = domain_stats['completed'] + domain_stats['failed'] + domain_stats['skipped']
        if domain_total > 0:
            c, f, s = domain_stats['completed'], domain_stats['failed'], domain_stats['skipped']
            print(f"   {domain.title()}: ✅ {c} completed | ❌ {f} failed | ⏭️  {s} skipped")
    
    print(f"\n🤖 Results by Model:")
    for model_name, model_stats in stats['by_model'].items():
        model_total = model_stats['completed'] + model_stats['failed'] + model_stats['skipped']
        if model_total > 0:  # Only show models that were actually run
            c, f, s = model_stats['completed'], model_stats['failed'], model_stats['skipped']
            print(f"   {model_name}: ✅ {c} | ❌ {f} | ⏭️  {s}")
    
    print(f"\n📁 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
