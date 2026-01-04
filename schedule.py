#!/usr/bin/env python3
"""
Simple wrapper for running VMEvalKit inference experiments.
Calls VMEvalKit's generate_videos.py with proper paths.
Supports multi-GPU parallel execution.
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse
from multiprocessing import Process
from datetime import datetime
import time

# Get the VMEvalKit path
VMEVALKIT_DIR = Path(__file__).parent.parent / "VMEvalKit"
GENERATE_SCRIPT = VMEVALKIT_DIR / "examples/generate_videos.py"


def run_inference_on_gpu(model, questions_dir, output_dir, task_ids, gpu_id, log_dir="inference_logs"):
    """Run inference on a specific GPU"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"🚀 Starting inference for model: {model}")
    print(f"📊 Using GPU: {gpu_id}")
    print(f"📝 Log directory: {log_dir}")
    print("=" * 70)
    
    # Build command for VMEvalKit
    print(f"Questions dir: {questions_dir}")
    cmd = [
        sys.executable, str(GENERATE_SCRIPT),
        "--model", model,
        "--questions-dir", questions_dir,
        "--output-dir", output_dir,
        "--gpu", str(gpu_id)
    ]
    
    if task_ids:
        cmd.extend(["--task-id"] + task_ids)
    
    # Create log files
    log_file = os.path.join(log_dir, f"{model.replace('/', '_')}_gpu{gpu_id}.log")
    err_file = os.path.join(log_dir, f"{model.replace('/', '_')}_gpu{gpu_id}.err")
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        with open(log_file, "w") as fout, open(err_file, "w") as ferr:
            result = subprocess.run(cmd, stdout=fout, stderr=ferr, cwd=VMEVALKIT_DIR)
        
        if result.returncode == 0:
            print(f"✅ Successfully completed inference for: {model} on GPU {gpu_id}")
            print(f"📄 Log saved to: {log_file}")
        else:
            print(f"❌ Failed inference for: {model} on GPU {gpu_id}")
            print(f"📄 Log saved to: {log_file}")
            print(f"❌ Error log saved to: {err_file}")
            
            try:
                with open(err_file, "r") as f:
                    error_lines = f.readlines()
                    if error_lines:
                        print("Last few error lines:")
                        for line in error_lines[-5:]:
                            print(f"  {line.strip()}")
            except:
                pass
                
    except Exception as e:
        print(f"❌ Exception during inference for {model} on GPU {gpu_id}: {e}")
        try:
            with open(err_file, "a") as ferr:
                ferr.write(f"Exception: {e}\n")
        except:
            pass


def launch_inference(model, questions_dir, output_dir, task_ids, gpu_id, log_dir):
    """Launch inference process for a model on specific GPU"""
    p = Process(target=run_inference_on_gpu, 
                args=(model, questions_dir, output_dir, task_ids, gpu_id, log_dir))
    p.start()
    print(f"▶️  Started inference for {model} on GPU {gpu_id}")
    return p


def smart_schedule(models, devices, questions_dir, output_dir, task_ids, log_dir):
    """Schedule models across available GPUs with smart queuing"""
    pending = list(enumerate(models))
    running = {}
    completed = []

    # Fill up initial GPUs
    for gpu_id in devices:
        if pending:
            i, model = pending.pop(0)
            running[gpu_id] = (launch_inference(model, questions_dir, output_dir, 
                                               task_ids, gpu_id, log_dir), i, model)

    # Process remaining tasks
    while running or pending:
        done = []
        for gpu_id, (p, i, model) in running.items():
            if not p.is_alive():
                p.join()
                done.append(gpu_id)
                completed.append((i, model))
                print(f"✅ GPU {gpu_id} completed inference for {model}")

        # Free up GPUs and assign new tasks
        for gpu_id in done:
            del running[gpu_id]
            if pending:
                i, model = pending.pop(0)
                running[gpu_id] = (launch_inference(model, questions_dir, output_dir, 
                                                   task_ids, gpu_id, log_dir), i, model)

        if running:
            time.sleep(2)

    return completed


def main():
    parser = argparse.ArgumentParser(description="Run VMEvalKit inference with multi-GPU support")
    parser.add_argument("--model", required=True, nargs="+", help="Models to run")
    parser.add_argument("--questions-dir", default="./data/questions", help="Questions directory")
    parser.add_argument("--output-dir", default="./data/outputs", help="Output directory")
    parser.add_argument("--task-id", nargs="*", help="Specific task IDs to run")
    parser.add_argument("--gpu", type=str, help="GPU(s) to use (e.g., '0' or '0,1,2,3')")
    parser.add_argument("--log-dir", type=str, default="inference_logs", 
                       help="Directory to save inference logs")
    
    args = parser.parse_args()
    
    # Parse GPU list
    if args.gpu:
        devices = [int(g.strip()) for g in args.gpu.split(',')]
    else:
        # Default to GPU 0 if not specified
        devices = [0]
    
    models = args.model
    
    print("=" * 70)
    print(f"📊 VMEvalKit Multi-GPU Inference")
    print(f"Models: {len(models)} - {models}")
    print(f"GPUs: {devices}")
    print(f"Questions Dir: {args.questions_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Task IDs: {args.task_id if args.task_id else 'All'}")
    print(f"Log Dir: {args.log_dir}")
    print("=" * 70)
    
    # Single GPU mode - run directly without multiprocessing overhead
    if len(devices) == 1 and len(models) == 1:
        run_inference_on_gpu(models[0], args.questions_dir, args.output_dir, 
                            args.task_id, devices[0], args.log_dir)
        return
    
    # Multi-GPU mode - schedule tasks
    t0 = datetime.now()
    
    completed = smart_schedule(models, devices, args.questions_dir, 
                              args.output_dir, args.task_id, args.log_dir)
    
    t1 = datetime.now()
    
    print("=" * 70)
    print(f"✅ All inference tasks finished.")
    print(f"Completed: {len(completed)} models")
    print(f"Time: {t0.strftime('%F %T')} → {t1.strftime('%F %T')} ({t1 - t0})")
    print("=" * 70)


if __name__ == "__main__":
    main()
