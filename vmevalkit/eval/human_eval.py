"""Human evaluation interface for VMEvalKit using Gradio."""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import gradio as gr
from datetime import datetime
import pandas as pd

from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class HumanEvaluator(BaseEvaluator):
    """Human evaluation interface for assessing video generation results.
    
    This evaluator provides a Gradio-based web interface for human annotators
    to evaluate generated videos for reasoning capabilities.
    """
    
    def __init__(self, 
                 output_dir: str = "data/evaluations",
                 experiment_name: str = "pilot_experiment",
                 annotator_name: str = "anonymous"):
        """Initialize the human evaluator.
        
        Args:
            output_dir: Base directory for evaluation outputs
            experiment_name: Name of the experiment to evaluate
            annotator_name: Name of the human annotator
        """
        super().__init__(output_dir, experiment_name)
        self.annotator_name = annotator_name
        self.evaluation_metadata["annotator"] = annotator_name
        
        # Track current evaluation state
        self.current_model = None
        self.current_task_type = None
        self.current_task_id = None
        self.evaluation_queue = []
        self.current_index = 0
        
        # Load all tasks to evaluate
        self._load_evaluation_queue()
    
    def _load_evaluation_queue(self):
        """Load all tasks that need to be evaluated."""
        self.evaluation_queue = []
        
        # Iterate through all models and tasks
        for model_dir in self.experiment_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "inference_log.json":
                continue
                
            model_name = model_dir.name
            
            for task_type_dir in model_dir.iterdir():
                if not task_type_dir.is_dir():
                    continue
                    
                task_type = task_type_dir.name
                
                for task_dir in task_type_dir.iterdir():
                    if not task_dir.is_dir():
                        continue
                        
                    task_id = task_dir.name
                    
                    # Check if already evaluated
                    eval_path = self.output_dir / self.experiment_name / model_name / task_type / task_id / "human-eval.json"
                    if eval_path.exists():
                        continue  # Skip already evaluated tasks
                    
                    self.evaluation_queue.append({
                        "model_name": model_name,
                        "task_type": task_type,
                        "task_id": task_id
                    })
        
        logger.info(f"Loaded {len(self.evaluation_queue)} tasks to evaluate")
    
    def evaluate_single(self, 
                       model_name: str,
                       task_type: str,
                       task_id: str,
                       video_path: str,
                       question_data: Dict[str, Any]) -> Dict[str, Any]:
        """This method is not used directly for human evaluation.
        
        Human evaluation is done through the Gradio interface.
        """
        raise NotImplementedError("Human evaluation is done through the Gradio interface")
    
    def _get_task_data(self, model_name: str, task_type: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific task."""
        task_dir = self.experiment_dir / model_name / task_type / task_id
        
        # Find the timestamped output directory
        output_dirs = list(task_dir.iterdir())
        if not output_dirs:
            return None
            
        output_dir = output_dirs[0]
        
        # Load question data
        question_json_path = output_dir / "question" / "question_metadata.json"
        prompt_path = output_dir / "question" / "prompt.txt"
        first_frame_path = output_dir / "question" / "first_frame.png"
        final_frame_path = output_dir / "question" / "final_frame.png"
        
        if not question_json_path.exists():
            return None
            
        with open(question_json_path, 'r') as f:
            question_data = json.load(f)
        
        # Read prompt
        prompt = ""
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                prompt = f.read()
        
        # Find video
        video_dir = output_dir / "video"
        video_files = list(video_dir.glob("*.mp4"))
        video_path = str(video_files[0]) if video_files else None
        
        return {
            "model_name": model_name,
            "task_type": task_type,
            "task_id": task_id,
            "question_data": question_data,
            "prompt": prompt,
            "first_frame": str(first_frame_path) if first_frame_path.exists() else None,
            "final_frame": str(final_frame_path) if final_frame_path.exists() else None,
            "video_path": video_path,
            "output_dir": str(output_dir)
        }
    
    def _save_evaluation(self, 
                        model_name: str,
                        task_type: str,
                        task_id: str,
                        evaluation: Dict[str, Any]):
        """Save a single evaluation result."""
        # Create output directory
        output_dir = self.output_dir / self.experiment_name / model_name / task_type / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        result = {
            "metadata": {
                "evaluator": "human-eval",
                "annotator": self.annotator_name,
                "timestamp": datetime.now().isoformat(),
                "model_name": model_name,
                "task_type": task_type,
                "task_id": task_id
            },
            "result": evaluation
        }
        
        # Save evaluation
        eval_path = output_dir / "human-eval.json"
        with open(eval_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved evaluation for {model_name}/{task_type}/{task_id}")
    
    def launch_interface(self, share: bool = False, port: int = 7860):
        """Launch the Gradio interface for human evaluation.
        
        Args:
            share: Whether to create a public link
            port: Port to run the interface on
        """
        with gr.Blocks(title="VMEvalKit Human Evaluation") as interface:
            gr.Markdown("# VMEvalKit Human Evaluation Interface")
            gr.Markdown(f"**Annotator:** {self.annotator_name} | **Experiment:** {self.experiment_name}")
            
            # Progress tracking
            with gr.Row():
                progress_text = gr.Markdown(f"Progress: 0/{len(self.evaluation_queue)} tasks evaluated")
                skip_evaluated = gr.Checkbox(label="Skip already evaluated tasks", value=True)
            
            # Current task info
            with gr.Row():
                model_info = gr.Textbox(label="Model", interactive=False)
                task_info = gr.Textbox(label="Task", interactive=False)
            
            # Display area
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Task Input")
                    input_image = gr.Image(label="Input Image", type="filepath")
                    prompt_text = gr.Textbox(label="Prompt", lines=3, interactive=False)
                    
                    if_has_final = gr.Checkbox(label="Has Expected Output", value=False, interactive=False)
                    expected_output = gr.Image(label="Expected Output (if available)", type="filepath")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Generated Video")
                    video_player = gr.Video(label="Generated Video")
            
            # Evaluation form
            gr.Markdown("### Evaluation")
            with gr.Row():
                with gr.Column():
                    # Simple correctness evaluation
                    correctness_score = gr.Radio(
                        choices=[1, 2, 3, 4, 5],
                        label="Solution Correctness",
                        info="Rate the correctness of the solution (1=Completely Wrong, 5=Perfect)"
                    )
                    
                    # Comments
                    comments = gr.Textbox(
                        label="Comments (Optional)",
                        lines=3,
                        placeholder="Any additional observations about the solution..."
                    )
            
            # Navigation buttons
            with gr.Row():
                prev_btn = gr.Button("← Previous", variant="secondary")
                submit_btn = gr.Button("Submit Evaluation", variant="primary")
                next_btn = gr.Button("Next →", variant="secondary")
            
            # Status
            status_text = gr.Textbox(label="Status", interactive=False)
            
            # Define the main update function
            def update_display(index):
                """Update the display with the current task."""
                if index < 0 or index >= len(self.evaluation_queue):
                    return {
                        model_info: "",
                        task_info: "",
                        input_image: None,
                        prompt_text: "",
                        expected_output: None,
                        video_player: None,
                        progress_text: f"Progress: {index}/{len(self.evaluation_queue)} tasks evaluated",
                        status_text: "No more tasks to evaluate!"
                    }
                
                task = self.evaluation_queue[index]
                task_data = self._get_task_data(
                    task["model_name"],
                    task["task_type"],
                    task["task_id"]
                )
                
                if not task_data:
                    return {
                        status_text: "Error loading task data"
                    }
                
                # Check if task has expected output
                has_final = task_data["final_frame"] is not None
                
                return {
                    model_info: task_data["model_name"],
                    task_info: f"{task_data['task_type']} - {task_data['task_id']}",
                    input_image: task_data["first_frame"],
                    prompt_text: task_data["prompt"],
                    if_has_final: has_final,
                    expected_output: task_data["final_frame"] if has_final else None,
                    video_player: task_data["video_path"],
                    progress_text: f"Progress: {index}/{len(self.evaluation_queue)} tasks",
                    status_text: "Task loaded successfully",
                    # Clear previous evaluation
                    correctness_score: None,
                    comments: ""
                }
            
            # Define navigation functions
            def go_previous():
                """Go to previous task."""
                nonlocal self
                self.current_index = max(0, self.current_index - 1)
                return update_display(self.current_index)
            
            def go_next():
                """Go to next task."""
                nonlocal self
                self.current_index = min(len(self.evaluation_queue) - 1, self.current_index + 1)
                return update_display(self.current_index)
            
            def submit_evaluation(correctness, comments_text):
                """Submit the current evaluation."""
                nonlocal self
                
                # Validate inputs
                if correctness is None:
                    return {status_text: "Please select a correctness score!"}
                
                # Get current task
                if self.current_index >= len(self.evaluation_queue):
                    return {status_text: "No task to evaluate!"}
                
                task = self.evaluation_queue[self.current_index]
                
                # Create evaluation result
                evaluation = {
                    "solution_correctness_score": correctness,
                    "comments": comments_text
                }
                
                # Save evaluation
                self._save_evaluation(
                    task["model_name"],
                    task["task_type"],
                    task["task_id"],
                    evaluation
                )
                
                # Move to next task
                self.current_index += 1
                updates = update_display(self.current_index)
                updates[status_text] = "Evaluation saved successfully!"
                return updates
            
            # Connect buttons
            prev_btn.click(
                go_previous,
                outputs=[
                    model_info, task_info, input_image, prompt_text,
                    if_has_final, expected_output, video_player,
                    progress_text, status_text,
                    correctness_score, comments
                ]
            )
            
            next_btn.click(
                go_next,
                outputs=[
                    model_info, task_info, input_image, prompt_text,
                    if_has_final, expected_output, video_player,
                    progress_text, status_text,
                    correctness_score, comments
                ]
            )
            
            submit_btn.click(
                submit_evaluation,
                inputs=[
                    correctness_score, comments
                ],
                outputs=[
                    model_info, task_info, input_image, prompt_text,
                    if_has_final, expected_output, video_player,
                    progress_text, status_text,
                    correctness_score, comments
                ]
            )
            
            # Load first task on startup
            interface.load(
                lambda: update_display(0),
                outputs=[
                    model_info, task_info, input_image, prompt_text,
                    if_has_final, expected_output, video_player,
                    progress_text, status_text,
                    correctness_score, comments
                ]
            )
        
        # Launch the interface
        interface.launch(share=share, server_port=port)
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """Get a summary of all evaluations as a pandas DataFrame.
        
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        # Iterate through evaluation directory
        eval_base = self.output_dir / self.experiment_name
        
        for model_dir in eval_base.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            for task_type_dir in model_dir.iterdir():
                if not task_type_dir.is_dir():
                    continue
                    
                task_type = task_type_dir.name
                
                for task_dir in task_type_dir.iterdir():
                    if not task_dir.is_dir():
                        continue
                        
                    task_id = task_dir.name
                    eval_path = task_dir / "human-eval.json"
                    
                    if eval_path.exists():
                        with open(eval_path, 'r') as f:
                            data = json.load(f)
                            
                        result = data["result"]
                        row = {
                            "model": model_name,
                            "task_type": task_type,
                            "task_id": task_id,
                            "annotator": data["metadata"]["annotator"],
                            "timestamp": data["metadata"]["timestamp"],
                            "solution_correctness_score": result.get("solution_correctness_score", result.get("solution_correctness", 0)),
                            "comments": result.get("comments", "")
                        }
                        results.append(row)
        
        return pd.DataFrame(results)
