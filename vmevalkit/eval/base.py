"""Base evaluator class for VMEvalKit evaluation methods."""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for all evaluation methods in VMEvalKit.
    
    This class provides common functionality for evaluating video generation models'
    reasoning capabilities.
    """
    
    def __init__(self, 
                 output_dir: str = "data/evaluations",
                 experiment_name: str = "pilot_experiment"):
        """Initialize the base evaluator.
        
        Args:
            output_dir: Base directory for evaluation outputs
            experiment_name: Name of the experiment to evaluate
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = Path("data/outputs") / experiment_name
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track evaluation metadata
        self.evaluation_metadata = {
            "evaluator_type": self.__class__.__name__,
            "start_time": datetime.now().isoformat(),
            "experiment_name": experiment_name,
        }
    
    @abstractmethod
    def evaluate_single(self, 
                       model_name: str,
                       task_type: str,
                       task_id: str,
                       video_path: str,
                       question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single video generation result.
        
        Args:
            model_name: Name of the model that generated the video
            task_type: Type of task (e.g., 'chess_task', 'maze_task')
            task_id: ID of the specific task (e.g., 'chess_0000')
            video_path: Path to the generated video
            question_data: Question metadata including prompt and images
            
        Returns:
            Dict containing evaluation results
        """
        pass
    
    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate all results for a specific model.
        
        Args:
            model_name: Name of the model to evaluate
            
        Returns:
            Dict containing all evaluation results for the model
        """
        model_dir = self.experiment_dir / model_name
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        results = {
            "model_name": model_name,
            "evaluations": {},
            "summary": {}
        }
        
        # Iterate through all task types
        for task_type_dir in model_dir.iterdir():
            if not task_type_dir.is_dir():
                continue
                
            task_type = task_type_dir.name
            results["evaluations"][task_type] = {}
            
            # Iterate through all tasks in this type
            for task_dir in task_type_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_id = task_dir.name
                
                # Find the timestamped output directory
                output_dirs = list(task_dir.iterdir())
                if not output_dirs:
                    logger.warning(f"No output found for {model_name}/{task_type}/{task_id}")
                    continue
                
                output_dir = output_dirs[0]  # Assume one output per task
                
                # Load question data
                question_json_path = output_dir / "question" / "question_metadata.json"
                if not question_json_path.exists():
                    logger.warning(f"Question metadata not found: {question_json_path}")
                    continue
                    
                with open(question_json_path, 'r') as f:
                    question_data = json.load(f)
                
                # Find video file
                video_dir = output_dir / "video"
                video_files = list(video_dir.glob("*.mp4"))
                if not video_files:
                    logger.warning(f"No video found in {video_dir}")
                    continue
                    
                video_path = str(video_files[0])
                
                # Evaluate this task
                try:
                    eval_result = self.evaluate_single(
                        model_name=model_name,
                        task_type=task_type,
                        task_id=task_id,
                        video_path=video_path,
                        question_data=question_data
                    )
                    
                    results["evaluations"][task_type][task_id] = eval_result
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}/{task_type}/{task_id}: {e}")
                    results["evaluations"][task_type][task_id] = {
                        "error": str(e),
                        "status": "failed"
                    }
        
        
        # Save results
        self._save_results(model_name, results)
        
        return results
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models in the experiment.
        
        Returns:
            Dict containing evaluation results for all models
        """
        all_results = {}
        
        # Find all model directories
        for model_dir in self.experiment_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                results = self.evaluate_model(model_name)
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                all_results[model_name] = {"error": str(e)}
        
        # Save combined results
        self.evaluation_metadata["end_time"] = datetime.now().isoformat()
        combined_results = {
            "metadata": self.evaluation_metadata,
            "results": all_results
        }
        
        output_path = self.output_dir / self.experiment_name / f"{self.evaluation_metadata['evaluator_type']}_all_models.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        return all_results
    
    
    
    def _save_results(self, model_name: str, results: Dict[str, Any]):
        """Save evaluation results to the output directory.
        
        Args:
            model_name: Name of the evaluated model
            results: Evaluation results to save
        """
        # Create output directory structure
        output_base = self.output_dir / self.experiment_name / model_name
        
        for task_type, task_results in results["evaluations"].items():
            for task_id, eval_result in task_results.items():
                # Create directory for this task
                task_output_dir = output_base / task_type / task_id
                task_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save evaluation result
                eval_filename = f"{self.evaluation_metadata['evaluator_type']}.json"
                eval_path = task_output_dir / eval_filename
                
                with open(eval_path, 'w') as f:
                    json.dump({
                        "metadata": {
                            "evaluator": self.evaluation_metadata['evaluator_type'],
                            "timestamp": datetime.now().isoformat(),
                            "model_name": model_name,
                            "task_type": task_type,
                            "task_id": task_id
                        },
                        "result": eval_result
                    }, f, indent=2)
        
        logger.info(f"Saved evaluation results for {model_name} to {output_base}")
