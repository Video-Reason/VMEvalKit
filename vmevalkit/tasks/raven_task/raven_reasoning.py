#!/usr/bin/env python3
"""
RAVEN Reasoning Task for VMEvalKit

This module generates Progressive Matrix (RPM) reasoning tasks for video model evaluation.


The task evaluates video models' ability to:
1. Recognize visual patterns across multiple panels
2. Apply abstract logical rules (progression, arithmetic, etc.)
3. Complete missing patterns through reasoning
4. Generate coherent reasoning sequences in video form

Author: VMEvalKit Team
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

# Use local minimal RAVEN core (no external submodule dependency)
try:
    from .local_raven import (
        IMAGE_SIZE,
        build_center_single,
        build_distribute_four,
        build_distribute_nine,
        generate_matrix,
        render_panel,
        sample_rules,
    )
    from .local_raven.rules import Rule_Wrapper
    RAVEN_AVAILABLE = True
except Exception as e:
    print(f"Warning: Local RAVEN core not available: {e}")
    RAVEN_AVAILABLE = False
    IMAGE_SIZE = 160


class RavenGenerator:
    """Self-contained RAVEN Progressive Matrix task generator."""
    
    # Configuration mapping - FOCUS on Python 3 compatible configurations
    # Note: RAVEN was designed for Python 2.7, so some configurations have compatibility issues
    CONFIGURATIONS = {
        "Center": "center_single",          # ✅ Most reliable
        "2x2Grid": "distribute_four",       # ⚠️ Some success  
        "3x3Grid": "distribute_nine"        # ⚠️ Some success
    }
    
    # (No local rule classification; use submodule behavior directly)
    
    def __init__(self):
        """Initialize RAVEN generator with configurations."""
        self.generated_tasks = []
        self.setup_configurations()
        
    def setup_configurations(self):
        """Setup RAVEN configuration trees."""
        if not RAVEN_AVAILABLE:
            raise RuntimeError("RAVEN submodule is not available. Please initialize and install the RAVEN submodule to generate tasks.")
            
        # FOCUS on Python 3 compatible configurations only
        # Note: Other RAVEN configurations have Python 2/3 compatibility issues
        self.config_trees = {
            "center_single": build_center_single(),        # Most reliable
            "distribute_four": build_distribute_four(),    # Some success
            "distribute_nine": build_distribute_nine()     # Some success
        }
        
    def generate_single_task(self, config_name: str, difficulty: str = None) -> Dict[str, Any]:
        """Generate a single RAVEN task."""
        if not RAVEN_AVAILABLE:
            raise RuntimeError("RAVEN submodule is not available. Cannot generate tasks without it.")
        
        # Get configuration tree
        if config_name not in self.config_trees:
            raise ValueError(f"Unknown configuration: {config_name}")
            
        root = self.config_trees[config_name]
        
        # Sample rules for this configuration
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                rule_groups = sample_rules()
                new_root = root.prune(rule_groups)
                
                if new_root is not None:
                    panels = self.generate_panels(new_root, rule_groups)
                    return {
                        "config_name": config_name,
                        "config_display": [k for k, v in self.CONFIGURATIONS.items() if v == config_name][0],
                        "matrix": panels,
                        "attempts": attempt + 1,
                    }
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {config_name}: {e}")
                continue
                
        raise RuntimeError(f"Failed to generate valid task for {config_name} after {max_attempts} attempts")
    
    def generate_panels(self, root, rule_groups) -> List[np.ndarray]:
        """Generate the 9 panels of a Progressive Matrix following RAVEN logic."""
        import copy

        start_node = root.sample()

        def build_row(base_node):
            to_merge = None
            for group_idx, rule_group in enumerate(rule_groups):
                rule_num_pos = rule_group[0]
                c2 = rule_num_pos.apply_rule(base_node)
                c3 = rule_num_pos.apply_rule(c2)
                for rule in rule_group[1:]:
                    c2 = rule.apply_rule(base_node, c2)
                    c3 = rule.apply_rule(c2, c3)
                if group_idx == 0:
                    to_merge = [copy.deepcopy(base_node), c2, c3]
                else:
                    self.merge_component(to_merge[1], c2, group_idx)
                    self.merge_component(to_merge[2], c3, group_idx)
            return to_merge

        row1 = build_row(copy.deepcopy(start_node))
        row2_base = copy.deepcopy(start_node)
        row2_base.resample(True)
        row2 = build_row(row2_base)
        row3_base = copy.deepcopy(start_node)
        row3_base.resample(True)
        row3 = build_row(row3_base)

        nodes = [row1[0], row1[1], row1[2], row2[0], row2[1], row2[2], row3[0], row3[1], row3[2]]
        return [render_panel(node) for node in nodes]
    
    def merge_component(self, dst_aot, src_aot, component_idx):
        """Merge component from src to dst (from RAVEN main.py)."""
        src_component = src_aot.children[0].children[component_idx]
        dst_aot.children[0].children[component_idx] = src_component
    
    # No local rule summarization or difficulty heuristics
    
    def generate_tasks(self, num_tasks: int = 50) -> List[Dict[str, Any]]:
        """Generate tasks; fallback to alternates and mocks succinctly."""
        print(f"🎯 Generating {num_tasks} RAVEN tasks across {len(self.CONFIGURATIONS)} configurations...")
        tasks: List[Dict[str, Any]] = []
        configs = list(self.CONFIGURATIONS.values())
        for i in range(num_tasks):
            cfg = configs[i % len(configs)]
            tasks.append(self.generate_single_task(cfg))
            print(f"✅ {i+1}/{num_tasks}: {tasks[-1]['config_display']}")
        self.generated_tasks = tasks
        return tasks


def generate_task_images(task_data: Dict[str, Any], output_dir: str, task_id: str) -> Tuple[str, str]:
    """
    Generate first and final frame images for a RAVEN task.
    
    Args:
        task_data: Generated task data containing matrix
        output_dir: Base output directory
        task_id: Task identifier for naming
    
    Returns:
        (first_image_path, final_image_path)
    """
    matrix = task_data["matrix"]
    
    # Create temporary files that will be moved to per-question folders
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Image paths
    first_image_path = os.path.join(temp_dir, f"{task_id}_first.png")
    final_image_path = os.path.join(temp_dir, f"{task_id}_final.png")

    # Always compose into a 3x3 RAVEN matrix for consistency
    generate_rpm_image(matrix, first_image_path, incomplete=True)
    generate_rpm_image(matrix, final_image_path, incomplete=False)
    
    # Return temp paths that will be moved by create_dataset.py
    return first_image_path, final_image_path


def generate_rpm_image(matrix_panels: List[np.ndarray], output_path: str, incomplete: bool = False):
    """Render a 3x3 RAVEN matrix image using the local renderer."""
    import numpy as np
    from PIL import Image

    # Prepare exactly 9 panels; fill missing with white
    panels: List[np.ndarray] = []
    take = min(len(matrix_panels), 9)
    if incomplete and take >= 8:
        panels.extend(matrix_panels[:8])
        panels.append(np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255)
    else:
        panels.extend(matrix_panels[:take])
        while len(panels) < 9:
            panels.append(np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255)

    grid = generate_matrix(panels)
    Image.fromarray(grid).save(output_path)


def generate_prompt(task_data: Dict[str, Any]) -> str:
    """Generate concise prompt with config-aware base; no rule-specific hints."""
    config_display = task_data["config_display"]
    base = {
        "Center": "Complete this center-focused pattern matrix",
        "2x2Grid": "Complete this 2x2 grid pattern matrix",
        "3x3Grid": "Complete this 3x3 grid pattern matrix",
    }.get(config_display, "Complete this pattern matrix")
    panel = "4th panel" if config_display == "2x2Grid" else "9th panel" if config_display == "3x3Grid" else "missing panel"
    return f"{base}. Show what goes in the missing {panel}."


def create_task_pair(task_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Create a RAVEN task pair in VMEvalKit format."""
    
    # Generate images
    base_dir = Path(__file__).parent.parent.parent.parent
    first_image_path, final_image_path = generate_task_images(task_data, str(base_dir), task_id)
    
    # Generate prompt  
    prompt = generate_prompt(task_data)
    
    # Create task pair following VMEvalKit structure
    task_pair = {
        "id": task_id,
        "prompt": prompt,
        "first_image_path": first_image_path,
        "final_image_path": final_image_path,
        "task_category": task_data["config_display"],
        "raven_data": {
            "generation_method": "RAVEN Progressive Matrix Generator",
            "configuration": task_data["config_name"],
            "matrix_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "pattern_type": "Progressive Matrix"
        },
        "configuration_type": task_data["config_display"],
        "created_at": datetime.now().isoformat()
    }
    
    return task_pair


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Create Progressive Matrix tasks across multiple configurations for better generation success."""
    
    print(f"🎯 Creating RAVEN Progressive Matrix dataset with {num_samples} samples across 3 configurations...")
    
    # Generate tasks
    generator = RavenGenerator()
    tasks = generator.generate_tasks(num_samples)
    
    if len(tasks) == 0:
        raise RuntimeError("Failed to generate any valid RAVEN tasks")
    
    # Create task pairs
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"raven_{i:04d}"
        
        try:
            pair = create_task_pair(task_data, task_id)
            pairs.append(pair)
            print(f"✅ Created task {task_id}: {pair['task_category']}")
        except Exception as e:
            print(f"❌ Failed to create task pair {task_id}: {e}")
            continue
    
    if len(pairs) == 0:
        raise RuntimeError("Failed to create any valid task pairs")
    
    # Create dataset
    dataset = {
        "name": "raven_tasks",
        "description": f"RAVEN Progressive Matrix tasks across 3 configurations (2x2, 3x3, center) for video reasoning evaluation ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    # Don't save to intermediate folder anymore - will be handled by create_dataset.py
    print(f"📊 Dataset stats:")
    
    # Print statistics
    categories = {}
    for pair in pairs:
        cat = pair['task_category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"   Categories: {categories}")
    
    return dataset


# Dataset creation should only be done via vmevalkit/runner/create_dataset.py
# This module only provides the create_dataset() function as an API
