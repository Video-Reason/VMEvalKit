#!/usr/bin/env python3
"""
ARC (Abstraction and Reasoning Corpus) Task for VMEvalKit

Self-contained ARC task generation system that creates abstract reasoning puzzles.
Follows the same data format as other VMEvalKit tasks with first/final frames and prompts.

Author: VMEvalKit Team
"""

import random
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from PIL import Image, ImageDraw

from .PROMPTS import PROMPTS, DEFAULT_PROMPT_INDEX


# ARC color palette (standard 10 colors)
ARC_COLORS = {
    0: (0, 0, 0),        # Black (background)
    1: (0, 116, 217),    # Blue
    2: (255, 65, 54),    # Red
    3: (46, 204, 64),    # Green
    4: (255, 220, 0),    # Yellow
    5: (170, 170, 170),  # Gray
    6: (240, 18, 190),   # Magenta
    7: (255, 133, 27),   # Orange
    8: (127, 219, 255),  # Light Blue
    9: (135, 12, 37),    # Maroon
}


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """
    Create ARC reasoning task dataset.
    
    Args:
        num_samples: Number of task pairs to generate
    
    Returns:
        Dictionary containing 'name' and 'pairs' list
    """
    print(f"🎯 Generating {num_samples} ARC reasoning tasks...")
    
    pairs = []
    generator = ARCTaskGenerator()
    
    for i in range(num_samples):
        task_id = f"arc_{i:04d}"
        
        # Generate a random ARC task
        task_data = generator.generate_task()
        
        # Create task pair with images
        pair = create_arc_task_pair(task_data, task_id)
        pairs.append(pair)
        
        if (i + 1) % 10 == 0:
            print(f"  ↳ Generated {i + 1}/{num_samples} tasks...")
    
    print(f"✅ Generated {len(pairs)} ARC task pairs")
    
    return {
        "name": "arc_tasks",
        "pairs": pairs
    }


class ARCTaskGenerator:
    """Generator for various ARC-style transformation tasks."""
    
    def __init__(self):
        self.task_types = [
            self._generate_color_swap_task,
            self._generate_fill_pattern_task,
            self._generate_mirror_task,
            self._generate_rotate_task,
            self._generate_scale_task,
            self._generate_translate_task,
            self._generate_count_and_fill_task,
            self._generate_border_task,
            self._generate_complete_pattern_task,
            self._generate_extract_shape_task,
        ]
    
    def generate_task(self) -> Dict[str, Any]:
        """Generate a random ARC task."""
        task_func = random.choice(self.task_types)
        return task_func()
    
    def _create_empty_grid(self, height: int, width: int) -> List[List[int]]:
        """Create an empty grid filled with zeros (black)."""
        return [[0 for _ in range(width)] for _ in range(height)]
    
    def _generate_color_swap_task(self) -> Dict[str, Any]:
        """Generate a task where two colors need to be swapped."""
        height, width = random.randint(5, 10), random.randint(5, 10)
        input_grid = self._create_empty_grid(height, width)
        
        # Add random colored cells
        color1, color2 = random.sample(range(1, 10), 2)
        num_cells = random.randint(5, 15)
        
        for _ in range(num_cells):
            y, x = random.randint(0, height-1), random.randint(0, width-1)
            input_grid[y][x] = random.choice([color1, color2])
        
        # Create output by swapping colors
        output_grid = [[color2 if cell == color1 else (color1 if cell == color2 else cell) 
                       for cell in row] for row in input_grid]
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "color_swap",
            "description": f"Swap color {color1} with color {color2}",
            "difficulty": "easy"
        }
    
    def _generate_fill_pattern_task(self) -> Dict[str, Any]:
        """Generate a task where enclosed regions need to be filled."""
        height, width = random.randint(7, 12), random.randint(7, 12)
        input_grid = self._create_empty_grid(height, width)
        
        # Draw a rectangle border
        border_color = random.randint(1, 9)
        fill_color = random.randint(1, 9)
        while fill_color == border_color:
            fill_color = random.randint(1, 9)
        
        y1, x1 = random.randint(1, height//3), random.randint(1, width//3)
        y2, x2 = random.randint(2*height//3, height-2), random.randint(2*width//3, width-2)
        
        # Draw border
        for x in range(x1, x2+1):
            input_grid[y1][x] = border_color
            input_grid[y2][x] = border_color
        for y in range(y1, y2+1):
            input_grid[y][x1] = border_color
            input_grid[y][x2] = border_color
        
        # Create output with filled interior
        output_grid = [row[:] for row in input_grid]
        for y in range(y1+1, y2):
            for x in range(x1+1, x2):
                output_grid[y][x] = fill_color
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "fill_pattern",
            "description": f"Fill enclosed rectangle with color {fill_color}",
            "difficulty": "medium"
        }
    
    def _generate_mirror_task(self) -> Dict[str, Any]:
        """Generate a horizontal or vertical mirror task."""
        height, width = random.randint(5, 8), random.randint(5, 8)
        input_grid = self._create_empty_grid(height, width)
        
        # Create random pattern on one side
        num_cells = random.randint(5, 12)
        for _ in range(num_cells):
            y = random.randint(0, height-1)
            x = random.randint(0, width//2 - 1)
            input_grid[y][x] = random.randint(1, 9)
        
        # Mirror horizontally
        output_grid = [row[:] for row in input_grid]
        for y in range(height):
            for x in range(width//2):
                output_grid[y][width-1-x] = input_grid[y][x]
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "mirror",
            "description": "Mirror the pattern horizontally",
            "difficulty": "easy"
        }
    
    def _generate_rotate_task(self) -> Dict[str, Any]:
        """Generate a 90-degree rotation task."""
        size = random.randint(5, 8)
        input_grid = self._create_empty_grid(size, size)
        
        # Create random pattern
        num_cells = random.randint(5, 12)
        for _ in range(num_cells):
            y, x = random.randint(0, size-1), random.randint(0, size-1)
            input_grid[y][x] = random.randint(1, 9)
        
        # Rotate 90 degrees clockwise
        output_grid = [[input_grid[size-1-x][y] for x in range(size)] for y in range(size)]
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "rotate",
            "description": "Rotate the pattern 90 degrees clockwise",
            "difficulty": "medium"
        }
    
    def _generate_scale_task(self) -> Dict[str, Any]:
        """Generate a 2x scaling task."""
        small_h, small_w = random.randint(3, 5), random.randint(3, 5)
        input_grid = self._create_empty_grid(small_h, small_w)
        
        # Create small pattern
        num_cells = random.randint(3, 8)
        for _ in range(num_cells):
            y, x = random.randint(0, small_h-1), random.randint(0, small_w-1)
            input_grid[y][x] = random.randint(1, 9)
        
        # Scale 2x
        output_grid = self._create_empty_grid(small_h * 2, small_w * 2)
        for y in range(small_h):
            for x in range(small_w):
                color = input_grid[y][x]
                output_grid[y*2][x*2] = color
                output_grid[y*2+1][x*2] = color
                output_grid[y*2][x*2+1] = color
                output_grid[y*2+1][x*2+1] = color
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "scale",
            "description": "Scale the pattern by 2x",
            "difficulty": "medium"
        }
    
    def _generate_translate_task(self) -> Dict[str, Any]:
        """Generate a translation/movement task."""
        height, width = random.randint(6, 10), random.randint(6, 10)
        input_grid = self._create_empty_grid(height, width)
        
        # Create a small shape
        shape_color = random.randint(1, 9)
        start_y, start_x = random.randint(0, 2), random.randint(0, 2)
        shape_h, shape_w = random.randint(2, 3), random.randint(2, 3)
        
        for dy in range(shape_h):
            for dx in range(shape_w):
                if start_y + dy < height and start_x + dx < width:
                    input_grid[start_y + dy][start_x + dx] = shape_color
        
        # Translate
        move_y, move_x = random.randint(2, 4), random.randint(2, 4)
        output_grid = self._create_empty_grid(height, width)
        
        for dy in range(shape_h):
            for dx in range(shape_w):
                new_y, new_x = start_y + dy + move_y, start_x + dx + move_x
                if 0 <= new_y < height and 0 <= new_x < width:
                    output_grid[new_y][new_x] = shape_color
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "translate",
            "description": f"Move the shape by ({move_y}, {move_x})",
            "difficulty": "easy"
        }
    
    def _generate_count_and_fill_task(self) -> Dict[str, Any]:
        """Generate a task where counting objects determines output."""
        height, width = random.randint(6, 10), random.randint(6, 10)
        input_grid = self._create_empty_grid(height, width)
        
        # Place random colored dots
        dot_color = random.randint(1, 9)
        num_dots = random.randint(3, 7)
        positions = []
        
        for _ in range(num_dots):
            y, x = random.randint(0, height-1), random.randint(0, width-1)
            input_grid[y][x] = dot_color
            positions.append((y, x))
        
        # Output: create a bar showing the count
        output_grid = self._create_empty_grid(height, width)
        for i in range(min(num_dots, width)):
            output_grid[height-1][i] = dot_color
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "count_and_fill",
            "description": f"Count dots and create a bar of length {num_dots}",
            "difficulty": "hard"
        }
    
    def _generate_border_task(self) -> Dict[str, Any]:
        """Generate a task to add border around non-zero cells."""
        height, width = random.randint(7, 10), random.randint(7, 10)
        input_grid = self._create_empty_grid(height, width)
        
        # Create a shape in the center
        shape_color = random.randint(1, 9)
        border_color = random.randint(1, 9)
        while border_color == shape_color:
            border_color = random.randint(1, 9)
        
        center_y, center_x = height // 2, width // 2
        shape_size = random.randint(2, 3)
        
        for dy in range(-shape_size//2, shape_size//2 + 1):
            for dx in range(-shape_size//2, shape_size//2 + 1):
                y, x = center_y + dy, center_x + dx
                if 0 <= y < height and 0 <= x < width:
                    input_grid[y][x] = shape_color
        
        # Add border
        output_grid = [row[:] for row in input_grid]
        for y in range(height):
            for x in range(width):
                if input_grid[y][x] == 0:
                    # Check if adjacent to shape
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if input_grid[ny][nx] == shape_color:
                                output_grid[y][x] = border_color
                                break
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "border",
            "description": f"Add border of color {border_color} around the shape",
            "difficulty": "medium"
        }
    
    def _generate_complete_pattern_task(self) -> Dict[str, Any]:
        """Generate a pattern completion task."""
        size = random.randint(6, 8)
        full_grid = self._create_empty_grid(size, size)
        
        # Create a repeating pattern
        pattern_color = random.randint(1, 9)
        pattern_size = 2
        
        for y in range(size):
            for x in range(size):
                if (y + x) % pattern_size == 0:
                    full_grid[y][x] = pattern_color
        
        # Remove some cells for input
        input_grid = [row[:] for row in full_grid]
        num_remove = random.randint(3, 6)
        for _ in range(num_remove):
            y, x = random.randint(0, size-1), random.randint(0, size-1)
            input_grid[y][x] = 0
        
        return {
            "input_grid": input_grid,
            "output_grid": full_grid,
            "task_type": "complete_pattern",
            "description": "Complete the repeating pattern",
            "difficulty": "hard"
        }
    
    def _generate_extract_shape_task(self) -> Dict[str, Any]:
        """Generate a task to extract a specific colored shape."""
        height, width = random.randint(8, 12), random.randint(8, 12)
        input_grid = self._create_empty_grid(height, width)
        
        # Add multiple colored shapes
        target_color = random.randint(1, 9)
        other_colors = [c for c in range(1, 10) if c != target_color]
        
        # Add target shape
        target_positions = []
        start_y, start_x = random.randint(1, height-4), random.randint(1, width-4)
        for dy in range(random.randint(2, 3)):
            for dx in range(random.randint(2, 3)):
                y, x = start_y + dy, start_x + dx
                input_grid[y][x] = target_color
                target_positions.append((y, x))
        
        # Add noise shapes
        for _ in range(random.randint(3, 6)):
            y, x = random.randint(0, height-1), random.randint(0, width-1)
            if input_grid[y][x] == 0:
                input_grid[y][x] = random.choice(other_colors)
        
        # Output: only the target shape
        output_grid = self._create_empty_grid(height, width)
        for y, x in target_positions:
            output_grid[y][x] = target_color
        
        return {
            "input_grid": input_grid,
            "output_grid": output_grid,
            "task_type": "extract_shape",
            "description": f"Extract only the color {target_color} shape",
            "difficulty": "medium"
        }


def render_grid_to_image(grid: List[List[int]], cell_size: int = 40) -> Image.Image:
    """
    Render an ARC grid to a PIL Image.
    
    Args:
        grid: 2D list of color indices (0-9)
        cell_size: Size of each cell in pixels
    
    Returns:
        PIL Image object
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    img_width = width * cell_size
    img_height = height * cell_size
    
    img = Image.new('RGB', (img_width, img_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        for x in range(width):
            color_idx = grid[y][x]
            color = ARC_COLORS.get(color_idx, (0, 0, 0))
            
            x1 = x * cell_size
            y1 = y * cell_size
            x2 = x1 + cell_size - 1
            y2 = y1 + cell_size - 1
            
            # Draw filled rectangle
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Draw grid lines
            draw.rectangle([x1, y1, x2, y2], outline=(50, 50, 50), width=1)
    
    return img


def create_arc_task_pair(task_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """
    Create an ARC task pair with first frame (input) and final frame (output).
    
    Args:
        task_data: Dictionary containing input_grid, output_grid, and metadata
        task_id: Unique identifier for this task
    
    Returns:
        Task pair dictionary with all required fields
    """
    # Create temporary directory for images
    temp_dir = Path(tempfile.mkdtemp())
    
    first_frame_path = temp_dir / "first_frame.png"
    final_frame_path = temp_dir / "final_frame.png"
    
    # Render grids to images
    input_img = render_grid_to_image(task_data["input_grid"])
    output_img = render_grid_to_image(task_data["output_grid"])
    
    # Save images
    input_img.save(first_frame_path, "PNG")
    output_img.save(final_frame_path, "PNG")
    
    # Select prompt
    prompt = PROMPTS[DEFAULT_PROMPT_INDEX]
    
    return {
        "id": task_id,
        "prompt": prompt,
        "first_image_path": str(first_frame_path),
        "final_image_path": str(final_frame_path),
        "domain": "arc",
        "task_category": "ARC",
        "difficulty": task_data.get("difficulty", "medium"),
        "arc_data": {
            "task_type": task_data["task_type"],
            "description": task_data["description"],
            "input_grid": task_data["input_grid"],
            "output_grid": task_data["output_grid"]
        },
        "created_at": datetime.now().isoformat()
    }


# For testing
if __name__ == "__main__":
    # Test task generation
    dataset = create_dataset(num_samples=5)
    print(f"\nGenerated {len(dataset['pairs'])} task pairs")
    
    for pair in dataset['pairs'][:3]:
        print(f"\n{pair['id']}:")
        print(f"  Type: {pair['arc_data']['task_type']}")
        print(f"  Description: {pair['arc_data']['description']}")
        print(f"  Difficulty: {pair['difficulty']}")
        print(f"  First frame: {pair['first_image_path']}")
        print(f"  Final frame: {pair['final_image_path']}")
