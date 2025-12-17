"""
ARC (Abstraction and Reasoning Corpus) Task Module for VMEvalKit

This module provides ARC reasoning tasks for evaluating video models' ability to:
- Understand abstract visual patterns from grid-based input
- Identify transformation rules between input and output grids
- Apply pattern recognition and logical reasoning
- Demonstrate solutions through generated video

The ARC tasks test abstract reasoning, pattern recognition, and rule inference
capabilities in video models.
"""

from .arc_reasoning import create_dataset

__all__ = ['create_dataset']
