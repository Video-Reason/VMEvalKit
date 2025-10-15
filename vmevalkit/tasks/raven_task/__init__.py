"""
Raven Progressive Matrices (RPM) task module for VMEvalKit.

This module provides functionality to generate RPM-style puzzles
for evaluating video model reasoning capabilities.
"""

from .rpm_generator import RPMPuzzleGenerator, generate_puzzle
from .batch_generate import main as batch_generate

__all__ = [
    'RPMPuzzleGenerator',
    'generate_puzzle', 
    'batch_generate'
]

# Task metadata
TASK_INFO = {
    'name': 'Raven Progressive Matrices',
    'description': 'Generate and solve RPM-style visual reasoning puzzles',
    'version': '2.0.0',
    'capabilities': [
        'pattern_recognition',
        'logical_reasoning',
        'spatial_reasoning',
        'abstract_thinking'
    ]
}