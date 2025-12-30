"""
Prompts for ARC (Abstraction and Reasoning Corpus) Tasks

This file centralizes all prompts used for ARC reasoning tasks.
Modify prompts here to experiment with different instruction styles.
"""

# Standardized prompts for ARC tasks
PROMPTS = [
    "Observe the input grid and apply the transformation rule to produce the output grid.",
    "Analyze the pattern in the input and show the correct output transformation.",
    "Study the input grid carefully and demonstrate the logical transformation to the output.",
    "Identify the transformation rule and apply it to transform the input grid to the output.",
    "Look at the input pattern and show how it should be transformed based on the underlying rule.",
]

# Default prompt index to use
DEFAULT_PROMPT_INDEX = 0

# Task-specific guidance for scoring
SCORING_GUIDANCE = """
Evaluate the ARC task completion:
1. Check if the output grid matches the expected transformation
2. Verify that the correct colors/values are in the correct positions
3. Assess whether the transformation rule was correctly applied
Give a score from 0-10 based on accuracy of the transformation.
"""
