"""Centralized prompts for Sliding Puzzle Task."""

# Single prompt with dynamic move count and constraints
PROMPTS = [
    "Complete this sliding puzzle in exactly {num_moves} move{plural}. "
    "Move one tile per move horizontally or vertically into the empty space. "
    "Do not make extra moves. "
    "Keep the camera view fixed and maintain the grid structure unchanged.",
]

DEFAULT_PROMPT_INDEX = 0

