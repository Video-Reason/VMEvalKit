"""
Batch generator for RPM-style puzzles.

Usage:
    python batch_generate.py --n 20 --out ./dataset --seed 2025 --solutions
"""

import argparse
import os
from rpm_generator import generate_puzzle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="Number of puzzles")
    ap.add_argument("--out", type=str, default="dataset", help="Output directory")
    ap.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    ap.add_argument("--tile_size", type=int, default=192)
    ap.add_argument("--solutions", action="store_true", help="Also save solution sheets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        seed_i = None if args.seed is None else (args.seed + i)
        out_dir = os.path.join(args.out, f"puzzle_{i:04d}")
        meta = generate_puzzle(
            out_dir=out_dir,
            seed=seed_i,
            tile_size=args.tile_size,
            save_solution_sheet=args.solutions
        )
        print(f"[{i+1}/{args.n}] -> {out_dir} | correct={meta['correct_option_index']}")

if __name__ == "__main__":
    main()
