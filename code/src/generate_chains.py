"""
Generate and save the ultrametric Markov chains used by run.py (replaces the
"Generate sequ to save" cell of notebooks/ultrametric_chain.ipynb).

Chains are seeded with numpy Generators, so re-running this script reproduces
exactly the same files.

Usage: python generate_chains.py --datafolder ../../data --depths 3 5 6 7 8 --seeds 0 1 2
"""
import argparse
import os
import sys
from os.path import join

import numpy as np

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, join(curr_path, "../utils/"))
from functions_markov import generate_markov_chain

# Temperatures used in the thesis (notebooks/ultrametric_chain.ipynb, T_fixed)
TEMPERATURE_PER_DEPTH = {3: 0.25, 5: 0.4, 6: 0.45, 7: 0.475, 8: 0.55}
CHAIN_LENGTH = 1000000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=str, default='../../data')
    parser.add_argument('--depths', type=int, nargs='+', default=[5, 6, 7, 8, 3])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(50)))
    args = parser.parse_args()

    out_dir = join(args.datafolder, 'saved_chains')
    os.makedirs(out_dir, exist_ok=True)
    for depth in args.depths:
        for seed in args.seeds:
            chain = generate_markov_chain(chain_length=CHAIN_LENGTH, T=TEMPERATURE_PER_DEPTH[depth],
                                          tree_levels=depth, dia=0, seed=seed)
            if np.max(chain) < 2**depth - 1:
                print(f"WARNING: not all leaves visited for depth {depth}, seed {seed}")
            np.save(join(out_dir, f"tree_levels{depth:02d}_clen{CHAIN_LENGTH:.1e}_seed{seed}"), chain)
            print(f"saved depth {depth} seed {seed}")


if __name__ == '__main__':
    main()
