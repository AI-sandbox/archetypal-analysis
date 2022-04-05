import argparse
import sys
from src.run_pipeline import run_pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True, type=str, help='Defines the input file / path. File must be in vcf, bed, pgen or npy format. If npy, assumes data is already projected')
    parser.add_argument('-o', '--output_file', required=True, type=str, help='Defines the output file / path. File name does not need any extensions')
    parser.add_argument('-k', '--n_archetypes', required=True, type=int, help='Defines the number of archetypes')
    parser.add_argument('--tolerance', required=False, type=float, default=0.001, help='Defines when to stop optimization')
    parser.add_argument('--max_iter', required=False, type=int, default=200, help='Defines the maximum number of iterations')
    parser.add_argument('--random_state', required=False, type=int, default=0, help='Defines the RNG seed')
    parser.add_argument('-C', '--constraint_coef', required=False, type=float, default=0.0001, help='''C is a constraint coefficient to ensure that the summation ofalfa's and beta's equals to 1. C is considered to be inverse of M^2 in the original paper''')
    parser.add_argument('--initialize', required=False, type=str, default='furthest_sum', choices=['furthest_sum', 'random', 'random_idx'], help='''
                    Defines the initialization method to guess initial archetypes:
                        1. furthest_sum (Default): the idea and code taken from https://github.com/ulfaslak/py_pcha and the original author is: Ulf Aslak Jensen.
                        2. random:  Randomly selects archetypes in the feature space. The points could be any point in space
                        3. random_idx:  Randomly selects archetypes from points in the dataset.''')    
    parser.add_argument('--redundancy_try', required=False, type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    status = run_pipeline(
        input_file=args.input_file,
        output_file=args.output_file,
        n_archetypes=args.n_archetypes,
        tolerance=args.tolerance,
        max_iter=args.max_iter,
        random_state=args.random_state,
        C=args.constraint_coef,
        initialize=args.initialize,
        redundancy_try=args.redundancy_try
    )
    sys.exit(status)