import os
import torch
from solver import Solver
from option import get_option


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    
    solver = Solver(opt)
    solver.fit()
    
if __name__ == "__main__":
    main()