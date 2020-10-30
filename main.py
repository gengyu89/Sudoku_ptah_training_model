## main program for testing subroutines
## created by: Yu Geng
## 2020-04-29

from __future__ import print_function

from bin.pmprocess import *   # permutation search
from bin.nqueens import *     # back tracking
from bin.logcircuit import *  # boolean satisfiability
from bin.dancinlinks import * # dancing links

# ----- Select a problem -----

# when new puzzles are added
# search for duplicates in this directory
# check_duplicate('./input/puzzles/')

# easy (30~37 clues) 
# pzz = load_puzzle('./input/puzzles/shawn_lee.grid')  # two solutions that look super close
# pzz = load_puzzle('./input/puzzles/csdn_example.grid')
# pzz = load_puzzle('./input/puzzles/opensourc_es.grid')
# pzz = load_puzzle('./input/puzzles/sector_f.grid')
# pzz = load_puzzle('./input/puzzles/kwon_and_jain.grid')

# medium (24~28 clues)
# pzz = load_puzzle('./input/puzzles/sinkhorns_algorithm.grid')
# pzz = load_puzzle('./input/puzzles/sudoku_mp.grid')
# pzz = load_puzzle('./input/puzzles/rohit_shreekant.grid')
# pzz = load_puzzle('./input/puzzles/chiu_et_al.grid')
# pzz = load_puzzle('./input/puzzles/olszowy_wiktor.grid')  # this problem has five solutions
# pzz = load_puzzle('./input/puzzles/opensourc_harder.grid')
# pzz = load_puzzle('./input/puzzles/blogs_sas.grid')
# pzz = load_puzzle('./input/puzzles/syed_and_merugu.grid')

# difficult (21~22 clues)
# pzz = load_puzzle('./input/puzzles/harrysson_paper.grid')
# pzz = load_puzzle('./input/puzzles/mathworks.grid')
# pzz = load_puzzle('./input/puzzles/nicolae_ileana.grid')
# pzz = load_puzzle('./input/puzzles/platinum_blonde.grid')

# tough (17 clues)
# pzz = load_puzzle('./input/puzzles/hamiltonian_cycle.grid')
pzz = load_puzzle('./input/puzzles/wikipedia.grid')
# pzz = load_puzzle('./input/puzzles/lynce_paper.grid')

# cnblogs all levels
# pzz = load_puzzle('./input/puzzles/cnblogs_easy.grid')
# pzz = load_puzzle('./input/puzzles/cnblogs_medium.grid')  # this problem has over 254 solutions
# pzz = load_puzzle('./input/puzzles/cnblogs_difficult.grid')  # Arto Inkala's problem

# ----- The main function -----

if __name__ == '__main__':
    """Display and solve."""

    # display the original puzzle
    pzz_e = Sudoku(pzz)  # create an object for data protection
    pzz_e.show()

    # each method accesses __grid via get_puzzle()
    # the original puzzle is not modified

    t = time.time()
    pb_force_solve(pzz_e, None)
    show_time(t)

    t = time.time()
    n_queens_solve(pzz_e)
    show_time(t)

    t = time.time()
    sat_main(pzz_e)
    show_time(t)

    t = time.time()
    excover_sdk_solve(pzz_e)
    show_time(t)

    # note that the training model
    # must be specified in cnnsolver.py

    # from bin.cnnsolver import *

    # t = time.time()
    # cnn_main(pzz_e)
    # show_time(t)

