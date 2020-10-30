## subroutines for implementing the n queens method
## created by: Yu Geng
## 2020-04-29

import multiprocessing
from represent import *     # includes print_function
from cpropagation import *  # includes math, numpy and product

def sel_empty_cell(grid):
    """Traverses through a 9x9 sudoku; returns the index
    of the first empty cell encountered."""
    
    for i,j in product(range(N), range(N)):
        if grid[i][j]:
            pass
        else:
            return i,j

# For optimization, refer to these articles:
#   https://www.mscto.com/game/376675.html
#   https://www.cnblogs.com/grenet/p/3138654.html

def n_queens(grid):
    
    if is_full(grid):
        print("Base case triggered!")
        return grid
    
    else:
        # select a cell and list its candidates
        clues = cache_clues(grid)  # save clues in structures for fast caching
        ti,tj = sel_empty_cell_sorted(clues, grid)
        candidates = lst_candidates_fast(clues, ti,tj)
        
        if candidates:
            
            for c in candidates:
                grid[ti][tj] = c
                sol = n_queens(grid)
                if sol:
                    return sol
                else:
                    grid[ti][tj] = 0  # try resetting it to zero
                    continue
            
            # after trying all candidates,
            # no solution found
            grid[ti][tj] = 0
            return False
        
        else:
            return False

# ----- Constraint Propagation -----

class MyClass(multiprocessing.Process):
    """Encapsulate multi processing methods in a class."""

    def __init__(self, arr, queue):
        super().__init__()
        self.arr = arr
        self.queue = queue

    def calPow(self):
        res = []
        for i in self.arr:
            res.append(i * i)
        self.queue.put(res)

    def run(self):
        self.calPow()

def sel_empty_cell_sorted(clues, grid, reverse=False):
    """Provided a 9x9 Sudoku; returns the indices of
    the empty cell with the smallest number of candidates."""
    
    # traverse through the grid
    # and enumerate candidates
    avail_cells = dict()
    for i,j in product(range(N), range(N)):
        if grid[i][j]:
            pass
        else:
            candidates = \
                lst_candidates_fast(clues, i,j)
            avail_cells[(i,j)] = len(candidates)
    
    if reverse:
        N_target = max(avail_cells.values())
    else:
        N_target = min(avail_cells.values())
    
    # find the one with the
    # smallest/largest number of candidates
    for indices, length in \
        avail_cells.iteritems():
        if length == N_target:
            return indices  # returns the 1st one encountered
        else:
            pass

def n_queens_solve(pzz):
    """Main subroutine for solving resursively
    with constraint propagation included."""
    
    # unpack the puzzle
    grid = pzz.get_puzzle()
    
    # iteratively fill out singletons
    # until there is not any
    fillout_singles(grid)
    
    # check if it is already solved
    if is_full(grid):
        print("The grid is full! Exiting...")
    else:
        print("Calling recursively...")
        n_queens(grid)  # this will mutate grid
    
    # present
    sdk = Sudoku(grid)
    sdk.show()

