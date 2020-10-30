## subroutines for implementing constraint propagations
## used by both permutation search and n queens
## 2020-05-18

from __future__ import print_function

import math
import numpy as np
from itertools import product

# define size of the sudoku
# as global constants
R, C = 3, 3
N = R * C

# In each iteration, the work flow is:
#   Enumerate clues -> Apply Rule I ->
#   Enumerative candidates -> Apply Rule II

def is_full(grid):
    """Determining the completeness of
    a Sudoku grid in numpy manner."""
    mask = (np.array(grid)==0)
    
    return not mask.sum()

def cache_clues(grid):
    """Pre process a 9x9 Sudoku grid
    and cache existing numbers."""
    
    clues_row = dict()
    clues_col = dict()
    clues_box = dict()
    sdk_arr = np.array(grid)
    
    for i in range(N):
        row = sdk_arr[i,:]
        clues = filter(lambda e:e, row.tolist())
        clues_row[i] = clues
    
    for j in range(N):
        col = sdk_arr[:,j]
        clues = filter(lambda e:e, col.tolist())
        clues_col[j] = clues
    
    for m, n in product(range(R), range(C)):
        box = []  # current box members in a list
        i_range = range(3*m, 3*m+3)
        j_range = range(3*n, 3*n+3)
        for i, j in product(i_range, j_range):
            box.append(sdk_arr[i,j])
        clues = filter(lambda e:e, box)
        clues_box[(m,n)] = clues
    
    return (clues_row, clues_col, clues_box)

def lst_candidates_fast(clues, ei, ej):
    """A smarter way of programming
    lst_candidates(), at least I think."""
    
    # find box number from cell indices
    I, J = ei+1, ej+1  # convert into natural indices
    A = math.ceil(I/3.0)
    B = math.ceil(J/3.0)
    m, n = int(A-1), int(B-1)
    
    # fast caching
    clues_row, clues_col, clues_box = clues  # unpack clues
    filled_row = clues_row[ei]
    filled_col = clues_col[ej]
    filled_box = clues_box[(m,n)]
    
    # remove existing numbers
    filled_all = set(filled_row + filled_col + filled_box)
    candidates = range(1, N+1)
    for f in filled_all:
        candidates.remove(f)
    
    return candidates

# Although this version of lst_candidates() is faster,
# you have to be careful when using it. (clues) must be
# calculated in real time, not be used globally.

def check_peers(peers):
    """Subroutine for checking if there is
    a missing candidate from all peers; returns
    always a list even when there is no element in it."""
    
    # collect choices from cached dictionary
    union = set()
    for candidates in peers.values():
        for c in candidates:
            union.add(c)
    
    # check if there is a missing
    # candidate for all its peers
    whole = set(range(1,N+1))
    missing = whole.difference(union)
    
    smsg_1 = "Rule II of constraint propagation "
    smsg_2 = "yields at most one missing element!"
    assert len(missing) < 2, smsg_1 + smsg_2
    
    return list(missing)

def lst_candidates_cp(rule_1, grid, ei,ej):
    """Apply constraint propagation rule ii
    to a cell specified by indices (ei,ej); returns
    either a list containing one element or an empty list."""
    # rule_1 <type 'dict'>
    #   - indices of empty cells and candidates
    #     pre-calculated and cached
    
    # obtain range of indices for searching
    I, J = ei+1, ej+1  # convert into natural indices
    A = math.ceil(I/3.0)
    B = math.ceil(J/3.0)
    a, b = int(A-1), int(B-1)  # box number in python convention
    
    # use the box number to find indices
    i_range = range(3*a, 3*a+3)
    j_range = range(3*b, 3*b+3)
    
    # apply rule ii in the same box
    peers_box = dict()
    for i, j in product(i_range, j_range):
        if grid[i][j]:
            peers_box[(i,j)] = [grid[i][j]]
        elif i==ei and j==ej:
            pass
        else:
            candidates = rule_1[(i,j)]
            peers_box[(i,j)] = candidates
    
    # check if you can determine the answer
    missing = check_peers(peers_box)
    if missing:
        return missing
    else:
        pass  # continue checking row and column
    
    # apply rule ii in the same row
    peers_row = dict()
    for j in range(N):
        if grid[ei][j]:
            peers_row[(ei,j)] = [grid[ei][j]]
        elif j==ej:
            pass
        else:
            candidates = rule_1[(ei,j)]
            peers_row[(ei,j)] = candidates
    
    # check if you can determine the answer
    missing = check_peers(peers_row)
    if missing:
        return missing
    else:
        pass  # continue checking column
    
    # apply rule ii in the same column
    peers_col = dict()
    for i in range(N):
        if grid[i][ej]:
            peers_col[(i,ej)] = [grid[i][ej]]
        elif i==ei:
            pass
        else:
            candidates = rule_1[(i,ej)]
            peers_col[(i,ej)] = candidates
    
    # check if you can determine the answer
    missing = check_peers(peers_col)
    if missing:
        return missing
    else:
        return list()  # the current cell is not a singleton
    
    # note that existing numbers
    # must be treated as "possibilities"

def find_singletons(grid):
    """Provided a 9x9 Sudoku grid, returns the indices and
    the answer for cells that have exactly one candidate."""
    
    # enumerate existing clues and cache them
    clues = cache_clues(grid)
    singles = dict()
    
    # apply constraint propagation
    # rule i and cache them
    rule_1 = dict()
    for i, j in product(range(N), range(N)):
        if grid[i][j]:
            pass
        else:
            candidates = lst_candidates_fast(clues, i,j)
            rule_1[(i,j)] = candidates
    
    # preview output from rule 1
    # print("Rule I:")
    # for k,v in rule_1.iteritems():
    #     print(k,v)
    
    # collect singletons from rule 1
    for k,v in rule_1.iteritems():
        if len(v) == 1:
            singles[k] = v[0]
        else:
            pass
    
    # apply constraint propagation
    # rule ii and cache them
    rule_2 = dict()
    for i, j in product(range(N), range(N)):
        if grid[i][j]:
            pass
        else:
            missing = lst_candidates_cp(rule_1, grid, i,j)
            rule_2[(i,j)] = missing
    
    # preview output from rule 2
    # print("Rule II:")
    # for k,v in rule_2.iteritems():
    #     print(k,v)
    
    # collect singletons from rule 2
    for k,v in rule_2.iteritems():
        if len(v) == 1:
            singles[k] = v[0]
        else:
            pass
    
    # preview the merged dictionary
    # print("Merged:")
    # for k,v in singles.iteritems():
    #     print(k,v)
    
    return singles

def fill_out(grid, sol):
    """A subroutine that fills in answers
    by mutating the grid structure."""
    # sol - a dictionary that stores singletons
    #   or any solutions to be verified
    
    for (i,j), v in sol.iteritems():
        grid[i][j] = v

def fillout_singles(grid):
    """A subroutine that repeatedly fills out cells
    that have determined answers until there is not any."""
    # either an np array or a 2d list,
    # grid is a mutable structure
    
    print("Searching for singletons iteratively...")
    singles = find_singletons(grid)
    
    if singles == dict():
        print("No singleton found!")
    else:
        while singles != dict():
            print('%4d singleton(s) found!' % len(singles))
            fill_out(grid, singles)
            singles = find_singletons(grid)
    
    print("")

