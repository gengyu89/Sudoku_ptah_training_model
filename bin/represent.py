## subroutines for loading and displaying a sudoku
## created by: Yu Geng
## 2020-04-29

from __future__ import print_function

# define size of the sudoku
# as global constants
R, C = 3, 3
N = R * C

"""Whatever method that attempts to fetch data
from a Sudoku object, they have to access it
with get_puzzle(); get_puzzle() will create a new copy
of __grid; your algorithm does not mutate the object."""

# The workflow is:
#   1.load_puzzle() returns pzz <type 'list'>
#   2.Use Sudoku(pzz) to create pzz_e
#   3.In each method, unpack the puzzle by using
#     grid = pzz.get_puzzle()
#   4.Use Sudoku(sol) to create sdk and show it

# ----- The Sudoku class -----

from copy import deepcopy
from itertools import product

def show_hbeam():
    
    beam = '-' * 5
    hbeam = '%s + %s + %s' % \
        (beam, beam, beam)
    
    print('* %s *' % hbeam)

def map_entry(n):
    
    if n:
        return '%d' % n
    else:
        return ' '

def show_stringers(row):
    
    assert len(row) == N, \
           "Invalid puzzle width!"
    
    row = map(map_entry, row)
    
    print("|", end=' ')
    for a in range(C):
        for c in range(3*a, 3*a+3):
            print('%s' % row[c], end=' ')
        print("|", end=' ')
    
    print("")

class Sudoku(object):
    
    def __init__(self, pzz):
        self.__grid = pzz
    
    def get_puzzle(self):
        return deepcopy(self.__grid)
    
    def show(self):
        """__str__() and __repr__()
        can only return strings."""
        # grid can be either a 2d list
        #   or an np array
        
        assert len(self.__grid) == N, \
               "Invalid puzzle height!"
        
        show_hbeam()
        for a in range(R):
            for r in range(3*a, 3*a+3):
                show_stringers(self.__grid[r])
            show_hbeam()
        
        print("")
    
    def is_correct(self):
        """Subroutine for checking the correctness
        of a solution from CNN."""
        
        sdk = np.array(self.__grid)
        
        # traverse through rows
        for i in range(N):
            row = sdk[i,:]
            N_miss = N - len(set(row))
            if N_miss:
                return False
            else:
                continue
        
        # traverse through columns
        for j in range(N):
            col = sdk[:,j]
            N_miss = N - len(set(col))
            if N_miss:
                return False
            else:
                continue
        
        # traverse through boxes
        for a,b in product(range(R), range(C)):
            
            i_range = range(3*a, 3*a+3)
            j_range = range(3*b, 3*b+3)
            
            box = []
            for i,j in product(i_range, j_range):
                box.append(sdk[i,j])  # collect current box members
            N_miss = N - len(set(box))
            
            if N_miss:
                return False
            else:
                continue
        
        return True  # in case no duplicate is found
    
# ----- Subroutines for visiting -----

import numpy as np

def load_puzzle(filename):
    """Nothing special just the
    numpy version of load_sdk()."""
    
    # tell people which problem you're solving
    print("Loading", filename)
    
    sdk = np.loadtxt(filename, dtype=int)
    assert (N,N) == np.shape(sdk), \
           'Invalid input: %s!' % filename
    
    # convert back to a 2d list
    pzz = sdk.tolist()
    N_clues = count_clues(pzz)
    print('%4d clues given.' % N_clues)
    
    return pzz

def check_validity(pzz):
    """This is not a function for checking
    the correctness of a solution."""
    total_entries = 0
    
    for row in pzz:
        non_zeros = filter(lambda e:e, row)
        N_entries = len(non_zeros)
        total_entries += N_entries
    
    return total_entries <= 17

def count_clues(pzz):
    """A modification of check_validity()."""
    total_entries = 0
    
    for row in pzz:
        non_zeros = filter(lambda e:e, row)
        N_entries = len(non_zeros)
        total_entries += N_entries
    
    return total_entries

# ----- Other functions that are essential -----

import os
import time

def show_time(t):
    
    print('Elapsed time is %f seconds.\n' \
        % (time.time() - t))

def check_duplicate(folder):
    # folder <type 'str'>
    #   - must include a slash
    
    # enumerate all grid files
    grids = []
    f_all = [f for f in os.listdir(folder) if f.endswith('.grid')]
    for f in f_all:
        fullpath = folder + f
        grid = np.loadtxt(fullpath, dtype=int)
        grids.append(grid)
    
    # store their indices
    N_grd = len(grids)
    duplicates = []
    
    print("Searching for duplicates...")
    
    # an o(n^2) implementation
    for i, grid_a in enumerate(grids):
        for j in range(i+1,N_grd):
            grid_b = grids[j]
            if np.array_equal(grid_a, grid_b):
                duplicates.append((i,j))
            else:
                pass
    
    # visit f_all with the indices
    if duplicates:
        print("Duplicated puzzles:")
        for i,j in duplicates:
            print(f_all[i], f_all[j])
    else:
        print("There are no duplicated puzzles.")
    
    print("")

# class Timer:
#     """Works in Python 3, a Timer class
#     that can be used within a context manager."""
    
#     def __init__(self, func=time.perf_counter):
#         self.elapsed = 0.0
#         self._func = func
#         self._start = None
    
#     def start(self):
#         if self._start is not None:
#             raise RuntimeError('Already started')
#         else:
#             pass
#         self._start = self._func()
    
#     def stop(self):
#         if self._start is None:
#             raise RuntimeError('Not started')
#         else:
#             pass
#         end = self._func()
#         self.elapsed += end - self._start
#         self._start = None
    
#     def reset(self):
#         self.elapsed = 0.0
    
#     @property
#     def running(self):
#         return self._start is not None
    
#     def __enter__(self):
#         self.start()
#         return self
    
#     def __exit__(self, *args):
#         self.stop()
#         print('Elapsed time is %f seconds.\n' \
#             % self.elapsed)

# Here are several ways you can invoke it
# 
# t = Timer()
# with t:
#     # run something
# 
# with Timer() as t:
#     # run something
# 
# You do not even have to
#   provide it a variable name, such as
# 
# with Timer():
#     # run something
# 
