## subroutines for implementing the parallelized version of
## permutation brute force
## 2020-05-11

import multiprocessing
from represent import *     # includes print_function
from cpropagation import *  # includes numpy and product
from itertools import permutations

# In this approach, a Sudoku grid
#   is divided into three areas:
#   * ----- + ----- + ----- *
#   |       |       |       |
#   |   0   |   0   |   0   |
#   |       |       |       |
#   * ----- + ----- + ----- *
#   |       |       |       |
#   |   1   |   1   |   1   |
#   |       |       |       |
#   * ----- + ----- + ----- *
#   |       |       |       |
#   |   2   |   2   |   2   |
#   |       |       |       |
#   * ----- + ----- + ----- *
# 
# For each box that consists of 3x3 cells,
#   all permutations of an array containing
#   numbers 1~9 after excluding existing clues
#   are generated.
# 
# For each permutation of the current box, a
#   validity check is performed by searching both
#   horizontally and vertically; the permutations
#   that have conflicts with existing clues
#   are excluded. An example is illustrated
#   in this diagram:
#   * ----- + ----- + ----- *
#   | 9 8 7 |       |       |
#   | 6 5 4 |     3 |   8 5 | <--- conflict with 5
#   | 3 2 1 |   2   |       | <--- conflict with 2
#   * ----- + ----- + ----- *
#   |       | 5   7 |       |
#   |     4 |       | 1     |
#   |   9   |       |       |
#   * ----- + ----- + ----- *
#   | 5     |       |   7 3 |
#   |     2 |   1   |       |
#   |       |   4   |     9 |
#   * ----- + ----- + ----- *
#         ^
#         |
# 
#    with 4
# 
# Here is the most important part: for each sub area
#   that consists of 3x9 cells, we must find all valid
#   combinations of the above solutions for the three
#   boxes and save them.
# 
# Subsequently, the global iteration is performed
#   with only valid sub area solutions, which
#   requires less number of validity checks
#   than an elementary brute force search.
# 

def is_valid_box(grid, m,n):
    """Given the number of a box that consists of
    3x3 elements, check its conflicts with existing clues."""
    
    # check validities horizontally
    for i in range(3*m, 3*m+3):
        
        curr_row = []
        for j in range(N):  # extends to the entire matrix
            v = grid[i][j]
            if v:
                curr_row.append(v)
            else:
                pass
        
        N_miss = len(curr_row) - len(set(curr_row))
        if N_miss:
            return False
        else:
            continue
    
    # check validities vertically
    for j in range(3*n, 3*n+3):
        
        curr_col = []  # stores only non zero elements
        for i in range(N):  # extends to the entire matrix
            v = grid[i][j]
            if v:
                curr_col.append(v)
            else:
                pass
        
        N_miss = len(curr_col) - len(set(curr_col))
        if N_miss:
            return False
        else:
            continue
    
    return True

def is_valid_area(grid, s):
    """Given s, the numbering of a sub area,
    returns the validity of the current area."""
    
    for i in range(3*s, 3*s+3):
        curr_row = grid[i]
        N_miss = N - len(set(curr_row))
        if N_miss:
            return False
        else:
            continue
    
    return True

def is_valid_2areas(grid, s1, s2):
    """Subroutine for checking the validity
    of the solution given by the outer two loops."""
    
    # create indices for slicing
    i_range = range(3*s1, 3*s1+3) \
        + range(3*s2, 3*s2+3)
    
    sdk = np.array(grid)
    for j in range(N):
        col = sdk[i_range,j]
        N_miss = len(col) - len(np.unique(col))
        if N_miss:
            return False
        else:
            continue
    
    return True

def is_valid_global(grid):
    """Assuming each sub area yields a
    legal solution, the global validity check is
    performed only in the vertical direction."""
    
    sdk = np.array(grid)
    
    for j in range(N):
        col = sdk[:,j]
        N_miss = N - len(set(col))
        if N_miss:
            return False
        else:
            continue
    
    return True

def find_all_4box(grid, m,n):
    """Given a 9x9 sudoku and a box number (m,n),
    returns all the legal solutions
    as a list of dictionaries."""
    
    # find cell indices from the box number
    i_range = range(3*m, 3*m+3)
    j_range = range(3*n, 3*n+3)
    
    # create a list of candidates by
    # excluding existing numbers
    list_avail_num = range(1,N+1)
    
    for i,j in product(i_range, j_range):
        v = grid[i][j]
        if v:
            list_avail_num.remove(v)
        else:
            pass
    
    sols_box = []  # a list of dictionaries
    
    for avail_num in permutations(list_avail_num):
        
        # create a copy and convert into
        # a list at the same time
        candidates = list(avail_num)
        
        sol = dict()  # for the current permutation
        for i,j in product(i_range, j_range):
            v = grid[i][j]
            if v:
                pass
            else:
                e = candidates.pop()
                sol[(i,j)] = e
                # grid[i][j] = e  # do not fill it in here
        
        # fill in the solution for validity check
        fill_out(grid, sol)
        
        if is_valid_box(grid, m,n):
            sols_box.append(sol)
        else:
            pass
        
        # reset to zero for the next iteration
        reset(grid, sol)
        
        # if you don't do this then
        # v = grid[i][j] is always non zero
    
    smsg_1 = '%5d permutations found' % len(sols_box)
    smsg_2 = ' for Box {}!'.format((m,n))
    print(smsg_1 + smsg_2)
    
    return sols_box

def find_all_4area(grid, s):
    """Given a 9x9 sudoku and the number of a sub area,
    returns all the legal solutions
    as a list of dictionaries."""
    # note that the area number is the same as
    # the 1st index of a box numer, i.e. m == s 
    
    N_valid = 0
    sols_area = []  # a list of dictionaries
    
    # enumerate valid solutions
    # for each box and save them
    sols_box_0 = find_all_4box(grid, s,0)  # do not overwrite s
    sols_box_1 = find_all_4box(grid, s,1)
    sols_box_2 = find_all_4box(grid, s,2)
    
    # traverse through all saved solutions
    for sol_box_0 in sols_box_0:
        fill_out(grid, sol_box_0)
        
        for sol_box_1 in sols_box_1:
            fill_out(grid, sol_box_1)
            
            for sol_box_2 in sols_box_2:
                fill_out(grid, sol_box_2)
                
                # to avoid repeatedly filling out the same
                # solution, place each fill_out()
                # statement outside of next loop
                
                if is_valid_area(grid, s):
                    curr_sol_area = dict(sol_box_0)
                    curr_sol_area.update(sol_box_1)
                    curr_sol_area.update(sol_box_2)
                    
                    sols_area.append(curr_sol_area)
                    N_valid += 1
                else:
                    pass
    
    smsg_1 = 'There are %3d valid solutions' % N_valid
    smsg_2 = ' for Area %d.' % s
    print(smsg_1 + smsg_2)
    print('-' * 40)
    
    return sols_area

def fill_out(grid, sol):
    """Fills out a solution (for a sudoku,
    for a box, or for a sub area) for tests."""
    # sol - <type 'dict'>
    
    for (i,j), v in sol.iteritems():
        grid[i][j] = v
        
    # note that the original matrix is mutated for that
    # in this algorithm, we do not need backtracking

def reset(grid, sol):
    """Clears up the spots specified by a solution."""
    
    for (i,j) in sol.keys():
        grid[i][j] = 0

# ----- Multiprocessing implementations -----

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

def iter_globally_pool(grid, order, \
    sols_iter_0, sols_iter_1, sols_iter_2):
    """Subroutine for parallelization."""
    # grid <type 'numpy.ndarray'>
    #   - must be converted back into a 2d
    #     python list before appending,
    #     because there are issues if you use
    #     numpy inside of a python multiprocess
    
    N_iter = 0
    sols_proc = []
    
    for sol_iter_0 in sols_iter_0:
        fill_out(grid, sol_iter_0)
        
        for sol_iter_1 in sols_iter_1:
            fill_out(grid, sol_iter_1)
            
            if is_valid_2areas(grid, order[0], order[1]):
                pass  # execute the rest of the loop
            else:
                continue  # skip the rest
            
            for sol_iter_2 in sols_iter_2:
                fill_out(grid, sol_iter_2)
                
                N_iter += 1
                if is_valid_global(grid):
                    sols_proc.append(grid.tolist())
                else:
                    pass
    
    return N_iter, sols_proc

def pb_force_pool(grid, N_jbs=None):
    """The Pool version of pb_force();
    the output is a generator."""
    # N_jbs <type 'int' or 'NoneType'>
    #   - set to 0 or None in order
    #     to use multiprocessing.cpu_count()
    
    # iteratively fill out singletons
    # until there is not any
    fillout_singles(grid)
    
    # check if it is already solved
    if is_full(grid):
        print("The grid is full! Exiting...")
        yield grid
        return
    else:
        pass  # execute the rest of the code
    
    print("Searching for valid solutions...")
    
    # enumerate valid solutions
    # for each area and save them
    sols_area_0 = find_all_4area(grid, 0)
    reset(grid, sols_area_0[-1])
    
    sols_area_1 = find_all_4area(grid, 1)
    reset(grid, sols_area_1[-1])
    
    sols_area_2 = find_all_4area(grid, 2)
    reset(grid, sols_area_2[-1])
    
    # determine lay out of the structures
    numbering = np.array([0, 1, 2])
    lengths = np.array([len(sols_area_0), \
        len(sols_area_1), len(sols_area_2)])
    indices = lengths.argsort()
    order = numbering[indices]
    
    sols = (sols_area_0, sols_area_1, sols_area_2)
    sols_iter_0 = sols[order[0]]
    sols_iter_1 = sols[order[1]]
    sols_iter_2 = sols[order[2]]
    
    # assign a number of jobs if there's not
    L = len(sols_iter_0)
    if N_jbs:
        pass
    else:
        print("Using multiprocessing.cpu_count()...")
        N_jbs = multiprocessing.cpu_count()
    
    # validity check before integer division
    if N_jbs > L:
        print("Too many jobs planned!")
        N_jbs = L
    else:
        pass
    
    # set up pools for parallelization
    print('Iterating globally (%d jobs)...' % N_jbs)
    pool = multiprocessing.Pool()
    jobs = []
    
    for p in range(N_jbs):
        # slice array by the number of jobs
        start = L//N_jbs * p
        s_end = L//N_jbs * (p+1)
        if p+1 == N_jbs:
            sols_part_0 = sols_iter_0[start:]
        else:
            sols_part_0 = sols_iter_0[start:s_end]
        
        # start a new job
        arglist = (np.array(grid),order, \
                    sols_part_0,sols_iter_1,sols_iter_2)
        proc = pool.apply_async(\
            iter_globally_pool, args=arglist)
        
        # another solution is to use deepcopy():
        #   from copy import deepcopy
        
        jobs.append(proc)
        
    # assemble results
    N_iter = 0
    
    # recycle the pool
    pool.close()
    pool.join()  # you can also do this after collecting
    
    for i,proc in enumerate(jobs):
        N_iter_pr, sols_p = proc.get()
        N_iter += N_iter_pr
        for sol_p in sols_p:
            print("Unpacking...")
            yield sol_p
    
    print('Total number of iterations: %d' % N_iter)

def pb_force_solve(pzz, N_jbs=None):
    """Subroutine for presenting solutions one by one."""
    
    # unpack the puzzle
    grid = pzz.get_puzzle()
    
    # solve
    sols = pb_force_pool(grid, N_jbs)
    
    # present the solution(s)
    for i, sol in enumerate(sols):
        print('The %dth solution:' % (i+1))
        sdk = Sudoku(sol)
        sdk.show()

