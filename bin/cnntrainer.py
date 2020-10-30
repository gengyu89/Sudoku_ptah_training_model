## subroutines developed for generating test cases and cnn training
## created by: Yu Geng
## 2020-05-24

from represent import *  # includes os, numpy, product, and print_function

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Use$ sudo conda install nomkl
#   to fix any issues in this code
# Refer to:
#   https://github.com/openai/spinningup/issues/16

# ----- Subroutines for creating the base sudo -----

import math
import random

def is_full(grid):
    """Copied from ./Sudoku_0510/"""
    mask = (np.array(grid)==0)
    
    return not mask.sum()

def sel_empty_cell_rand(grid):
    """Provided a 9x9 sudoku, returns the index of
    an empty cell selected randomly."""
    
    # enumerate all available cells
    available = []
    
    for i,j in product(range(N), range(N)):
        if grid[i][j]:
            pass
        else:
            available.append((i,j))
    
    return random.choice(available)

# Trying one cell next to another will
# provide you many more constraints.

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

def get_submatrix(grid, ei, ej):
    """Provided a 9x9 sudoku, returns the submatrix that
    the element specified by i and j belongs to."""
    
    # find box number from cell indices
    I, J = ei+1, ej+1  # convert into natural indices
    A = int(math.ceil(I/3.0))
    B = int(math.ceil(J/3.0))
    
    # determine ranges of iteration
    m, n = A-1, B-1  # box number in python convention
    i_range = range(3*m, 3*m+3)
    j_range = range(3*n, 3*n+3)
    
    # enumerate box members and store in a list
    submatrix = []
    for si, sj in product(i_range, j_range):
        submatrix.append(grid[si][sj])
    
    return submatrix

def lst_candidates(grid, ei, ej):
    """Provided a 9x9 sudoku, returns a list containing
    all the candidates that can be filled into the spot
    specified by i and j."""
    
    # find the submatrix, the row, and the column
    sdk = np.array(grid)
    row, col = sdk[ei,:], sdk[:,ej]
    box = get_submatrix(grid, ei, ej)
    
    # keep only non-zero elements
    filled_row = filter(lambda e:e, row)
    filled_col = filter(lambda e:e, col)
    filled_box = filter(lambda e:e, box)
    
    # find their common set
    filled_all = set(filled_row + filled_col + filled_box)
    
    # remove these elements from candidates
    candidates = range(1,N+1)
    for filled in filled_all:
        candidates.remove(filled)
    
    return candidates

def n_queens_rand(grid):
    """Subroutine updated for creating the base sudo
    of a randomly generated Sudoku puzzle."""
    
    if is_full(grid):
        # print("Base case triggered!")
        return grid
    
    else:
        # select an empty cell one next to another
        # and enumerate its candidates
        ti,tj = sel_empty_cell(grid)
        candidates = lst_candidates(grid, ti,tj)
        
        if candidates:
            
            random.shuffle(candidates)
            for c in candidates:
                grid[ti][tj] = c
                sol = n_queens_rand(grid)
                if sol:
                    return sol
                else:
                    grid[ti][tj] = 0  # reset it to zero
                    continue
            
            # after trying all candidates,
            # no solution found
            grid[ti][tj] = 0
            return False
        
        else:
            return False

# ----- Subroutines for generating test cases -----

def get_row(sudo, row):
    return sudo[row, :]

def get_col(sudo, col):
    return sudo[:, col]

def get_block(sudo, row, col):
    row_start = row // R * R
    col_start = col // C * C
    return sudo[row_start: row_start + R, col_start: col_start + C]

def create_base_sudo():
    
    sudo = np.zeros((N, N), dtype=int)
    num = random.randrange(N) + 1
    
    for row_index in range(N):
        for col_index in range(N):
            
            sudo_row = get_row(sudo, row_index)
            sudo_col = get_col(sudo, col_index)
            sudo_block = get_block(sudo, row_index, col_index)
            
            while num in sudo_row or num in sudo_col or num in sudo_block:
                num = num % N + 1
            
            sudo[row_index, col_index] = num
            num = num % N + 1
    return sudo

def random_sudo(sudo, times):
    for _ in range(times):
        
        rand_row_base = random.randrange(R) * R
        rand_rows = random.sample(range(R), 2)
        row_1 = rand_row_base + rand_rows[0]
        row_2 = rand_row_base + rand_rows[1]
        sudo[[row_1, row_2], :] = sudo[[row_2, row_1], :]
        
        rand_col_base = random.randrange(C) * C
        rand_cols = random.sample(range(C), 2)
        col_1 = rand_col_base + rand_cols[0]
        col_2 = rand_col_base + rand_cols[1]
        sudo[:, [col_1, col_2]] = sudo[:, [col_2, col_1]]

def get_sudo_subject(sudo, del_nums):
    subject = sudo.copy()
    
    clears = random.sample(range(N*N), del_nums)
    for clear_index in clears:
        
        row_index = clear_index // N
        col_index = clear_index % N
        subject[row_index, col_index] = 0
    return subject

def lst2str(sudo):
    """Convert numbers list into a string
    with no spaces."""
    
    sudo_flatten = sudo.reshape(1,N*N)
    slst = sudo_flatten.tolist()[0]
    
    # [str(s) for s in slst] returns a list
    # (str(s) for s in slst) returns a generator
    
    return ''.join(str(s) for s in slst)

# it would be more helpful to define these things globally
max_clear_count = 64
min_clear_count = 14
each_level_count = (max_clear_count - min_clear_count) / 5

def generate(level):
    
    level_start = min_clear_count + (level - 1) * each_level_count
    del_nums = random.randrange(level_start, level_start + each_level_count)
    
    # print("Creating base...")
    base = np.zeros([N,N], dtype=int)
    base = n_queens_rand(base.tolist())  # with back tracking
    # sdk = Sudoku(base)
    # sdk.show()  # preview the base
    
    # print("Creating final...")
    sudo = np.array(base)
    random_sudo(sudo, 4)
    # sdk = Sudoku(sudo)
    # sdk.show()  # preview the solution
    
    subject = get_sudo_subject(sudo, del_nums)
    
    # skip the original output
    puzzle   = lst2str(subject)
    solution = lst2str(sudo)
    
    return puzzle, solution

# ----- Subroutines for loading training data -----

import sys
from sklearn.model_selection import train_test_split

def transform_qzz(quiz):
    # quiz <type 'str'>
    a = np.array([int(q) for q in quiz])
    return a.reshape([N,N,1])

def transform_sol(solution):
    # solution <type 'str'>
    a = np.array([int(s) for s in solution])
    return a.reshape([N*N,1]) - 1.0

def get_data_ascii(file):
    """The pandas version does not work for sudoku.csv"""
    
    print('Processing %s' % file)
    fp = open(file, 'r')
    raw = fp.readline()
    
    # pre allocate structures  # somehow this will cause a trouble
    # feat  = np.zeros([N_train,1])
    # label = np.zeros([N_train,1])
    feat, label = [], []  # pre allocating doesn't work here
    
    i = 0  # initialize counter
    while raw != "":
        
        status = '\rLoading quiz %d...' % (i+1)
        sys.stdout.write(status)
        sys.stdout.flush()
        
        quiz, solution = raw.split()
        # feat[i]  = transform_qzz(quiz)
        # label[i] = transform_sol(solution)
        feat.append(transform_qzz(quiz))
        label.append(transform_sol(solution))
        raw = fp.readline()
        i += 1
    
    feat  = np.array(feat)/9.0 - 0.5
    label = np.array(label)
    
    print("")
    fp.close()
    
    x_train, x_test, y_train, y_test = \
        train_test_split(feat, label, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test

# ----- Subroutines for building a training model -----

import keras
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape

def get_model():
    
    print("Building a model...")
    
    model = keras.models.Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', \
        input_shape=(9,9,1)))
    
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))
    
    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))
    
    return model

