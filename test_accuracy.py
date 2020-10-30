## script built to investigate cnn correctness
## created by: Yu Geng
## 2020-05-23

import sys
from bin.cnnsolver import *    # cnnsolver   <- represent <- numpy, time, os
from bin.dancinlinks import *  # dancinlinks <- represent <- numpy, time, os

# define size of the experiment
# as global constant
N_cases = 1000

# Downgrading tensorflow to 1.5 will solve
#   all the deprecation troubles:
# $ pip install tensorflow==1.5

# ----- Subroutines for Experiment I -----

def str2num(s):
    if s == '.':
        return 0
    else:
        return int(s)

def transform_puzzle(raw):
    """Subroutine for reformulating a line
    from sudoku17.1000.txt into a 2d grid."""
    
    pzz = map(str2num, raw.strip())
    a = np.array(pzz)
    
    return a.reshape(N,-1)

def time_each_quiz(quizzes, expdata):
    """Subroutine developed to find
    the most difficult quizzes."""
    
    fp = open(quizzes, 'r')
    raw = fp.readline()
    i = 0  # initialize counter
    
    # pre allocate for better performance
    N_clues = np.zeros([N_cases,1])
    t_solve = np.zeros([N_cases,1])
    
    while raw != "":
        
        status = '\rSolving test case %d...' % (i+1)
        sys.stdout.write(status)
        sys.stdout.flush()
        
        grid = transform_puzzle(raw)
        N_clues[i] = count_clues(grid)
        
        t = time.time()  # time only the solving part
        excover_sdk_iterator(grid)
        t_solve[i] = time.time() - t
        
        raw = fp.readline()
        i += 1  # update counter
    
    print("")
    fp.close()
    
    # save these data so that you do not
    # have to run this test one more time
    mat = np.concatenate((N_clues, t_solve), axis=1)
    np.savetxt(expdata, mat, fmt='%4d%10.6f')
    
    print('Done. File saved as: %s' % expdata)

# ----- Subroutines for Experiment II -----

def correctness(quizzes, level):
    """Subroutine designed to calculate
    the correctness of CNN method."""
    
    print('Solving collection %s...' % quizzes)
    print('Difficulty level: %d\n' % level)
    
    # pre allocate for better performance
    N_correct = 0
    
    # start processing line by line
    fp = open(quizzes, 'r')
    raw = fp.readline()
    i = 0  # initialize counter
    
    while raw != "":
        
        grid = transform_puzzle(raw)
        if cnn_iterator(grid):
            status = 'Case: %4d Correctness: %5s' % (i+1, True)
            N_correct += 1
        else:
            status = 'Case: %4d Correctness: %5s' % (i+1, False)
        print(status)
        
        raw = fp.readline()
        i += 1  # update counter
    
    fp.close()
    
    # calculate and show
    score = 1e2 * N_correct / N_cases
    print('Accuracy: %f\n' % score)
    
    return level, score

# ----- The main function -----

if __name__ == '__main__':
    
    # examine output directory
    outdir = './input/models/'
    if os.path.isdir(outdir):
        pass
    else:
        os.mkdir(outdir)
    
    # note that the training model
    # must be specified in cnnsolver.py

    # solve a thousand quizzes with cnn
    t = time.time()
    quizzes = './output/quizzes/level1.1000.txt'
    correctness(quizzes, 1)
    show_time(t)

