## script for training cnn solver and generating test cases
## adopted from http://yshblog.com/blog/196
## 2020-05-25

from __future__ import print_function

from bin.represent import *   # includes os, time, and numpy
from bin.cnntrainer import *  # includes sys, random, and keras

# ----- Subroutines for the generator -----

def zero2dot(s):
    if s == '0':
        return '.'
    else:
        return s

def haskell(puzzle):
    """Subroutine for formatting a puzzle
    into Haskell style."""
    
    return ''.join(map(zero2dot, puzzle))

weights = {5: 0.516, 4: 0.258, 3: 0.129, 2: 0.065, 1: 0.032}

def create_levels(length=1e3):
    
    levels = []
    for n, p in weights.iteritems():
        levels += [n] * int(p * length)
    
    # note that because of rounding,
    # the size of the resultant array
    # may not be equal to the desired length
    
    return levels

N_train = int(1e6)
N_tests = int(1e3)

def make_training_dat(filename, levels):
    
    fp = open(filename, 'w')
    
    t = time.time()
    for i in xrange(N_train):

        status = '\rGenerating quiz %d...' % (i+1)
        sys.stdout.write(status)
        sys.stdout.flush()
        
        level = random.choice(levels)
        fp.write('%83s%83s\n' % (generate(level)))
    
    print("")
    fp.close()
    
    print('Done. File saved as: %s' % filename)
    show_time(t)

def make_testing_cases(filename, level):

    fp = open(filename, 'w')
    
    t = time.time()
    for i in xrange(N_tests):

        status = '\rGenerating quiz %d...' % (i+1)
        sys.stdout.write(status)
        sys.stdout.flush()

        puzzle, _ = generate(level)  # puzzles only
        fp.write('%s\n' % haskell(puzzle))
    
    print("")
    fp.close()
    
    print('Done. File saved as: %s' % filename)
    show_time(t)

# ----- Subroutines for previewing training data -----

def make_hist(train_dat, exp_data):
    """Subroutine for showing the
    distribution for the number of clues."""
    
    fp = open(train_dat, 'r')
    raw = fp.readline()
    
    # pre allocate structure
    N_clues = np.zeros([N_train,1])
    i = 0  # initialize counter
    
    while raw != "":
        
        status = '\rProcessing quiz %d...' % (i+1)
        sys.stdout.write(status)
        sys.stdout.flush()
        
        quizz, _ = raw.split()
        puzzle = [int(q) for q in quizz]
        clues = filter(lambda p:p, puzzle)
        N_clues[i] = len(clues)
        
        raw = fp.readline()
        i += 1  # update the counter
    
    print("")
    fp.close()
    
    # write these data to hard disk
    # for higher flexibility
    np.savetxt(exp_data, N_clues, fmt='%4d')
    print('Done. File saved as: %s' % exp_data)

# ----- Subroutines for building the model -----

def train_cnn(train_dat, mod_name):
    """Subroutine for building, compiling
    a training model and timing."""
    
    # load the training dataset
    x_train, x_test, y_train, y_test = \
        get_data_ascii(train_dat)
    
    # build a training model
    model = get_model()
    
    t = time.time()
    print("Compiling (this will take a while)...")    
    
    adam = keras.optimizers.adam(lr=.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
    model.fit(x_train, y_train, batch_size=32, epochs=2)
    
    # save the model
    model.save(mod_name)
    
    print('Done. File saved as: %s' % mod_name)
    show_time(t)

# ----- The main function -----

if __name__ == '__main__':
    """Generate a million training quizzes
    and write them into an ASCII file."""
    
    # generate training data
    train_dat = './input/models/training.weighted.txt'
    # levels = range(1,6)
    levels = create_levels()
    make_training_dat(train_dat, levels)
    
    # preview the training data
    # exp_data = './report/hist.mega.dat'
    # make_hist(train_dat, exp_data)
    
    # build a training model
    mod_name = './input/models/model.weighted.h5'
    train_cnn(train_dat, mod_name)

    # generate testing cases
    # filename = './output/quizzes/demo.1000.txt'
    # make_testing_cases(filename, 5)

    # do not run repeatedly
    # to overwrite my previous test cases

