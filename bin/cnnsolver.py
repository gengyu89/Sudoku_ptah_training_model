## subroutines for implementing the cnn method
## transformed from Shiva Verma and fit natively to my project
## 2020-05-23

import keras

from copy import deepcopy
from cpropagation import *
from represent import *  # includes os, numpy, and print_function

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# select a model from here
mod_name = './input/models/model.weighted.h5'

print('Pre-trained model: %s' % mod_name)
model = keras.models.load_model(mod_name)

def norm(grid):
    """No idea how this works, but normalizing
    with ten will produce wrong answers."""
    a = np.array(grid)
    return a / float(N) - 0.5

def denorm(a):
    # a <type 'numpy.ndarray'>
    return (a + 0.5) * float(N)

def inference_sudoku(sample):
    """Main subroutine for solving with cnn."""
    # sample - 2d np array
    
    feat = deepcopy(sample)
    
    while True:
        
        out = model.predict(feat.reshape([1,N,N,1]))  
        out = out.squeeze()
        
        pred = np.argmax(out, axis=1).reshape([N,N])+1
        prob = np.around(np.max(out, axis=1).reshape([N,N]), 2) 
        
        feat = denorm(feat).reshape((N,N))
        mask = (feat==0)  # a boolean matrix showing locations of zeros
        
        if mask.sum():  # the grid is not yet full
            pass  # execute the rest
        else:
            break
        
        prob_new = prob * mask
        
        # locate the largest probability
        ind = np.argmax(prob_new)
        i,j = ind // N, ind % N
        
        # fill in the predicted number
        val = pred[i][j]
        feat[i][j] = val
        feat = norm(feat)
    
    return feat.astype(int)  # there is a mistake in the original code

# pretrained data is loaded when the module is imported
# so that you can time the main function only

def cnn_iterator(grid):
    """Subroutine designed for use
    inside of a loop."""
    
    # iteratively fill out singletons
    # until there is not any
    fillout_singles(grid)
    
    # check if it is already solved
    if is_full(grid):
        return True
    else:
        pass  # execute the rest
    
    # solve
    game = norm(grid)  # normalize
    sol = inference_sudoku(game)  # solve and denormalize
    
    # check correctness
    sdk = Sudoku(sol)
    
    return sdk.is_correct()

def cnn_solve(grid):
    """Main function of CNN solver."""
    
    game = norm(grid)  # normalize
    sol = inference_sudoku(game)  # solve and denormalize
    
    return sol

def cnn_main(pzz):
    """Subroutine for presenting the
    solution and checking correctness."""
    
    # unpack the puzzle
    grid = pzz.get_puzzle()
    
    # iteratively fill out singletons
    # until there is not any
    fillout_singles(grid)
    
    # check if it is already solved
    if is_full(grid):
        print("The grid is full! Exiting...")
        use_cnn = False
        sol = grid
    else:
        print("Predicting and filling numbers...")
        use_cnn = True
        sol = cnn_solve(grid)
    
    # for either case, present the solution
    sdk = Sudoku(sol)
    sdk.show()
    
    # for the 2nd case, show correctness
    if use_cnn:
        echo_correctness(sdk.is_correct())
    else:
        pass

def echo_correctness(is_correct):
    
    if is_correct:
        print("The solution is correct.")
    else:
        print("The solution is incorrect!")

