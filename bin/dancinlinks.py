## subroutines for implementing exact cover reduced version
## created by: Yu Geng
## 2020-04-30

import math
import random
from represent import *  # includes numpy and product

# ----- Exact cover in general -----

# Implement two dictionaries for bidirectional search
# 
# Dictionary X looks like:
# X={
#     1:{'A','B'},
#     2:{'E','F'},
#     3:{'D','E'},
#     4:{'A','B','C'},
#     5:{'C','D'},
#     6:{'D','E'},
#     7:{'A','C','E','F'}
# }
# 
# Dictionary Y looks like:
# Y={
#     'A':[1,4,7],
#     'B':[1,4],
#     'C':[4,5,7],
#     'D':[3,5,6],
#     'E':[2,3,6,7],
#     'F':[2,7]
# }
# 
# The problem aims to find a combination of Y keys
# that have all the X keys covered with no duplicates.
# 

def find_S(Y):
    """Subroutine used to find the complete set."""
    
    S = set()
    for numbers in Y.values():
        for num in numbers:
            S.add(num)
    
    return S

def find_X(Y):
    """Y is a dictionary with letters indexing lists of numbers;
    X is a dictionary with numbers indexing lists of letters."""
    
    # create an empty dictionary
    X = dict()
    
    for key, values in Y.items():
        for v in values:
            if X.has_key(v):
                X[v].add(key)
            else:
                X[v] = set([key])  # create a new set
    
    return X

def validate(S, sel):
    """Verify if a solution is an exact cover."""
    
    if S == find_S(sel):
        # check if there are duplicates
        all_num = []
        for numbers in sel.values():
            all_num = all_num + numbers
        return len(S) == len(all_num)
    
    else:  # the solution is insufficient
        return False

def excover(S, X, Y, sel=dict()):
    """Main function for calling recursively"""
    # S   - the complete set
    # sel - list of selected letters
    
    if Y == dict():
        if S == find_S(sel):
            return sel
        else:
            return False
    
    else:
        # create a list containing
        # letters with the 1st number
        letters = X.values()[0]
        random.shuffle(list(letters))
        
        for L in letters:
            # add to the solution
            lst_of_num = Y[L]
            sel[L] = lst_of_num
            
            l_to_be_removed = set()
            for num in lst_of_num:
                set_of_letters = X[num]
                l_to_be_removed = \
                    l_to_be_removed.union(set_of_letters)
            
            # remove these entries from X and Y
            Y_new = dict(Y)
            for l in l_to_be_removed:
                del Y_new[l]
            X_new = find_X(Y_new)
            
            # pass to the next level
            sol = excover(S, X_new, Y_new, sel)
            if sol:
                return sol
            else:
                del sel[L]
                continue   # try next one
        
        return False

def excover_solve(Y):
    """Provided Y, finds X and the complete set;
    drives the main function to solve it."""
    
    S, X = find_S(Y), find_X(Y)
    sol = excover(S, X, Y)
    
    print("Solution:")
    print(sol)

# ----- Exact cover for Sudokus -----

# Do not include constraint propagation in this approach.
# It turned out to be slowing things down.

def map_X(X_keys, Y):
    """Subroutine for finding the real x dictionary
    from the list created."""
    
    # map x keys into a dictionary with the same length
    X = list(X_keys)
    X = {j:set() for j in X}
    
    for key,values in Y.iteritems():
        for v in values:
            X[v].add(key)
    
    return X

def init_XY():
    """Initializes two dictionaries
    for representing an exact cover matrix."""
    
    # create a list storing x keys
    X_keys = ([("rc", rc) for rc in product(range(N), range(N))] + \
         [("rn", rn) for rn in product(range(N), range(1, N+1))] + \
         [("cn", cn) for cn in product(range(N), range(1, N+1))] + \
         [("bn", bn) for bn in product(range(N), range(1, N+1))])
    
    # create the y dictionary
    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N+1)):
        b = (r//R) * R + (c//C)  # works in both py2 and py3
        Y[(r,c,n)] = [("rc", (r,c)), \
            ("rn", (r,n)), \
            ("cn", (c,n)), \
            ("bn", (b,n))]
    
    # find x from y and verify their sizes
    X = map_X(X_keys, Y)
    assert len(X) == 4*N*N, "Invalid matrix height!"
    assert len(Y) == N*N*N, "Invalid matrix width!"
    
    return X, Y

# In this case,
#   X is the dictionary that has row elements indexing columns;
#   Y is the dictionary that has column elements indexing rows.
# 
# Still, the problem aims to find a combination of Y keys
#   that have all the X keys covered with no duplicate.
# 

def excover_sdk_solve(pzz):
    """Main subroutine for solving a
    Sudoku puzzle using exact cover."""
    
    # unpack the puzzle
    grid = pzz.get_puzzle()
    
    # create an initial exact cover matrix
    # and represent them with dictionaries
    X, Y = init_XY()
    
    # remove the initial elements
    for i,j in product(range(N), range(N)):
        v = grid[i][j]
        if v:
            select(X, Y, (i,j,v))
        else:
            pass
    
    # solve it with the subroutine
    # for exact cover problems in general
    print("Solving with dancing links...")
    all_sol = excover_sdk(X, Y)  # do not flip x and y
    
    # fill out the original 9x9 sudoku
    for sol in all_sol:  # unpack all existing solutions
        for (r,c,v) in sol:  # unpack numbers
            grid[r][c] = v
    
    # present
    sdk = Sudoku(grid)
    sdk.show()

# Refer to
#   https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html
# for the reduction from dancing links into python dicts.
# 

def excover_sdk(X,Y, sel=[]):
    """Subroutine for calling recursively."""
    
    if not X:
        yield list(sel)
        
    else:
        c = min(X, key=lambda c: len(X[c]))
        
        for r in list(X[c]):
            sel.append(r)
            cols=select(X,Y,r)
            
            for s in excover_sdk(X,Y, sel):
                yield s
            
            deselect(X,Y,r, cols)
            sel.pop()

def select(X,Y,r):
    
    cols = []  # backup of deleted letters
    
    for j in Y[r]:  # searching numbers from letter r
        for i in X[j]:  # searching letters containing j
            for k in Y[i]:  # searching numbers in letter i
                
                if k != j:
                    X[k].remove(i)  # delete letters i from number k in X
                else:
                    pass
                
        cols.append(X.pop(j))
        
        # X.pop(j) has all letters that contain the current number j
        # however, there could be multiple numbers in r
    
    return cols  # a list of sets containing letters

def deselect(X,Y,r, cols):
    
    # search for numbers from letter r (last comes 1st)
    for j in reversed(Y[r]):
        
        # restore a number and its letters
        # (the last element in cols)
        X[j] = cols.pop()
        
        for i in X[j]:  # search for letters from number j
            for k in Y[i]:  # search for numbers from letter i
                
                if k != j:
                    X[k].add(i)  # add letter i to number k in X
                else:
                    pass

def excover_sdk_iterator(grid):
    """Subroutine programmed
    for use inside of a loop."""
    
    # create an initial exact cover matrix
    # and represent them with dictionaries
    X, Y = init_XY()
    
    # remove the initial elements
    for i,j in product(range(N), range(N)):
        v = grid[i][j]
        if v:
            select(X, Y, (i,j,v))
        else:
            pass
    
    # solve it with the subroutine
    # for exact cover problems in general
    all_sol = excover_sdk(X, Y)  # do not flip x and y
    
    # fill out the original 9x9 sudoku
    for sol in all_sol:  # unpack all existing solutions
        for (r,c,v) in sol:  # unpack numbers
            grid[r][c] = v

