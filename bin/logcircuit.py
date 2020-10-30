## subroutines for implementing the boolean satisfiability method
## adopted from: Nicholas Pilkington
## 2020-05-19

import pycosat
from represent import *  # includes numpy and product

# define size of the sudoku
# as global constants
M, N = 3, 9

def exactly_one(variables):
    cnf = [variables]
    n = len(variables)
    
    for i in xrange(n):
        for j in xrange(i+1, n):
            v1 = variables[i]
            v2 = variables[j]
            cnf.append([-v1, -v2])
    
    return cnf

def transform(i,j,k):
    return i*N*N + j*N + k+1

def inverse_transform(v):
    v, k = divmod(v-1, N)
    v, j = divmod(v, N)
    v, i = divmod(v, N)
    return i, j, k

def sat_solver(grid):
    """Main function borrowed from Nicholas."""
    
    # conjunctive normal form
    cnf = []
    
    # cell, row and column constraints
    for i in xrange(N):
        for s in xrange(N):
            cnf = cnf + exactly_one([transform(i,j,s) for j in xrange(N)])
            cnf = cnf + exactly_one([transform(j,i,s) for j in xrange(N)])

        for j in xrange(N):
            cnf = cnf + exactly_one([transform(i,j,k) for k in xrange(N)])
    
    # sub-matrix constraints
    for k in xrange(N):
        for x in xrange(M):
            for y in xrange(M):
                v = [transform(y*M+i, x*M+j, k) \
                     for i in xrange(M) for j in xrange(M)]
                cnf = cnf + exactly_one(v)
    
    # constraints from the initial numbers
    constraints = []
    for i,j in product(range(N), range(N)):
        if grid[i][j]:
            clue = (i,j, grid[i][j])
            constraints.append(clue)
        else:
            pass
    
    print "Solving with satisfiability..."
    cnf = cnf + [[transform(z[0], z[1], z[2])-1] for z in constraints]
    
    # collect solution(s)
    for solution in pycosat.itersolve(cnf):
        
        # convert back into a sudoku grid
        X = [inverse_transform(v) for v in solution if v > 0]
        
        # assemble the current solution
        sol = []
        for i,cell in enumerate(sorted(X, key=lambda h: h[0]*N*N + h[1]*N)):
            sol.append(cell[2] + 1)
        
        # reshape and yield
        a = np.array(sol)
        yield a.reshape(N,-1)

def sat_main(pzz):
    """Main function for solving a Sudoku
    puzzle with boolean satisfiability."""
    
    # unpack the puzzle
    grid = pzz.get_puzzle()
    
    # solve it with sat
    sols = sat_solver(grid)
    
    # present the solution(s)
    for i,sol in enumerate(sols):
        print('The %dth solution:' % (i+1))
        sdk = Sudoku(sol)
        sdk.show()

    