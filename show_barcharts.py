## script for presenting the result of performance tests
## created by: Yu Geng
## 2020-06-04

from __future__ import print_function

import os
from pylab import *

import seaborn as sns
sns.set_style("darkgrid")

# define tick labels globally
# (they are the same for all tests)
alg = ['BF', 'NQ', 'LC', 'EXC', 'CNN']
algorithms = ['Permutation\nSearch\n(8 CPUs)', \
              'Backtracking', \
              'Boolean\nSatisfiability', \
              'Dancing\nLinks', \
              'Neural\nNetworks']

def show_algs(filename, N_clues):
    """Subroutine for plotting the
    execution time for all algorithms."""
    
    # obtain title from the filename
    c = filename.split('/')[-1]
    tstc = c.replace('.time', '.grid')
    
    # plot a bar chart with tick labels
    time = loadtxt(filename) * 1e3
    bar(algorithms, time)
    # xticks(rotation=45)
    
    # modify ticks and labels
    # xticks(range(len(dat)), tick_labels)
    
    # manipulate axes
    title('Test case: %s\nClues: %d' % (tstc, N_clues))
    ylabel('Execution time [ms]')
    # xlabel('Algorithms')

# tick labels
cpus = ['Sequential', '2 CPUs', \
        '3 CPUs', '4 CPUs', '5 CPUs', \
        '6 CPUs', '7 CPUs', '8 CPUs']

def show_cpus(filename, casename):
    """Subroutine for showing the impact of
    the # of jobs on execution time."""
    
    # plot a bar chart with tick labels
    time = loadtxt(filename)
    bar(cpus, time)
    
    # manipulate axes
    title('Test case: %s' % casename)
    ylabel('Execution time [sec]')
    xlabel('Number of jobs')
    
# Without Seaborn, default size: [6.0, 4.0]
# With    Seaborn, default size: [8.0, 5.5]

if __name__ == '__main__':
    
    # examine output directory
    outdir = './report/'
    if os.path.isdir(outdir):
        pass
    else:
        os.mkdir(outdir)
    
    # create a new window
    figure
    filename = outdir + 'performance.png'
    rcParams['figure.figsize'] \
        = [16, 5.5]  # double the width of seaborn plot
    
    print("Plotting...")
    subplot(1,2,1)
    show_algs(outdir+'shawn_lee.time', 37)
    
    subplot(1,2,2)
    show_algs(outdir+'rohit_shreekant.time', 27)
    
    # save the plot
    savefig(filename)
    print('Done. File saved as: %s\n' % filename)
    close()  # there is an issue with show() in wingware
    
    # create another new window
    figure
    filename = outdir + 'multiprocessing.png'
    rcParams['figure.figsize'] \
        = [8.0, 5.5]  # restore default window size
    
    print("Plotting...")
    show_cpus(outdir+'multiprocessing.time', \
              'cnblogs_medium.grid')
    
    # save the plot
    savefig(filename)
    print('Done. File saved as: %s\n' % filename)
    close()  # there is an issue with show() in wingware

