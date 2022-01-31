import numpy as np
import matplotlib.pyplot as plt
import random

def solve_bound_state(start_x, end_x, delta_x, V, h_bar, m):
    """
    The energy eigenvalues to a potential that admit only bound states.

    TODO: add eps for numercal stability.
    TODO: research methods to reduce error in the face of
          a natural-order h_bar.
    TODO: better docstring(s).
    """
    xs = np.arange(start_x, end_x, delta_x)
    N = len(xs)
    H = np.zeros((N,N))
    lam = (h_bar*h_bar)/(2*m*delta_x*delta_x)
    H[0,0] = (2*lam) + V(xs[0])
    if N > 1:
        H[0,1] = -lam
        H[1,0] = -lam
        H[1,1] = (2*lam) + V(xs[-1])
    for i in range(1, N-1):
        H[i, i-1] = -lam
        H[i, i] = (2*lam) + V(xs[i])
        H[i, i+1] = -lam
    w, v = np.linalg.eig(H)
    idx = np.argsort(w)
    return w[idx], np.swapaxes((v/(np.sum(v*v, 0)*np.sqrt(delta_x)))[:,idx], 0, 1)

def random_walk(n, d, rand, *args):
    """
    Simulate n steps of a d-dimensional random walk where the random walk
    at each step is distributed according to a probability distribution
    function rand. *args are the arguments to be passed into rand.

    TODO: Is there a way of generating random samples in parallel,
          then summing along each axis cumulatively to speed up
          this computation? It is unbearably slow as it is...
    """
    m = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            m[i, j] = rand(*args) + m[i-1, j] # m[-1, :] is 0 on init
    return m

def random_walk_ensemble(n, s, rand, *args):
    """
    Plotting utility for s samples of n-step random walks in 2 dimensions.
    This is an example of the Central Limit Theorem in action; no matter
    what probability distribution you put in for rand (even a dirac-delta
    function), you will observe some form of a gaussian distribution. *args
    are the arguments to be passed into rand.
    """
    _, ax = plt.subplots()
    plt.gca().set_aspect("equal")
    ps = np.array([random_walk(n, 2, rand, *args)[-1, :] for _ in range(s)])
    circ = plt.Circle((0, 0), np.sqrt(n), fill=False)
    ax.add_patch(circ)
    plt.scatter(ps[:, 0], ps[:, 1], s=1000/s)
    plt.title("Random Walk Endpoints for Standard Normal Distribution Steps With "+str(n)+" Steps and "+str(s)+" Trials.")
    plt.xlabel("final x displacement")
    plt.ylabel("final y displacement")
    ax.legend([circ], [r'$\sigma = \sqrt{N}$'], loc="best")
    plt.show()
    return
    
if __name__ == "__main__":
    pass
    # temporary main for testing
    
