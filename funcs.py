import numpy as np

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
            
    
if __name__ == "__main__":
    # temporary main for testing, matplotlib is hefty
    import matplotlib.pyplot as plt
