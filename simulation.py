import numpy as np
from scipy.integrate import RK45

EPS = 1e-12 # epsilon for numerical stability
COULOMB_K = 2.533e38 # unit: MeV pm^3/e^2 s^2

def coulomb_law(x1, x2, y1, y2, q1, q2, m1):
    # x-direction acceleration of q1 due to charge q2
    return COULOMB_K*q1*q2*(x1-x2)/(m1*np.linalg.norm((x1-x2,y1-y2))**3 + EPS)

def simulate_steps(state0, m, q, h, steps):
    # state0 is an array of shape (N,4) for N objects
    # each row contains x, y, vx, and vy
    # m and q are arrays of shape (N) for N objects
    # h is the step size in seconds
    # steps is the number of steps to simulate

    y0 = state0.flatten() # considering the state of all particles as a single vector

    def get_derivative(vec):
        state = np.reshape(vec, state0.shape)
        d = np.zeros_like(state)
        for i in range(state.shape[0]):
            xi = state[i,0]
            yi = state[i,1]
            for j in range(state.shape[0]):
                if i == j:
                    continue
                xj = state[j,0]
                yj = state[j,1]
                d[i,2] += coulomb_law(xi, xj, yi, yj, q[i], q[j], m[i])
                d[i,3] += coulomb_law(yi, yj, xi, xj, q[i], q[j], m[i])
        d[:,0] += state[:,2]
        d[:,1] += state[:,3]
        d = d.flatten()
        return d

    simulation = [state0]
    solver = RK45(
        lambda t,y: get_derivative(y), 0, y0, t_bound=h*steps+1, max_step=h
    )
    y = y0
    for _ in range(steps):
        # y += h * get_derivative(y)
        solver.step()
        y = solver.y
        state = np.reshape(y, state0.shape)
        simulation.append(np.copy(state))
    return np.array(simulation)

def E_field(state, q, bound, n):
    x = np.linspace(-bound, bound, n)
    y = np.linspace(-bound, bound, n)
    X, Y = np.meshgrid(x, y)
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
    for i in range(state.shape[0]):
        xi = state[i,0]
        yi = state[i,1]
        r2 = np.square(X-xi) + np.square(Y-yi)
        u += COULOMB_K*q[i]*(X-xi)/(r2**3/2 + EPS)
        v += COULOMB_K*q[i]*(Y-yi)/(r2**3/2 + EPS)
    return u, v