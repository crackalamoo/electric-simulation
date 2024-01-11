import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors, colormaps
from matplotlib.collections import LineCollection
from simulation import *

DT = 1e-19
N = 3
SIM_LEN = 5000
SIM_SPEED = 8
E_PLOT_N = 100

state = np.zeros((N,4))
m = np.array([
    938, 0.511, 938
]) # unit: MeV
q = np.array([
    1, -1, 1
]) # unit: q_e

dist = 52.9/q[0]
# angle = np.random.rand() * 2*np.pi
angle = 1.1
state[1,0] = dist * np.cos(angle)
state[1,1] = dist * np.sin(angle)
vi = np.sqrt(np.abs(COULOMB_K*q[1]*q[0])/(dist*m[1]))
state[1,2] = vi * (state[1,1]-state[0,1])/dist
state[1,3] = -vi * (state[1,0]-state[0,0])/dist
state[2,0] = -190
state[2,2] = vi*0.4
state[:,0] -= 20

simulation = simulate_steps(state, m, q, DT, SIM_LEN)

seismic = colormaps['seismic'].resampled(255)
newcolors = seismic(np.linspace(0, 1, 255))
for i in range(3):
    newcolors[:127,i] -= np.linspace(0,80,127)/256
    newcolors[127:,i] -= np.linspace(80,0,128)/256 
newcolors[np.where(newcolors < 0)] = 0
cmap = colors.ListedColormap(newcolors)

bound = dist * 3
x = np.linspace(-bound, bound, E_PLOT_N)
y = np.linspace(-bound, bound, E_PLOT_N)
X, Y = np.meshgrid(x, y)
Ex, Ey = E_field(state, q, bound, E_PLOT_N)
E_strength = np.log(Ex**2 + Ey**2 + EPS)

fig = plt.figure()
mesh = plt.pcolormesh(X, Y, E_strength, cmap='inferno')
scatter = plt.scatter(state[:,0], state[:,1], s=np.log(m/np.min(m)+1)*15,
                      c=q, cmap=cmap, vmin=-3, vmax=3)
axs = fig.get_axes()

def animate_func(i):
    Ex, Ey = E_field(simulation[i*SIM_SPEED], q, bound, E_PLOT_N)
    E_strength = np.log(Ex**2 + Ey**2)
    mesh.set_array(E_strength)
    scatter.set_offsets(simulation[i*SIM_SPEED])
    return scatter, mesh
anim = animation.FuncAnimation(
    fig, animate_func, frames=range(SIM_LEN//SIM_SPEED), interval=40)

axs[0].set_xlim(-bound, bound)
axs[0].set_ylim(-bound, bound)
fig.set_size_inches(6,6)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.gca().set_aspect('equal')
plt.gca().set_facecolor('xkcd:black')
plt.axis('off')

plt.show()
anim.save('./animation.gif', dpi=50)