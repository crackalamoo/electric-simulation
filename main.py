import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, colors, colormaps
from simulation import *

DT = 1e-19
N = 3
SIM_LEN = 4000
SIM_SPEED = 4

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
print(angle)
state[1,0] = dist * np.cos(angle)
state[1,1] = dist * np.sin(angle)
vi = np.sqrt(np.abs(COULOMB_K*q[1]*q[0])/(dist*m[1]))
state[1,2] = vi * (state[1,1]-state[0,1])/dist
state[1,3] = -vi * (state[1,0]-state[0,0])/dist
state[2,0] = -190
state[2,2] = vi*0.4

simulation = simulate_steps(state, m, q, DT, SIM_LEN)

seismic = colormaps['seismic'].resampled(255)
newcolors = seismic(np.linspace(0, 1, 255))
for i in range(3):
    newcolors[:127,i] -= np.linspace(0,80,127)/256
    newcolors[127:,i] -= np.linspace(80,0,128)/256 
newcolors[np.where(newcolors < 0)] = 0
cmap = colors.ListedColormap(newcolors)

print(len(simulation))
print(np.concatenate(simulation)[:,:2].shape)
max_coord = dist * 2
fig = plt.figure()
scatter = plt.scatter(state[:,0], state[:,1], s=np.log(m[:N]/np.min(m[:N])+1)*10,
                      c=q[:N], cmap=cmap, vmin=-3, vmax=3)

def animate_func(i):
    scatter.set_offsets(simulation[i*SIM_SPEED])
    return scatter,
ani = animation.FuncAnimation(
    fig, animate_func, frames=range(SIM_LEN//SIM_SPEED), interval=20)

axs = fig.get_axes()
fig.get_axes()[0].set_xlim(-max_coord, max_coord)
fig.get_axes()[0].set_ylim(-max_coord, max_coord)
plt.gca().set_aspect('equal')
plt.show()