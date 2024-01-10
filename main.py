import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cm
from simulation import *

DT = 0.01
N = 4
SIM_LEN = 10
SIM_SPEED = 10

state = np.zeros((N,4))
m = np.array([
    938*3+940*4, 0.511, 0.511, 0.511
]) # unit: MeV
q = np.array([
    3, -1, -1, 0
]) # unit: q_e
DISTANCES = np.array([
    0, 1, 1, 4
]) * 52.9/3 # unit: pm (based on Bohr radius)
ANGLES = np.random.rand(N) * 2*np.pi
ANGLES[2] = ANGLES[1] + np.pi


for i in range(1, N):
    dist = DISTANCES[i]
    angle = ANGLES[i]
    state[i,0] = dist * np.cos(angle)
    state[i,1] = dist * np.sin(angle)
    vi = np.sqrt(m[0]/dist)
    state[i,2] = vi * (state[i,1]-state[0,1])/dist
    state[i,3] = vi * (state[i,0]-state[0,0])/dist

import matplotlib
seismic = matplotlib.colormaps['seismic'].resampled(255)
newcolors = seismic(np.linspace(0, 1, 255))
for i in range(3):
    newcolors[:127,i] -= np.linspace(0,80,127)/256
    newcolors[127:,i] -= np.linspace(80,0,128)/256 
newcolors[np.where(newcolors < 0)] = 0
cmap = colors.ListedColormap(newcolors)

max_coord = np.max(DISTANCES)
fig = plt.figure()
scatter = plt.scatter(state[:,0], state[:,1], s=np.log(m/np.min(m)+1)*10,
                      c=q, cmap=cmap, vmin=-3, vmax=3)
plt.colorbar(scatter)

axs = fig.get_axes()
fig.get_axes()[0].set_xlim(-max_coord, max_coord)
fig.get_axes()[0].set_ylim(-max_coord, max_coord)
plt.gca().set_aspect('equal')
plt.show()