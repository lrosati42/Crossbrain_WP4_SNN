import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
from pathlib import Path
from par_space_dynamics import *

def init():
    line.set_data([], [])
    return line,

def update(frame):
    frame *= subsample_factor
    line.set_data(x_r[:frame], np.abs(y_r[:frame]))
    return line,

np.random.seed(4)
exp_id = 0

walker = Parameter_Walker(ds=0.015, x0_0 = 0.1, CpES_0 = 0.1)
steps = 3600 * 2
x, y = walker.walk(steps, return_traj=True)
y_r, x_r = denorm_params(y, x)

subsample_factor = 20
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0.1, 1)
ax.set_ylim(2, 4.5)
line, = ax.plot([], [], marker='o', markersize=2)

frames = len(x) // subsample_factor
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=20, repeat=False)
ani.save("plots/random_walk_subsampled.gif", writer=PillowWriter(fps=30))

expath = f'data/synthetic/epileptor/full/{exp_id}'
expath = expath + '/' + f'{exp_id}'
Path(expath).mkdir(parents=True, exist_ok=True)
np.save(f'{expath}/x0.npy', y)
np.save(f'{expath}/x0_real.npy', y_r)
np.save(f'{expath}/cp.npy', x)
np.save(f'{expath}/cp_real.npy', x_r)
