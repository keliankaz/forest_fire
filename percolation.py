#%%
from cellular_fire import ForestFire
import numpy as np
from matplotlib import pyplot as plt

number_of_splits = 10
p = np.linspace(0.1,0.8, number_of_splits)

grid_size = 400
fig, ax = plt.subplots()
pc = 0.59
for i_p in p:
    grid = np.random.random((grid_size, grid_size))
    grid = grid<i_p
    areas = ForestFire().get_areas(grid)
    ax.plot(
        *np.unique(areas,return_counts=True),
        c=[np.abs(i_p-pc)*2, 0, 1-np.abs(i_p-pc)*2],
        alpha=0.5,
    )

grid = np.random.random((grid_size, grid_size))
grid = grid<pc
areas = ForestFire().get_areas(grid)
ax.plot(
    *np.unique(areas,return_counts=True),
    c='k',
)    

ax.set(
    yscale='log',
    xscale='log',
)