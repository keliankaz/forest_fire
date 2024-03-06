#%% 
import numpy as np
from scipy.signal import convolve2d
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio

#%%

class ForestFire:
    def __init__(
        self,
        num_x: int = 128,
        num_y: int = 128,
        tree_frequency: float = 1.,
        spark_frequency: float = 1/2000,
    ) -> None:
        self.num_x = num_x
        self.num_y = num_y
        self.tree_frequency = tree_frequency
        self.spark_frequency = spark_frequency

        # states: 
        self.burning = False
        self.tree_grid, self.fire_grid, self.burnt_grid = [
            self.get_null_grid() for _ in range(3)
        ] # 3 grids
        
        self.number_of_steps = 0
        
        # history:
        self.burn_record = [] 
        self.tree_record = []
        self.record = [] # trees=1, burn=2
    
    def area(self):
        return self.num_x * self.num_y
        
    def get_null_grid(self):
        return np.zeros((self.num_x,self.num_y)).astype(int)
    
    def combine_tree_and_burn(self):
        return self.tree_grid + 2*self.burnt_grid
    
    def step(self) -> None:
        
        self.burnt_grid = self.get_null_grid()
        
        for _ in range(np.random.poisson(self.tree_frequency)):
            random_x,random_y = [np.random.randint(0,N) for N in [self.num_x, self.num_y]]
            self.tree_grid[random_x,random_y] = True
        
        for _ in range(np.random.poisson(self.spark_frequency)):
            random_x,random_y = [np.random.randint(0,N) for N in [self.num_x, self.num_y]]
            self.fire_grid[random_x,random_y] = True
    
        if np.any(self.fire_grid==1):
            self.burning = True
            while self.burning:
                
                self.tree_grid[self.fire_grid==1] = False # burn trees
                self.burnt_grid = (self.burnt_grid + self.fire_grid)>0
                
                possible_fire_grid = (
                    convolve2d(
                        self.fire_grid,
                        np.array([
                            [0,1,0],
                            [1,1,1],
                            [0,1,0],
                        ]),
                        mode='same',
                    ) > 0).astype(int) # sparks neighbors
                
                
                self.fire_grid = np.bitwise_and(possible_fire_grid, self.tree_grid)
                
                if not np.any(self.fire_grid): # stop fire
                    self.burning = False
                    
        self.number_of_steps += 1
    
    def log(self, record_no_burn=False):
        if record_no_burn:
            self.burn_record.append(self.burnt_grid)
        elif np.any(self.burnt_grid):
            self.burn_record.append(self.burnt_grid)
            
        self.tree_record.append(self.tree_grid.copy())
        
        self.record.append(self.combine_tree_and_burn())
    
    def get_burn_record_count(self) -> np.ndarray:
        return np.array(
            [np.sum(fire) for fire in self.burn_record]
        )
    
    def get_burn_tree_count(self) -> np.ndarray:
        return np.array(
            [np.sum(trees) for trees in self.tree_record]
        )
        
    def plot_state(self, state=None, ax=None):
        
        if state is None:
            state = self.combine_tree_and_burn()
        
        if ax is None:
            _, ax = plt.subplots()
            
        cmap = mpl.colors.ListedColormap(['w', 'g', 'r'])
        ax.matshow(state, cmap=cmap, vmin=0, vmax=2)
        
        return ax
    
    def plot_time_series(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        
        ax.plot(range(self.number_of_steps), self.get_burn_tree_count())
        
        return ax  
    
    def plot_scaling(self, ax=None):
        
        if ax is None:
            _, ax = plt.subplots()
        
        bins = np.logspace(0,np.log10(self.area()), 20)    
        ax.hist(self.get_burn_record_count(), bins=bins)

        ax.set(
            xscale='log',
            yscale='log',
            ylabel=r'$N_f$',
            xlabel=r'$A_F$',
        )

        return ax
               
    def make_gif(
        self,
        file_name = "forest_fire_simulation_in_memory.gif",
        fps = 5,
        plot_every_n_steps = 1
    ) -> None: 
        images_memory = []
        for i, state in enumerate(self.record):
            if i % plot_every_n_steps == 0:
                fig, ax = plt.subplots()
                self.plot_state(state,ax=ax)
                fig.canvas.draw()
                image_from_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_fig = image_from_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images_memory.append(image_from_fig)
                plt.close(fig)
            
        imageio.mimsave(file_name, images_memory, fps=fps)
        
         
#%% 

if __name__ == "__main__":
    
    forest_fire_automota = ForestFire()
    for timestep in range(100000): # run for 100 "seasons"
        forest_fire_automota.step()
        forest_fire_automota.log()

    # plot the final state of the "forest"
    forest_fire_automota.plot_state()
    
    # plot a time series of the number of trees
    forest_fire_automota.plot_time_series()
    
    # plot area scaling for forest fires
    forest_fire_automota.plot_scaling()
    
    # make a gif (avoid using an unreasonably large number of timesteps)
    # forest_fire_automota.make_gif() 

