from elements.environment import Environment
from utils.plot_functions import plot_paths

if __name__ == '__main__':

    for i in range(2, 10):
        grid_side = 0
        if i <= 4:
            grid_side = 120
        elif i <= 9:
            grid_side = 180

        env = Environment(grid_side, 750, i, 150, -1, 12, seed=0)
        env.compute_clusters()
        for c in env.clusters:
            plot_paths(env.G, c.od_pairs)



# todo creare una classe results che tenga il dataframe e la matrice di similarità
# todo non dividere la griglia in quadranti ma creare "copie" della griglia