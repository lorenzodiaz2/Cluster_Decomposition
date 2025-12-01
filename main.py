from elements.environment import Environment
from utils.plot_functions import plot_paths

if __name__ == '__main__':

    for i in range(5, 6):
        grid_side = 0
        if i <= 4:
            grid_side = 60
        elif i <= 9:
            grid_side = 90

        env = Environment(grid_side, 250, i, 50, -1, 12, seed=0)
        env.compute_clusters()
        for c in env.clusters:
            plot_paths(env.G, c.od_pairs)
            print(f"cluster {c.id} -> {len(c.od_pairs)}")



# todo creare una classe results che tenga il dataframe e la matrice di similarità
# todo non dividere la griglia in quadranti ma creare "copie" della griglia