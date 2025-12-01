from elements.environment import Environment

if __name__ == '__main__':

    env = Environment(60, 750, 3, 150, -1, 12, seed=6)
    env.compute_clusters()



# todo creare una classe results che tenga il dataframe e la matrice di similarità
# todo non dividere la griglia in quadranti ma creare "copie" della griglia