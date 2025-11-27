from utils.environment import Environment

if __name__ == '__main__':

    # iteration, obj, time, other = [], [] ,[], []
    #
    # for i in range(10):
    #     iteration.append(i)
    #     obj.append(2)
    #     time.append(4)
    #     other.append(np.zeros((4, 4)))
    #
    #
    # df = pd.DataFrame({'iter': iteration, 'obj': obj, 'time': time, 'other': other})
    # df.to_csv('test.csv', index=False)
    #
    # df = pd.read_csv("test.csv")

    parallel_times = {}
    no_parallel_times = {}

    for n_quadrants in range(3, 17):
        print(f"\nnumero di quadranti = {n_quadrants}")
        parallel_times[n_quadrants] = []
        no_parallel_times[n_quadrants] = []
        for i in range(5):
            print(f"    iterazione {i}     ", end="")
            env = Environment(60, 750, n_quadrants, 150, 0, 12, False)

            env.compute_clusters()
            parallel_times[n_quadrants].append(env.matrix_time)
            print(f"parallel = {env.matrix_time}", end="     ")

            env.compute_clusters(True, False)
            no_parallel_times[n_quadrants].append(env.matrix_time)
            print(f"no parallel = {env.matrix_time}")






# todo cambiare il salvataggio e salvare su csv usando pandas
# todo fare qualche prova su istanze piccole
# todo iniziare a tirare fuori qualche numero