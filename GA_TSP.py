#-- coding:utf-8 --
import matplotlib.pyplot as plt
import numpy as np

N_CITIES = 50  # DNA size
CROSS_RATE = 0.8
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 5000


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])#五百个不同的顺序
        print(self.pop.shape)

    def translateDNA(self, DNA, city_position):     # get cities' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            #print('(xs,ys)',(xs,ys))
            #print('xs',np.square(np.diff(xs)))
            #print('ys',np.square(np.diff(ys)))
            #print('z',np.square(np.diff(xs)) + np.square(np.diff(ys)))
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness_max =2**total_distance                                         #算最大值时变异率0.02，交叉率0.8
        fitness_min=(self.DNA_size * 2 / 3**total_distance)                    #算最小值时变异率0.04，交叉率0.6
        #print('total',total_distance)
        #print('fitness',fitness)
        #print('?',self.DNA_size * 2 / total_distance)
        return fitness_max, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            idx=[i for i,v in enumerate(cross_points)if v==True]
            keep_city = parent[~cross_points]                                       # find the city number
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[idx] = swap_city
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class TravelSalesPerson(object):
    def __init__(self, n_cities):
        city=[]
        tsp1=open("./TSP2.txt")
        for line in tsp1.readlines():
            curline=line.strip().split(" ")
            floatline=[float(curline[0]),float(curline[1])]
            print(floatline)
            city.append(floatline)
        city=np.array(city)

        self.city_position = city
        print(self.city_position.dtype)



    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)
        plt.show()

    def fitness_analyse(self,fitness):
        plt.xlabel('generations')
        plt.ylabel('fitness')
        plt.plot(range(N_GENERATIONS),fitness)
        plt.show()



if __name__=="__main__":
    plt.ion()
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

    env = TravelSalesPerson(N_CITIES)
    global_best_distance=0                                                                #算最大值时设为0，最小值设为100
    global_best_lx=np.zeros(N_CITIES)
    global_best_ly=np.zeros(N_CITIES)
    fitness_count=np.zeros(N_GENERATIONS)
    for generation in range(N_GENERATIONS):
        lx, ly = ga.translateDNA(ga.pop, env.city_position)
        fitness, total_distance = ga.get_fitness(lx, ly)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)
        print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],'| best_distance',total_distance[best_idx])
        if(total_distance[best_idx]>global_best_distance):                              #最大值为大于（>),最小值为小于（<)
            global_best_distance=total_distance[best_idx]
            global_best_lx=lx[best_idx]
            global_best_ly=ly[best_idx]
        #env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])
        fitness_count[generation]=fitness[best_idx]
    env.plotting(global_best_lx,global_best_ly,global_best_distance)
    plt.ioff()
    plt.show()
    env.fitness_analyse(fitness_count)
