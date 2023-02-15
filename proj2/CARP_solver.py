import queue
import random
import sys
import time

# -------Global variables------- #
inf = float("inf")
log_interval = 100


class Graph:
    def __init__(self, filename):
        self.nodes = set()
        self.edges = set()
        self.task_edges = set()
        self.shortest_cost = [{}]
        # shortest_cost[i][j]:the min cost from i to j
        with open(filename) as f:
            for line in f.readlines():
                line = line.strip('\n')
                headline = line.split(' : ')
                nodeline = line.split()
                if len(headline) == 2:
                    if headline[0] == 'VERTICES':
                        self.VERTICES = int(headline[1])
                    if headline[0] == 'DEPOT':
                        self.DEPOT = int(headline[1])
                    elif headline[0] == 'REQUIRED_EDGES':
                        self.REQUIRED_EDGES = int(headline[1])
                    elif headline[0] == 'NON_REQUIRED_EDGES':
                        self.NON_REQUIRED_EDGES = int(headline[1])
                    elif headline[0] == 'CAPACITY':
                        self.CAPACITY = int(headline[1])
                elif len(nodeline) == 4:
                    # from, to, cost, demand
                    self.edges.add((int(nodeline[0]), int(nodeline[1]), int(nodeline[2]), int(nodeline[3])))
                    self.edges.add((int(nodeline[1]), int(nodeline[0]), int(nodeline[2]), int(nodeline[3])))
                    if int(nodeline[3]):
                        self.task_edges.add((int(nodeline[0]), int(nodeline[1]), int(nodeline[2]), int(nodeline[3])))
                    self.nodes.add(int(nodeline[0]))
                    self.nodes.add(int(nodeline[1]))

        self.adjacency_list = {node: set() for node in self.nodes}
        for edge in self.edges:
            self.adjacency_list[edge[0]].add((edge[1], edge[2]))
            # from: to,cost

    def dij_shortest_path(self, start_node):
        unvisited_nodes = self.nodes.copy()
        distance_from_start = {
            node: (0 if node == start_node else inf) for node in self.nodes
        }
        previous_node = {node: None for node in self.nodes}
        while unvisited_nodes:
            current_node = min(
                unvisited_nodes, key=lambda node: distance_from_start[node]
            )
            unvisited_nodes.remove(current_node)
            if distance_from_start[current_node] == inf:
                break

            for neighbor, distance in self.adjacency_list[current_node]:
                new_path = distance_from_start[current_node] + int(distance)
                if new_path < distance_from_start[neighbor]:
                    distance_from_start[neighbor] = new_path
                    previous_node[neighbor] = current_node
        return distance_from_start

    def path_scanning(self, criteria):
        cost_ordered_edges = self.task_edges.copy()
        # task_edge: (from, to, cost, demand)
        route = []  # [(i1,j1,cost,demand),(i2,j2,cost,demand),... ]...
        cost = 0
        cheapest = []
        while len(cost_ordered_edges):
            pre = self.DEPOT
            new_route = []
            Q = self.CAPACITY
            while Q > 0:
                cost_ordered_edges = sorted(cost_ordered_edges, key=lambda tup: self.shortest_cost[pre][tup[0]])
                cheapest.append(cost_ordered_edges[0])
                for i in range(1,len(cost_ordered_edges)):
                    if self.shortest_cost[pre][cheapest[0][1]] == self.shortest_cost[pre][cost_ordered_edges[i][1]]:
                        cheapest.append(cost_ordered_edges[i])
                    else:
                        break
                if len(cheapest) > 1:
                    if criteria == 0:  # (1）每单位剩余需求的距离cii最小化；
                        cheapest = sorted(cheapest,key=lambda tup: tup[2] / tup[3])
                        current = cheapest[0]
                    elif criteria == 1:  # (2）每单位剩余需求的距离cii最大化；
                        cheapest = sorted(cheapest,key=lambda tup: tup[3] / tup[2])
                        current = cheapest[0]
                    elif criteria == 2:  # (3）获得更短的周期
                        cheapest = sorted(cheapest, key=lambda tup: self.shortest_cost[tup[1]][self.DEPOT])
                        current = cheapest[0]
                    elif criteria == 3:  # (4）获得更长的周期
                        cheapest = sorted(cheapest, key=lambda tup: self.shortest_cost[tup[1]][self.DEPOT],reverse=True)
                        current = cheapest[0]
                    elif criteria == 4:   # (5) 如果车辆不足半满，则将从节点j返回住所的距离最大化，否则将该距离最小化。
                        if Q < self.CAPACITY/2:
                            cheapest = sorted(cheapest, key=lambda tup: self.shortest_cost[tup[1]][self.DEPOT])
                        else:
                            cheapest = sorted(cheapest, key=lambda tup: self.shortest_cost[tup[1]][self.DEPOT],reverse=True)
                        current = cheapest[0]
                    else:  # Random selection
                        current = random.choices(cheapest)[0]
                else:
                    current = cost_ordered_edges[0]
                Q -= current[3]
                if Q < 0:
                    break
                cost_ordered_edges.remove(current)
                new_route.append(current)
                cost += self.shortest_cost[pre][current[0]]
                cost += current[2]
                pre = current[1]
                cheapest.clear()
                if not len(cost_ordered_edges):
                    break
            cost += self.shortest_cost[pre][self.DEPOT]
            route.extend(new_route)

        # 每个标准的基本原理如下。
        # 使用标准1和2，我们查看每个弧的成本/需求比率。该比率可以被视为维修电弧的单位需求成本。
        # 使用标准1，我们在每个步骤中最小化该比率，并寻求快速的大回报。相反，使用标准2，我们将比率最大化，寻求在早期产生更大的费用，从而将其排除在外。
        # 对于标准3，我们倾向于获得更短的周期，
        # 而标准4，通常产生更长的周期。我们指出，较长周期的次数越少，总距离越短。
        # 标准5表示混合方法。在五个标准中的每一个标准下，一旦车辆容量被充分利用，循环的剩余时间就由到住所的最短返回路径给出。选择最小的循环总距离集作为算法的结果。
        return tuple(route), cost


# -------------- genetic algorithm -----------


class GeneticAlgorithm:
    def __init__(self, pop_size, generation, mutation_rate, DEPOT, CAPACITY, shortest_cost):
        self.pop_size = pop_size
        self.generation = generation
        self.mutation_rate = mutation_rate
        self.DEPOT = DEPOT
        self.CAPACITY = CAPACITY
        self.shortest_cost = shortest_cost

    def formulate_route(self, route0):
        route = []
        cost = 0
        p = 0
        while p < len(route0):
            pre = self.DEPOT
            new_route = [0, ]
            Q = self.CAPACITY
            while Q > 0:
                current = route0[p]
                Q -= current[3]
                if Q < 0:
                    break
                p += 1
                new_route.append((current[0], current[1]))
                cost += self.shortest_cost[pre][current[0]]
                cost += current[2]
                pre = current[1]
                if p >= len(route0):
                    break
            cost += self.shortest_cost[pre][self.DEPOT]
            new_route.append(0)
            route.extend(new_route)
        return tuple(route), cost

    def initial_pop(self, graph) -> list:  # [((route1),cost1),((route2),cost2)]
        population = {}
        for i in range(6):
            route, cost = graph.path_scanning(i)
            population.update({route: cost})
        bound = len(population)
        for i in range(self.pop_size - bound):
            gene_pool = list(graph.task_edges.copy())
            route = []
            while(gene_pool):
                current = random.choices(gene_pool)[0]
                route.append(current)
                gene_pool.remove(current)
            _, cost = self.formulate_route(route)
            population.update({tuple(route): cost})
        population = sorted(population.items(), key=lambda x: x[1])  # ascending order, the smaller the better
        return population

    def select(self, r, population):
        individuals = set()
        all_fitness = 0
        for individual in population:
            all_fitness += individual[1]
            individuals.add((individual[1], individual[0]))
        added = set()
        cnt = 0
        best_r = []
        while cnt < r:
            for i in individuals:
                if cnt == r:
                    break
                if random.uniform(0, 1) >= (i[0] / all_fitness):
                    added.add(i)
                    best_r.append(i[1])
                    cnt += 1
            individuals = individuals - added
        return best_r

    def recombine(self, x0, y0):  # Order Crossover/Linear Order Crossover
        def recombine0(x, y):
            i = random.randrange(0, len(x))
            j = random.randrange(i, len(x))
            son = list(x)
            p = j + 1
            for k in range(len(y)):
                if y[k] not in son[i:j + 1] and (y[k][1], y[k][0], y[k][2], y[k][3]) not in son[i:j + 1]:
                    if p == len(son):
                        p = 0
                    son[p] = y[k]
                    p += 1
            return tuple(son)

        if random.randrange(0, 1) > 0.5:
            return recombine0(x0, y0)
        else:
            return recombine0(y0, x0)

    def mutate(self, x, pmut):  # Enhanced Local Search
        # Ｍ１－移动任务ｆ到任务ｇ后；
        # Ｍ２－移动２个连续任务（ｆ，ｉ）到任务ｇ后；
        # Ｍ３－交换任务ｆ和ｇ；
        # Ｍ４－交换任务ｆ和２个连续任务（ｇ，ｊ）；
        # Ｍ５－交换２个连续任务（ｆ，ｉ）和（ｇ，ｊ）；
        # Ｍ６－２－ｏｐｔ．
        if random.uniform(0, 1) > pmut:
            return x
        x = list(x)
        i = random.randrange(0, len(x))
        x[i] = (x[i][1], x[i][0], x[i][2], x[i][3])

        j = random.randrange(0, len(x))
        temp = x[j-1]
        x[j-1] = x[j]
        x[j] = temp

        return tuple(x)

    def reproduce(self, population):
        next_gen = []
        for _ in range(len(population)):
            new_route = self.mutate(self.recombine(*self.select(2, population)), self.mutation_rate)
            next_gen.append((new_route, self.formulate_route(new_route)[1]))
        return sorted(next_gen, key=lambda x: x[1])  # ascending order,the bigger the worse

    def replacement(self, old, new):
        current_best = old[0]
        replace_buffer = []
        # ascending order,the bigger the worse
        for i in range(3):
            new.pop()
            replace_buffer.append(old[i])
        for i in range(1, len(old) // 6):
            new.pop()
            new_one = self.mutate(current_best[0], 1)
            replace_buffer.append((new_one, self.formulate_route(new_one)[1]))
        new.extend(replace_buffer)
        return sorted(new, key=lambda x: x[1])


def main():
    filename = 'CARP_samples/gdb10.dat'
    seed = 10
    # filename = sys.argv[1]
    # seed = int(sys.argv[5])

    random.seed(seed)
    start = time.time()
    graph = Graph(filename)
    for node in range(1, graph.VERTICES + 1):
        graph.shortest_cost.append(graph.dij_shortest_path(node))
    GA = GeneticAlgorithm(mutation_rate=0.1, pop_size=30, generation=5000,
                          DEPOT=graph.DEPOT, CAPACITY=graph.CAPACITY, shortest_cost=graph.shortest_cost)
    # --------- initialize the population---------
    population = GA.initial_pop(graph)
    # [((route1),cost1),((route2),cost2),...]
    best = population[0]
    # --------- reproduce generations-------------
    for gen in range(GA.generation):
        next_gen = GA.reproduce(population)
        population = GA.replacement(population, next_gen)
        if gen % log_interval == 0 and population[0][1] < best[1]:
            best = population[0]
    # --------- formulate result------------------
    route, cost = GA.formulate_route(best[0])
    run_time = (time.time() - start)
    print(run_time)
    print('s', str(route).lstrip('(').strip(')').replace(' ', ''))
    print('q', cost)


if __name__ == "__main__":
    main()
