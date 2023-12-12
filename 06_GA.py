import random
import numpy as np
import warnings


warnings.filterwarnings('ignore')


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent1_idx = k % len(parents)
        parent2_idx = (k + 1) % len(parents)

        for i in range(crossover_point):
            offspring[k][i] = parents[parent1_idx][i]
            # print(i)

        for i in range(crossover_point, offspring_size[1]):
            offspring[k][i] = parents[parent2_idx][i]
            # print(i)
    return offspring


def fitness_assessment(population, y):
    p = []
    for i in range(len(population)):
        s = 0
        for j in range(len(diofantov_expr)):
            s += population[i][j] * diofantov_expr[j]
        r = np.abs(y - s) + 1
        p.append(1 / r)
    return p


def mutation_test(pop_after_cross, mutation_rate):
    population_nextgen = []
    for i in range(0, len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                random_value = np.random.randint(-1.0, 1.0, 1)
                chromosome[j] = chromosome[j] + random_value
        population_nextgen.append(chromosome)
    return population_nextgen


def parents_selection(population, parents_num, p):
    choose_parents = []
    for i in range(parents_num):
        max_ind = [j for j in range(len(p)) if p[j] == max(p)][0]
        choose_parents.append(population[max_ind])
        p.remove(max(p))
    return choose_parents


if __name__ == '__main__':
    diofantov_expr = [-2, 10, -3, 2, 7, -1]
    y = 200

    diofantov_weights = len(diofantov_expr)

    count_chromosome = 8

    population_size = (count_chromosome, diofantov_weights)
    new_population = np.random.randint(low=-len(diofantov_expr) * 3, high=len(diofantov_expr) * 3, size=population_size)

    count_iteration = 1000
    for iteration in range(count_iteration):
        fitness = fitness_assessment(new_population, y)

        new_parents = parents_selection(new_population, 6, fitness)
        new_offspring_cross = crossover(parents=new_parents, offspring_size=(population_size[0] - len(new_parents), diofantov_weights))
        new_offspring_mut = mutation_test(new_offspring_cross, 0.1)

        for i in range(len(new_parents)):
            new_population[i] = new_parents[i]

        current_count = 0

        for i in range(len(new_parents), len(new_parents) + len(new_offspring_mut)):
            new_population[i] = new_offspring_mut[current_count]
            current_count += 1

    fitness = fitness_assessment(new_population, y)

    max_fitness = max(fitness)
    need_index = fitness.index(max_fitness)

    print("Best: ", new_population[need_index])
    best_match_idx = np.where(fitness == np.max(fitness))
    # print(fitness)
    # print(best_match_idx)
    result_sum = 0
    for i in range(len(diofantov_expr)):
        result_sum += diofantov_expr[i] * new_population[need_index][i]
    print("Total sum:", result_sum)
    # print(new_population)
    # print(new_population[best_match_idx, :])
