import numpy as np
import random


class genetic_algorithm():
    def __init__(self, init_chromosome_list, num_of_population=10, extend_range=0.01):
        self.chromosomes = init_chromosome_list
        self.number_of_population = num_of_population
        self.extend_range = extend_range
        self.population_list = []

    def make_population(self):
        for i in range(self.number_of_population):
            temp = []
            for chromosome in self.chromosomes:
                shape_of_chromosome = np.shape(chromosome)
                population = np.random.uniform(low=-0.01, high=0.01, size=len(np.reshape(chromosome, (-1))))
                population = np.reshape(population, (shape_of_chromosome[0], shape_of_chromosome[1]))
                temp.append(population)

            temp.append([0])
            self.population_list.append(temp)

        return self.population_list

    def set_fitness(self, index, fitness):
        self.population_list[index][-1] = fitness

    def extend_line_crossover(self, x, y):
        if x <= y:
            return random.uniform(x-self.extend_range, y+self.extend_range)
        else:
            return random.uniform(y-self.extend_range, x+self.extend_range)

    def generation(self, num_of_replacement):
        self.population_list = sorted(self.population_list, key=lambda x: x[-1], reverse=True)
        print("Best Accuracy : " + str(self.population_list[0][-1]))

        best_population = self.population_list[0]
        offspring_set = []
        for num in range(num_of_replacement):
            selected_population = self.selection()
            offspring = self.crossover(selected_population)
            offspring_set.append(offspring)

        self.replacement(offspring_set)

        return self.population_list, best_population

    def crossover(self, selected_list):
        offspring_list = []
        for i in range(len(selected_list[0]) - 1):
            list_shape = np.shape(selected_list[0][i])
            parent1 = np.reshape(selected_list[0][i], (-1))
            parent2 = np.reshape(selected_list[1][i], (-1))

            offspring = []
            for ele_index in range(len(parent1)):
                offspring_element = self.extend_line_crossover(parent1[ele_index], parent2[ele_index])
                offspring.append(offspring_element)

            offspring = np.reshape(offspring, (list_shape[0], list_shape[1]))
            offspring_list.append(offspring)

        offspring_list.append([0])
        return offspring_list

    def mutation(self, offspring=[]):
        random_element = random.choice(offspring[:-1])
        offspring[offspring.index(random_element)] = random_element + np.random.normal()
        return offspring

    def selection(self):
        sum_of_fitness = 0
        for fitness in self.population_list:
            sum_of_fitness += fitness[-1]

        selection_list = []
        for count in range(2):
            sum = 0
            point = np.random.uniform(0, sum_of_fitness)
            for choice in enumerate(self.population_list):
                sum += choice[1][-1]
                if point < sum:
                    selection_list.append(self.population_list[choice[0]])
                    break

        return selection_list

    def replacement(self, selection_list):
        for i in range(len(selection_list)):
            self.population_list[-(i+1)] = selection_list[-(i+1)]

    def test_print(self):
        for i in self.population_list:
            print(i[-1])
