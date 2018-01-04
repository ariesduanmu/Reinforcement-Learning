from numpy import random, sqrt, exp, dot, argmax
from random import choices, choice, uniform
import pickle
import os

class Genome():
    def __init__(self, weights = [random.randn(x,y)/sqrt(x) for x, y in zip([9,18,36],[18,36,9])],
                       biases = [random.randn(1,y) for y in [18,36,9]],
                       fitness = 0):
        self.biases = biases
        self.weights = weights
        self.fitness = fitness

#train 10 games test 1 games to get fitness for that genomes
class Genetic:
    def __init__(self):
        self.mutation_rate = 0.2
        self.mutation_step = 0.2

        self.current_genome = -1
        self.population_size = 10
        self.genomes = []
        self.read_dataset()
        self.evaluate_next_genome()

    def update(self, score):
        self.genomes[self.current_genome].fitness = score
        self.evaluate_next_genome()

    def predict(self, input_data):
        genome = self.genomes[self.current_genome]
        output = input_data
        for i in range(3):
            weight = genome.weights[i]
            biase = genome.biases[i]
            output = self.sigmoid(dot(output, weight) + biase)

        choice = argmax(output)
        row = choice // 3
        col = choice % 3
        return [row, col]

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def evaluate_next_genome(self):
        self.current_genome += 1
        if self.current_genome >= self.population_size:
            self.evolve()

    def evolve(self):
        self.current_genome = 0
        self.genomes = sorted(self.genomes, key = lambda x: -x.fitness)
        while len(self.genomes) > self.population_size // 2:
            self.genomes.pop()
        children = [self.genomes[0]]
        while len(children) < self.population_size:
            children += [self.make_child(choices(self.genomes, k=2))]
        self.genomes = children

    def make_child(self, parents):
        mum, dad = parents
        weights = [choice([mum.weights[i], dad.weights[i]]) for i in range(3)]
        biases = [choice([mum.biases[i], dad.biases[i]]) for i in range(3)]
        child = Genome(weights = weights,
                       biases = biases)
        if uniform(0, 1) < self.mutation_rate:
            child.weights[0] += random.randn(9,18) * self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.weights[1] += random.randn(18,36) * self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.weights[2] += random.randn(36,9) * self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.biases[0] += random.randn(1,18) * self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.biases[1] += random.randn(1,36) * self.mutation_step
        if uniform(0, 1) < self.mutation_rate:
            child.biases[2] += random.randn(1,9) * self.mutation_step
        return child

    def save_dataset(self):
        with open('genomes', 'wb+') as f:
            pickle.dump((self.genomes, self.current_genome), f, -1)
    def read_dataset(self):
        if not os.path.isfile('genomes'):
            self.genomes = [Genome() for _ in range(self.population_size)]
            
        else:
            with open('genomes', 'rb') as f:
                self.genomes, self.current_genome = pickle.load(f)
