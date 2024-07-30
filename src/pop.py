'''
Population & Individual Classes for TRNDP Problem
Jacob Emmerson & Chris Hinson
'''
import numpy as np # <3 numpy
import random
import json
import copy
from .lib import Line # genes

PEN = 1/2

class Individual ():
    '''
    Each individual has a chromosome representing a transit network
    '''
    def __init__(self, chromosome, fitness):
        chrom = []
        for gene in chromosome:
            new_line = Line(gene.name, gene.DOW, gene.geometry)
            new_line.stops = gene.stops
            new_line.start = gene.start
            new_line.end = gene.end
            chrom.append(new_line)

        self.chromosome = np.array(chrom)
        self.fitness = fitness
        self.gene_fitness = np.zeros(len(chromosome)) # initialize as zero

    def fit(self, demand_matrix, stop_indexes):
        fit = 0
        for i, gene in enumerate(self.chromosome):
            # for each line object
            stops = gene.stops
            gene.unique_stops = set(stops[:,0]) | {gene.start, gene.end}
            fr = [stop_indexes[s] for s in stops[:,0]]
            to = [stop_indexes[s] for s in stops[:,1]]
            demands = demand_matrix[fr,to]

            self.gene_fitness[i] = np.sum(demands) #- (PEN*len(stops)/gene.freq)
            fit += self.gene_fitness[i]

        self.fitness = fit

class Population ():
    '''
    Set of individuals with functions for the genetic algorithm
    '''
    def __init__(self, network, intial_individual, demand_matrix_path, stop_index_path, population_size = 100, args = {}):
        self.N = population_size
        self.N_lines = len(intial_individual)
        self.demands = np.load(demand_matrix_path)
        self.network = network # takes a Map

        with open(stop_index_path, 'r') as f:
            self.sti = {int(clever):index for clever,index in json.load(f).items()}

        self.individuals = [Individual(list(intial_individual.values()), 0) for n in range(population_size)]
        self.get_fitness() # initial fitness calculation, only needs to be called once

        for k,v in args:
            setattr(k, v)

    def sort_pop(self):
        self.individuals = sorted(self.individuals, key = lambda x: x.fitness, reverse = True) # descending, sort by fitness

    def get_fitness(self):
        '''
        Returns a list of size N with the fitness scores of every individual
        '''
        for ind in self.individuals:
            ind.fit(self.demands, self.sti) 

        return [ind.fitness for ind in self.individuals]
    
    def elite_selection(self, p_elite):
        '''
        Select the top n*p_elite individuals for the next generation
        MUST SORT BEFORE CALLING
        '''
        return self.individuals[:int(p_elite * self.N)] # most fit individuals
    
    def culling(self, p_cull):
        '''
        Occurs before crossover; prevents low quality solutions from being considered during tournament selection
        MUST SORT BEFORE CALLING
        '''
        if p_cull == 0:
            pass
        else:
            self.individuals = self.individuals[:-int(p_cull*self.N)] # population WITHOUT the least fit individuals

    def tournament_selection(self, k):
        '''
        Randomly sample k individuals from the population 
        '''
        # randomly sample k individuals and sort by fitness (descending)
        # random choice does NOT maintain order (hence sorting)
        selected = sorted(
            np.random.choice(self.individuals, size = k),
            key = lambda x: x.fitness,
            reverse = True
        )
        #print(selected)
        return selected[0:2] # return the top two most fit from random sample

    def big_mutation(self,individual):
        '''
        Takes a single individual and randomy expands a terminal node
        '''
        least_fit = np.argmin(individual.gene_fitness) # index of the least fit gene
        mute = individual.chromosome[least_fit] # gene to expand
        node = np.random.randint(0,2) # coin flip, 0 = start, 1 = end

        if node == 0: # mutate start
            n = mute.start
        else: # mutate end
            n = mute.end

        choices = self.network[n] # possible expansions from terminal node
        ni = self.sti[n]
        ci = [self.sti[c] for c in choices]
   
        # get our demands (order matters)
        if node == 0: # start
            demand = self.demands[ci,ni] # (1,len(choices))
        else:
            demand = self.demands[ni,ci]

        # greedily select the best
        best = np.argmax(demand)
        best_fit = demand[best]
        best_stop = choices[best]
        mute.unique_stops.add(best_stop)

        # update gene
        if node == 0: # start
            mute.stops = np.append(mute.stops,[[best_stop, n]],axis=0)
            mute.start = best_stop
        else:
            mute.stops = np.append(mute.stops,[[n, best_stop]],axis=0)
            mute.end = best_stop

        # update fitness
        individual.fitness += best_fit# - (PEN/mute.freq)
        individual.gene_fitness[least_fit] += best_fit# - (PEN/mute.freq)

        return individual
    
    def midpoint_mutation(self, individual):
        '''
        Takes a single individual an randomly mutates a midpoint
        '''
        least_fit = np.argmin(individual.gene_fitness) # index of the least fit gene
        mute = individual.chromosome[least_fit]

        # stop indexes
        fr = [self.sti[s] for s in mute.stops[:,0]]
        to = [self.sti[s] for s in mute.stops[:,1]]
        demand = self.demands[fr,to]

        # midpoint to mutate
        stop = np.random.randint(0,demand.shape[0]) # index of the stop to mutate
        old_demand = demand[stop]

        source = mute.stops[:,0][stop]
        sink = mute.stops[:,1][stop]

        new = mute.start
        while new in mute.unique_stops:
            new = np.random.choice(self.network[source]) # select a non terminal node

        # stop indexes
        fi = self.sti[source]
        ni = self.sti[new]
        ti = self.sti[sink]

        new_demands = self.demands[[fi,ni],[ni,ti]]

        # change the chromosome
        mute.stops[stop,:] = [source,new]
        mute.stops = np.append(mute.stops,[[new,sink]],axis=0)
        mute.unique_stops.add(new)

        # update fitness
        delta_demand = np.sum(new_demands) - old_demand# - (PEN/mute.freq)
        individual.fitness += delta_demand
        individual.gene_fitness[least_fit] += delta_demand

        return individual

    def mutate(self, individual, p_mutate):
        '''
        p_mutate > 0.5
        p_mutate = probability of a midpoint mutation; if not a midpoint mutation do a big mutation
        '''
        assert p_mutate > 0.5

        p = random.random() # [0,1]
        #print(f"{p = }")
        if p > p_mutate: # big mutation
            #print("BIG")
            return self.big_mutation(individual)

        else:
            #print("SMALL")
            return self.midpoint_mutation(individual)
    
    def crossover(self, p_crossover, parents = []):
        '''
        In the final implementation, we don't have to calculate fitness each generation. When swapping two genes, we can alter the stored fitnesses as well
        '''
        # must have two parents for reproduction
        assert len(parents) == 2
        
        # make two children from two parents (might need to change to deep copy)
        child1 = Individual(parents[0].chromosome, parents[0].fitness)
        child1.gene_fitness = parents[0].gene_fitness

        child2 = Individual(parents[1].chromosome, parents[1].fitness)
        child2.gene_fitness = parents[1].gene_fitness

        n_to_swap = int(p_crossover * self.N_lines)
        swap_indicies = np.random.randint(0, self.N_lines, n_to_swap) # generates n_to_swap integers between [0, n_lines)
        #print(f"{swap_indicies = }")
        # list of genes to swap
        g1 = child1.chromosome[swap_indicies]
        g2 = child2.chromosome[swap_indicies]

        g1_fits = child1.gene_fitness[swap_indicies] # fitness of each gene we are swapping
        g2_fits = child2.gene_fitness[swap_indicies]

        delta_g1 = np.sum(g2_fits) - np.sum(g1_fits) # total fitness changes
        delta_g2 = np.sum(g1_fits) - np.sum(g2_fits)

        # swap the genes
        child1.chromosome[swap_indicies] = g2
        child2.chromosome[swap_indicies] = g1

        # adjust the fitnesses
        child1.fitness += delta_g1
        child2.fitness += delta_g2

        child1.gene_fitness[swap_indicies] = g2_fits
        child2.gene_fitness[swap_indicies] = g1_fits

        # return the new individuals
        return (child1, child2)
    
    def create_generation(self, p_cull = 0.05, p_elite = 0.05, p_crossover = 0.5, p_mutate = 0.1, p_mutate_type = 0.9, k_select = 5):

        # sort and cull the least fit
        self.sort_pop()
        self.culling(p_cull=p_cull)

        # start the new population with the individuals we are pushing forward
        elites = copy.deepcopy(self.elite_selection(p_elite=p_elite)) # deep copy to make "new" children
        new_population = []

        # may not round nicely, we can just discard the least fit individuals until we reach N individuals
        total_iter = int((self.N * (1 - p_elite))/2)
        for i in range(total_iter+1): # exclusive stop, add 1
            parents = self.tournament_selection(k = k_select)
            children = self.crossover(p_crossover=p_crossover, parents=parents)
            new_population += children # add the children to the new population

        new_population = np.array(new_population)
        # mutate
        # p_mutate = proportion of individuals to mutate; p_mutate_type passed to self.mutate()
        n_mute = int(self.N * p_mutate)
        mute_idx = np.random.randint(0,new_population.shape[0],n_mute)
        #print(f"{mute_idx = }")
        mutations = [self.mutate(i,p_mutate_type) for i in new_population[mute_idx]] # mutations alter objects, no need to append
        new_population = np.concatenate((new_population, elites))
        
        # check number of individuals
        if len(new_population) > self.N:
            new_population = new_population[-(self.N):] # if too many individuals, get the most recent N

        #print(new_population[0])
        #print(new_population[1])
        #self.get_fitness() # update the fitness of every individual
        return new_population
    
    def run_generations(self, n_generations, n_to_return, **kwargs):
        # how many generations
        for gen in range(n_generations):
            new_pop = self.create_generation(**kwargs) # can pass parameters
            self.individuals = new_pop

        self.sort_pop()
        return self.individuals[:n_to_return] # return the top n-individuals