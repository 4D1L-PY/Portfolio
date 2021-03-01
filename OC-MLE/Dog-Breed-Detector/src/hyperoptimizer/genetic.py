import random
from deap import base, creator, tools, algorithms
from src.analyzer.univariate import np, pd, plt


class GeneticAlgorithmOptimizer:

    def __init__(self,
                 hyperparameters,
                 model_trainer,
                 population_size=10,
                 crossover_probability=0.7,
                 mutation_probability=0.1,
                 number_of_generations=5,
                 int_scale='binary'):  # Power of two
        """

        """
        self.hyperparameters = hyperparameters
        self.model_trainer = model_trainer
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.number_of_generations = number_of_generations
        self.int_scale = int_scale
        self.toolbox = None
        self.max_genes = None
        self.best_params = None
        self.pop = None
        self.log = None

    @staticmethod
    def randint_by_power_of_two_step(min_val, max_val):
        """
        """
        n = 1 << random.randrange(int(np.log2(min_val)), int(np.log2(max_val)) + 1)
        return n

    def build_individual_and_population(self,
                                        fitness_type='maximize',
                                        combined_genes_order=None,
                                        n_cycles=1):
        """

        """
        fitness_value = 1.0 if fitness_type is 'maximize' else -1.0
        creator.create("FitnessMax", base.Fitness,
                       weights=(fitness_value,))  # Maximise the fitness function value (accuracy)
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        # Possible parameter values
        randint_function = self.randint_by_power_of_two_step if self.int_scale is 'binary' else random.randint
        for hyperparam_label, hyperparam_value in self.hyperparameters.items():
            # define how each gene will be generated (e.g. criterion is a random choice from the criterion list).
            if type(hyperparam_value[0]) is str:
                self.toolbox.register(f"attr_{hyperparam_label}", random.choice, hyperparam_value)
            elif type(hyperparam_value[0]) is int:
                self.toolbox.register(f"attr_{hyperparam_label}", randint_function, *hyperparam_value)
            elif type(hyperparam_value[0]) is float:
                self.toolbox.register(f"attr_{hyperparam_label}", random.uniform, *hyperparam_value)
        # This is the order in which genes will be combined to create a chromosome
        if combined_genes_order is None:
            attributes_generator = tuple(attr for key, attr in self.toolbox.__dict__.items() if key.startswith('attr'))
        else:
            attributes = [attr for key, attr in self.toolbox.__dict__.items() if key.startswith('attr')]
            attributes_generator = tuple(attributes[i] for i in combined_genes_order)
            # TO DO
            # - add other forms of individual generation (initRepeat etc ... )
        self.max_genes = len(attributes_generator)
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attributes_generator, n=n_cycles)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    # TO DO : replace by toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
    def mutate(self, individual):
        """
        Randomly selects a gene and randomly generates a new value for it based on a set of rules

        :param individual: a individual chromosome

        :return: a mutated individual chromosome
        """
        gene_idx = random.randint(0, self.max_genes - 1)  # select which parameter to mutate
        # Get selected gene (= hyperparameter type)
        selected_gene = list(self.hyperparameters.values())[gene_idx]
        # Get selected gene data types (= hyperparameter values from hyperparameter type)
        selected_gene_data_types = list(set([type(hyperparameter_value) for hyperparameter_value in selected_gene]))
        if len(selected_gene_data_types) == 1:
            if selected_gene_data_types[0] is str:
                individual[gene_idx] = [val for val in selected_gene if val != individual[gene_idx]]
            elif selected_gene_data_types[0] is int:
                if self.int_scale is 'binary':
                    individual[gene_idx] = random.choice([val for val in selected_gene if val != individual[gene_idx]])
                elif self.int_scale is 'linear':
                    individual[gene_idx] = random.randint(*selected_gene)
                else:
                    return Exception("{} is invalid (choose between 'binary' or 'linear')".format(self.int_scale))
            elif selected_gene_data_types[0] is float:
                individual[gene_idx] = random.uniform(*selected_gene)
            return individual,
        else:
            return Exception('Hyperparameter values must have same datatype')

    def evaluate(self, individual, display_individual_genes=False):
        """
        Build and test a model based on the parameters in an individual and return evaluated metric

        :param individual: a individual chromosome

        :return: score from evaluated metric
        """
        #  Print the values of the parameters from the individual chromosome
        if display_individual_genes:
            print('\n')
            print('------------------------------------------')
            print('-           Selected genes               -')
            print('------------------------------------------')
            print('\n')
            for i, k in enumerate(self.hyperparameters.keys()):
                print(f'{k} : {individual[i]}')
        # Train model
        score = self.model_trainer(individual)
        return score,

    def initiate_natural_selection(self,
                                   crossover_type=tools.cxOnePoint,
                                   selection_type=tools.selTournament,
                                   selection_kwargs={'tournsize': 2},
                                   print_individual_genes=False,
                                   v=True):
        """

        """
        self.toolbox.register("mate", crossover_type)  # crossover
        self.toolbox.register("mutate", self.mutate)  # mutate
        self.toolbox.register("select", selection_type, **selection_kwargs)  # individuals selection
        self.toolbox.register("evaluate", self.evaluate,
                              display_individual_genes=print_individual_genes)  # individuals evaluation

        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        self.pop, self.log = algorithms.eaSimple(pop,
                                                 self.toolbox,
                                                 cxpb=self.crossover_probability,
                                                 stats=stats,
                                                 mutpb=self.mutation_probability,
                                                 ngen=self.number_of_generations,
                                                 halloffame=hof,
                                                 verbose=v)
        self.best_params = {hyperparam_label: hyperparam_val for hyperparam_label, hyperparam_val in zip(self.hyperparameters.keys(), hof[0])}
        return self.best_params, self.pop, self.log

    def plot_training_curve(self, xticks_step=5):
        evolution = pd.DataFrame({'Generation': self.log.select("gen"),
                                  'Max Accuracy': self.log.select("max"),
                                  'Average Accuracy': self.log.select("avg"),
                                  'Min Accuracy': self.log.select("min")})
        plt.title('Hyperparameter Optimisation')
        plt.plot(evolution['Generation'], evolution['Min Accuracy'], 'b', color='C1', label='Min')
        plt.plot(evolution['Generation'], evolution['Average Accuracy'], 'b', color='C2', label='Average')
        plt.plot(evolution['Generation'], evolution['Max Accuracy'], 'b', color='C3', label='Max')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('Generation')
        plt.xticks([x for x in range(0, self.number_of_generations + 1, xticks_step)])
        plt.show()
