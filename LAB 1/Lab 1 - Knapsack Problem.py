import random

# --- Problem Definition ---
values = [60, 100, 120, 90, 75]
weights = [10, 20, 30, 25, 15]
capacity = 50
pop_size = 10      # number of chromosomes
generations = 50   # number of iterations
mutation_rate = 0.1

n_items = len(values)

# --- Fitness Function ---
def fitness(individual):
    total_value = sum(values[i] for i in range(n_items) if individual[i] == 1)
    total_weight = sum(weights[i] for i in range(n_items) if individual[i] == 1)
    if total_weight > capacity:
        return 0  # invalid solution
    return total_value

# --- Generate Initial Population ---
def generate_population():
    return [[random.randint(0, 1) for _ in range(n_items)] for _ in range(pop_size)]

# --- Selection (Tournament) ---
def selection(pop, fitnesses):
    i, j = random.sample(range(pop_size), 2)
    return pop[i] if fitnesses[i] > fitnesses[j] else pop[j]

# --- Crossover (Single Point) ---
def crossover(parent1, parent2):
    point = random.randint(1, n_items - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# --- Mutation ---
def mutate(individual):
    for i in range(n_items):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# --- Main GA Loop ---
population = generate_population()

for gen in range(generations):
    fitnesses = [fitness(ind) for ind in population]
    new_population = []

    # Elitism (keep the best)
    best_idx = fitnesses.index(max(fitnesses))
    best_individual = population[best_idx]
    new_population.append(best_individual)

    # Generate new offspring
    while len(new_population) < pop_size:
        parent1 = selection(population, fitnesses)
        parent2 = selection(population, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1))
        if len(new_population) < pop_size:
            new_population.append(mutate(child2))

    population = new_population

# --- Final Result ---
fitnesses = [fitness(ind) for ind in population]
best_idx = fitnesses.index(max(fitnesses))
best_solution = population[best_idx]

print("Best Solution (0/1 vector):", best_solution)
print("Total Value:", fitness(best_solution))
print("Total Weight:", sum(weights[i] for i in range(n_items) if best_solution[i] == 1))
