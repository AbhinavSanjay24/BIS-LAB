import random

# Target number
TARGET = 42

# Fitness function: how close is the number to the target?
def fitness(num):
    return abs(TARGET - num)

# Mutation: randomly add or subtract 1 from the number
def mutate(num):
    return num + random.choice([-1, 1])

# Simple evolutionary loop
def evolve_number(generations=20):
    current = random.randint(0, 100)  # start from random number
    print(f"Starting number: {current}, fitness: {fitness(current)}")

    for gen in range(generations):
        new = mutate(current)
        if fitness(new) < fitness(current):
            current = new  # accept mutation if better
        print(f"Gen {gen+1}: number = {current}, fitness = {fitness(current)}")

evolve_number()