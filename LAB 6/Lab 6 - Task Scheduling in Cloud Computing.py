import random

# --- Problem setup ---
num_tasks = 5
num_machines = 3
population_size = 6
max_iterations = 10

# Simulated task durations (for each task)
task_durations = [5, 8, 3, 6, 4]

# Generate a random schedule (each task assigned to a machine)
def random_schedule():
    return [random.randint(0, num_machines - 1) for _ in range(num_tasks)]

# Calculate fitness (lower is better)
# We'll use "load imbalance" as fitness: difference between max and min machine load
def fitness(schedule):
    machine_loads = [0] * num_machines
    for i, machine in enumerate(schedule):
        machine_loads[machine] += task_durations[i]
    return max(machine_loads) - min(machine_loads)

# Initialize population (wolves)
wolves = [random_schedule() for _ in range(population_size)]

# Evaluate all wolves
fitness_values = [fitness(w) for w in wolves]

# Identify alpha, beta, delta (top 3)
def update_hierarchy(wolves, fitness_values):
    sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
    alpha = wolves[sorted_indices[0]]
    beta  = wolves[sorted_indices[1]]
    delta = wolves[sorted_indices[2]]
    return alpha, beta, delta

alpha, beta, delta = update_hierarchy(wolves, fitness_values)

# --- Main optimization loop ---
for iteration in range(max_iterations):
    a = 2 - (2 * iteration / max_iterations)  # exploration/exploitation control

    for i in range(population_size):
        new_schedule = []
        for t in range(num_tasks):
            # Each wolf follows alpha, beta, delta in spirit
            choices = [alpha[t], beta[t], delta[t]]
            # With some random exploration
            if random.random() < a / 2:
                new_task_assignment = random.choice(choices)
            else:
                new_task_assignment = random.randint(0, num_machines - 1)
            new_schedule.append(new_task_assignment)

        # Replace if better
        if fitness(new_schedule) < fitness(wolves[i]):
            wolves[i] = new_schedule

    # Update hierarchy
    fitness_values = [fitness(w) for w in wolves]
    alpha, beta, delta = update_hierarchy(wolves, fitness_values)

    print(f"Iteration {iteration+1} | Best fitness: {fitness(alpha)} | Best schedule: {alpha}")

print("\nFinal Best Schedule (Alpha Wolf):", alpha)
