import numpy as np

# ---------------------------------
# Distance Matrix (example with 5 cities)
# ---------------------------------
distance_matrix = np.array([
    [0, 2, 9, 10, 7],
    [1, 0, 6, 4, 3],
    [15, 7, 0, 8, 9],
    [6, 3, 12, 0, 11],
    [9, 7, 5, 6, 0]
])

num_cities = distance_matrix.shape[0]
num_ants = 10
num_iterations = 50

# ACO Parameters
alpha = 1.0   # influence of pheromone
beta = 2.0    # influence of distance
rho = 0.5     # pheromone evaporation rate
Q = 100       # pheromone deposit constant

# Initialize pheromone levels
pheromone = np.ones((num_cities, num_cities))

# ---------------------------------
# Helper Functions
# ---------------------------------
def route_length(route):
    """Compute total length of a given route"""
    length = 0
    for i in range(len(route) - 1):
        length += distance_matrix[route[i], route[i+1]]
    length += distance_matrix[route[-1], route[0]]  # return to start
    return length

def choose_next_city(pheromone, visibility, alpha, beta, visited, current_city):
    """Roulette-wheel selection for next city"""
    probs = []
    for j in range(num_cities):
        if j not in visited:
            prob = (pheromone[current_city][j] ** alpha) * (visibility[current_city][j] ** beta)
            probs.append(prob)
        else:
            probs.append(0)
    probs = np.array(probs)
    probs /= probs.sum()
    return np.random.choice(range(num_cities), p=probs)

# Visibility = 1 / distance (attractiveness)
visibility = 1 / (distance_matrix + np.eye(num_cities))
np.fill_diagonal(visibility, 0)

best_route = None
best_length = np.inf

# ---------------------------------
# Main ACO Loop
# ---------------------------------
for iteration in range(num_iterations):
    all_routes = []
    all_lengths = []
    
    for ant in range(num_ants):
        route = [np.random.randint(num_cities)]
        while len(route) < num_cities:
            next_city = choose_next_city(pheromone, visibility, alpha, beta, route, route[-1])
            route.append(next_city)
        all_routes.append(route)
        all_lengths.append(route_length(route))
    
    # Update pheromones
    pheromone *= (1 - rho)
    for route, length in zip(all_routes, all_lengths):
        for i in range(num_cities - 1):
            pheromone[route[i]][route[i+1]] += Q / length
        pheromone[route[-1]][route[0]] += Q / length  # return edge
    
    # Track best
    min_length = min(all_lengths)
    if min_length < best_length:
        best_length = min_length
        best_route = all_routes[np.argmin(all_lengths)]
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration:02d} | Best Length = {best_length:.2f}")

# ---------------------------------
# Final Output
# ---------------------------------
print("\nOptimal route found:")
print(" -> ".join(str(c) for c in best_route + [best_route[0]]))
print(f"Total distance: {best_length:.2f}")
