import numpy as np

# -----------------------------
# Objective Function (Fitness)
# -----------------------------
def antenna_fitness(params):
    """
    Fitness function to evaluate antenna design.
    params: [spacing_1, spacing_2, phase_1, phase_2, ...]
    """
    n = len(params) // 2
    spacings = params[:n]
    phases = params[n:]
    
    # Desired main beam direction (radians)
    theta_0 = 0  
    
    # Compute array factor (simplified)
    theta = np.linspace(-np.pi/2, np.pi/2, 200)
    array_factor = np.zeros_like(theta, dtype=complex)
    
    for i in range(n):
        array_factor += np.exp(1j * (2 * np.pi * spacings[i] * np.sin(theta) + phases[i]))
    
    pattern = np.abs(array_factor)
    pattern /= np.max(pattern)  # Normalize
    
    # Fitness = minimize side lobe level (avoid peaks outside main lobe)
    main_lobe_region = (np.abs(theta - theta_0) < np.pi/8)
    sidelobes = pattern[~main_lobe_region]
    main_lobe = pattern[main_lobe_region]
    
    # Penalize side lobes and reward strong main lobe
    fitness = np.mean(sidelobes) / np.max(main_lobe)
    return fitness  # lower is better


# -----------------------------
# Particle Swarm Optimization
# -----------------------------
def pso(num_particles=30, dim=6, iterations=50):
    # PSO Hyperparameters
    w = 0.7      # inertia
    c1 = 1.5     # cognitive (personal best)
    c2 = 1.5     # social (global best)
    
    # Initialize particles
    positions = np.random.uniform(-1, 1, (num_particles, dim))
    velocities = np.random.uniform(-0.1, 0.1, (num_particles, dim))
    personal_best = positions.copy()
    personal_best_scores = np.array([antenna_fitness(p) for p in positions])
    
    global_best = personal_best[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)
    
    for t in range(iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            
            # Update velocity
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - positions[i])
                + c2 * r2 * (global_best - positions[i])
            )
            
            # Update position
            positions[i] += velocities[i]
            
            # Evaluate new fitness
            score = antenna_fitness(positions[i])
            if score < personal_best_scores[i]:
                personal_best[i] = positions[i].copy()
                personal_best_scores[i] = score
                
        # Update global best
        if np.min(personal_best_scores) < global_best_score:
            global_best = personal_best[np.argmin(personal_best_scores)]
            global_best_score = np.min(personal_best_scores)
        
        if t % 10 == 0:
            print(f"Iteration {t:02d} | Best fitness = {global_best_score:.5f}")
    
    return global_best, global_best_score


# -----------------------------
# Run PSO
# -----------------------------
best_params, best_score = pso()
print("\nOptimal antenna design parameters found:")
print(best_params)
print(f"Best fitness value: {best_score:.5f}")
