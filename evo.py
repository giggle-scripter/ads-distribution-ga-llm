"""
Evolution Module - Genetic Algorithm Implementation

This module implements a genetic algorithm (GA) for solving the advertising distribution problem.
It includes both a standard GA and an LLM-enhanced GA that uses large language models to
suggest improvements when the algorithm gets stuck.

Key Components:
- Individual: Represents a solution (chromosome) with fitness evaluation
- Population: Collection of individuals
- Genetic Operations: Crossover, mutation, selection
- GA Functions: Standard and LLM-enhanced genetic algorithms
"""

import random
from typing import Literal
from problem import Problem
from llm_support import LLMSupporter, apply_transformations


class Individual:
    """
    Represents a single solution (individual) in the genetic algorithm.
    
    An individual contains:
    - chromosome: The solution representation (list of ad assignments)
    - fitness: The quality score of this solution
    
    Fitness Calculation:
    - Heavily penalizes constraint violations (-100,000 per violation)
    - Rewards revenue generation (+1 per unit revenue)
    - Rewards slot utilization (+10 per assigned slot)
    """
    
    # Fitness calculation constants
    DEFAULT_FITNESS = -1e8      # Default fitness for uninitialized individuals
    VIOLATIONS_FACTOR = -1e6    # Heavy penalty for constraint violations
    REVENUE_FACTOR = 1          # Reward for revenue generation
    ASSIGNED_CNT_FACTOR = 10    # Reward for assigning more slots
    BUDGET_PENALTY_FACTOR = 0   # Unnecessary in this context
    
    def __init__(self):
        """Initialize an empty individual with default fitness."""
        self.chromosome = None                    # Solution representation
        self.fitness = Individual.DEFAULT_FITNESS # Quality score
        
    def cal_fitness(self, problem: Problem) -> float:
        """
        Calculate and store the fitness value for this individual.
        
        Fitness combines multiple objectives:
        1. Minimize constraint violations (most important)
        2. Maximize revenue generation
        3. Maximize number of assigned slots
        
        Args:
            problem (Problem): The problem instance to evaluate against
            
        Returns:
            float: The calculated fitness value (higher is better)
        """
        # Calculate individual components
        violations = problem.cal_violations(self.chromosome)
        revenue = problem.cal_revenue(self.chromosome)
        assigned_cnt = problem.cal_assigned_cnt(self.chromosome)
        budget_penalty = problem.cal_budget_penalty(self.chromosome)
        
        # Combine components with weighted factors
        self.fitness = (Individual.VIOLATIONS_FACTOR * violations +    # Penalize violations heavily
                       Individual.REVENUE_FACTOR * revenue +           # Reward revenue
                       Individual.ASSIGNED_CNT_FACTOR * assigned_cnt +
                       Individual.BUDGET_PENALTY_FACTOR * budget_penalty)  # Reward slot usage
        
        return self.fitness
            
    def random_generate(self, problem: Problem):
        """
        Generate a random solution for this individual.
        
        The random generation process:
        1. Start with all slots unassigned (-1)
        2. Randomly select slots to assign (up to number of available ads)
        3. Assign ads sequentially to selected slots
        
        This ensures no duplicate ad assignments while allowing some slots to remain empty.
        
        Args:
            problem (Problem): The problem instance to generate solution for
        """
        # Initialize all slots as unassigned
        slot_assigned = [-1] * problem.num_slots
        
        # Randomly select which slots to assign (can't assign more ads than we have)
        num_to_assign = min(problem.num_slots, problem.num_ads)
        slots_to_assign = random.sample(range(problem.num_slots), k=num_to_assign)
        
        # Assign ads to selected slots sequentially
        for ads_id, slot_idx in enumerate(slots_to_assign):
            slot_assigned[slot_idx] = ads_id
            
        self.chromosome = slot_assigned.copy()


class Population:
    """
    Represents a collection of individuals (solutions) in the genetic algorithm.
    
    The population maintains a fixed size and provides methods for:
    - Random initialization
    - Managing the collection of individuals
    """
    
    def __init__(self, size: int):
        """
        Initialize a population with specified size.
        
        Args:
            size (int): Number of individuals in the population
        """
        self.size = size  # Population size
        self.inds = []    # List of Individual objects

    def random_generate(self, problem: Problem):
        """
        Generate a random population of solutions.
        
        Creates 'size' number of individuals, each with a random solution
        and calculated fitness.
        
        Args:
            problem (Problem): The problem instance to generate solutions for
        """
        for i in range(self.size):
            # Create new individual
            new_individual = Individual()
            
            # Generate random solution and calculate fitness
            new_individual.random_generate(problem)
            new_individual.cal_fitness(problem)
            
            # Add to population
            self.inds.append(new_individual)


def crossover(p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
    """
    Create two offspring from two parent individuals using uniform crossover.
    
    Uniform crossover works by:
    1. For each slot position, randomly choose which parent to inherit from
    2. 50% chance to inherit from parent 1, 50% chance from parent 2
    3. Create two children with swapped inheritance patterns
    
    Args:
        p1 (Individual): First parent individual
        p2 (Individual): Second parent individual
        
    Returns:
        tuple[Individual, Individual]: Two offspring individuals
    """
    # Create empty offspring
    c1 = Individual()
    c2 = Individual()
    
    num_slots = len(p1.chromosome)
    
    # Initialize offspring chromosomes
    c1.chromosome = [-1] * num_slots
    c2.chromosome = [-1] * num_slots
    
    # Uniform crossover: randomly inherit from each parent
    for i in range(num_slots):
        if random.random() < 0.5:
            # Inherit normally
            c1.chromosome[i] = p1.chromosome[i]
            c2.chromosome[i] = p2.chromosome[i]
        else:
            # Inherit swapped
            c1.chromosome[i] = p2.chromosome[i]
            c2.chromosome[i] = p1.chromosome[i]
            
    return c1, c2


def mutation(p: Individual, problem: Problem) -> Individual:
    """
    Create a mutated version of an individual.
    
    Mutation process:
    1. Select a random slot to mutate
    2. 30% chance: Unassign the slot (set to -1)
    3. 70% chance: Assign a different random ad to the slot
    
    Args:
        p (Individual): Parent individual to mutate
        problem (Problem): Problem instance for context
        
    Returns:
        Individual: Mutated offspring individual
    """
    # Create copy of parent
    c = Individual()
    c.chromosome = p.chromosome.copy()
    
    # Select random slot to mutate
    slot_idx = random.choice(range(len(c.chromosome)))
    
    if random.random() < 0.3:
        # Unassign the slot (30% chance)
        c.chromosome[slot_idx] = -1
    else:
        # Assign a different random ad (70% chance)
        # Get all possible ads except the currently assigned one
        current_ad = c.chromosome[slot_idx]
        available_ads = [x for x in range(problem.num_ads) if x != current_ad]
        
        if available_ads:  # Make sure we have options
            new_ad = random.choice(available_ads)
            c.chromosome[slot_idx] = new_ad
    
    return c


def topk_selection(population: Population, k: int, select_size: int) -> list[Individual]:
    """
    Select individuals using tournament selection.
    
    Tournament selection process:
    1. Randomly sample 'k' individuals from population
    2. Sort them by fitness (best first)
    3. Return the top 'select_size' individuals
    
    This balances between selection pressure and diversity.
    
    Args:
        population (Population): Population to select from
        k (int): Tournament size (number of candidates)
        select_size (int): Number of individuals to select
        
    Returns:
        list[Individual]: Selected individuals sorted by fitness
    """
    # Randomly sample candidates for tournament
    tournament_size = min(k, population.size)
    candidates = random.sample(population.inds, k=tournament_size)
    
    # Sort by fitness (highest first)
    candidates.sort(key=lambda x: x.fitness, reverse=True)
    
    # Return top performers
    return candidates[:select_size]


def ga(num_gen: int, pop_size: int, problem: Problem,
       pc: float = 0.8, pm: float = 0.1, elite_ratio: float = 0.1) -> Individual:
    """
    Standard Genetic Algorithm implementation.
    
    The algorithm follows these steps each generation:
    1. Create offspring through crossover and mutation
    2. Preserve elite individuals (best performers)
    3. Form new population from elite + selected offspring
    4. Track and report progress
    
    Args:
        num_gen (int): Number of generations to run
        pop_size (int): Population size
        problem (Problem): Problem instance to solve
        pc (float): Crossover probability (0.8 = 80% chance)
        pm (float): Mutation probability (0.1 = 10% chance)
        elite_ratio (float): Fraction of population to preserve as elite (0.1 = 10%)
        
    Returns:
        Individual: Best solution found
    """
    # Initialize population
    population = Population(pop_size)
    population.random_generate(problem)
    
    # Calculate elite count (minimum 1)
    elite_cnt = int(max(1, pop_size * elite_ratio))
    best = None
    
    # Evolution loop
    for gen in range(num_gen):
        offspring = []
        
        # Generate offspring until we have enough
        while len(offspring) < pop_size:
            # Crossover with probability pc
            if random.random() < pc:
                # Select parents using tournament selection
                parent1, parent2 = topk_selection(population, k=6, select_size=2)
                
                # Create offspring through crossover
                child1, child2 = crossover(parent1, parent2)
                
                # Apply mutation with probability pm
                if random.random() < pm:
                    child1 = mutation(child1, problem)
                if random.random() < pm:
                    child2 = mutation(child2, problem)
                
                # Evaluate offspring fitness
                child1.cal_fitness(problem)
                child2.cal_fitness(problem)
                
                # Add to offspring pool
                offspring.extend([child1, child2])
                
        # Elite preservation and population replacement
        sorted_population = sorted(population.inds, key=lambda x: x.fitness, reverse=True)
        elite_individuals = sorted_population[:elite_cnt]
        non_elite_individuals = sorted_population[elite_cnt:]
        
        # Combine non-elite with offspring and randomly sample
        replacement_pool = non_elite_individuals + offspring
        selected_non_elite = random.sample(replacement_pool, k=population.size - elite_cnt)
        
        # Form new population
        population.inds = elite_individuals + selected_non_elite
        
        # Update best solution
        current_best = population.inds[0]  # Elite is sorted, so first is best
        if best is None or current_best.fitness > best.fitness:
            best = current_best
            
        # Progress reporting
        sol = best.chromosome
        violations = problem.cal_violations(sol)
        revenue = problem.cal_revenue(sol)
        assigned_count = problem.cal_assigned_cnt(sol)
        budget_penalty = problem.cal_budget_penalty(sol)
        
        print(f'Gen {gen+1}, best fitness = {best.fitness:.2f}, '
              f'violations = {violations}, revenue = {revenue:.2f}, '
              f'budget penalty = {budget_penalty} '
              f'assigned slots = {assigned_count}')
        
    return best


def llm_ga(num_gen: int, pop_size: int, 
           problem: Problem, llm_supporter: LLMSupporter, 
           pc: float = 0.8, pm: float = 0.1, elite_ratio: float = 0.1,
           max_no_improvement: int = 10, max_transform_inds: int = 5,
           transform_chosen_policy: Literal['rand', 'topk'] = 'rand',
           max_time_transform: int = 5) -> Individual:
    """
    LLM-Enhanced Genetic Algorithm implementation.
    
    This extends the standard GA with LLM-based improvement suggestions:
    - Monitors for stagnation (no improvement for several generations)
    - When stuck, uses LLM to suggest solution transformations
    - Applies transformations to selected individuals
    - Continues evolution with enhanced population
    
    Args:
        num_gen (int): Number of generations to run
        pop_size (int): Population size  
        problem (Problem): Problem instance to solve
        llm_supporter (LLMSupporter): LLM interface for transformations
        pc (float): Crossover probability
        pm (float): Mutation probability  
        elite_ratio (float): Elite preservation ratio
        max_no_improvement (int): Generations without improvement before LLM intervention
        max_transform_inds (int): Number of individuals to transform with LLM
        transform_chosen_policy (str): How to choose individuals ('rand' or 'topk')
        max_time_transform (int): Maximum number of LLM interventions
        
    Returns:
        Individual: Best solution found
    """
    # Initialize tracking variables
    no_improvement = 0    # Generations without improvement
    transform_cnt = 0     # Number of LLM interventions used
    
    # Initialize population
    population = Population(pop_size)
    population.random_generate(problem)
    elite_cnt = int(max(1, pop_size * elite_ratio))
    best = None
    
    # Evolution loop
    for gen in range(num_gen):
        offspring = []
        
        # Standard GA operations (same as regular GA)
        while len(offspring) < pop_size:
            if random.random() < pc:
                parent1, parent2 = topk_selection(population, k=6, select_size=2)
                child1, child2 = crossover(parent1, parent2)
                
                if random.random() < pm:
                    child1 = mutation(child1, problem)
                if random.random() < pm:
                    child2 = mutation(child2, problem)
                
                child1.cal_fitness(problem)
                child2.cal_fitness(problem)
                
                offspring.extend([child1, child2])
                
        # Population replacement
        sorted_population = sorted(population.inds, key=lambda x: x.fitness, reverse=True)
        elite_individuals = sorted_population[:elite_cnt]
        non_elite_individuals = sorted_population[elite_cnt:]
        
        replacement_pool = non_elite_individuals + offspring
        selected_non_elite = random.sample(replacement_pool, k=population.size - elite_cnt)
        
        population.inds = elite_individuals + selected_non_elite
        
        # Update best solution and track improvement
        current_best = population.inds[0]
        if best is None or current_best.fitness > best.fitness:
            best = current_best
            no_improvement = 0  # Reset counter
        elif best.fitness == current_best.fitness:
            no_improvement += 1  # Increment stagnation counter
            
        # LLM Transformation Phase
        # Trigger when stuck and still have transformations available
        if (no_improvement > max_no_improvement and 
            transform_cnt < max_time_transform):
            
            transform_cnt += 1
            print(f'No improvement for {no_improvement} generations, '
                  f'applying LLM transformations (attempt {transform_cnt})')
            
            # Select individuals for transformation
            transformation_individuals = []
            if transform_chosen_policy == 'rand':
                # Random selection
                num_to_select = min(max_transform_inds, len(population.inds))
                transformation_individuals = random.sample(population.inds, k=num_to_select)
            elif transform_chosen_policy == 'topk':
                # Select best individuals
                transformation_individuals = topk_selection(
                    population, k=max_transform_inds, select_size=max_transform_inds)
                
            # Apply LLM transformations
            new_individuals = []
            for individual in transformation_individuals:
                # Get transformation suggestions from LLM
                transformations = llm_supporter.get_transformation(
                    individual.chromosome, problem)
                
                if transformations:  # Only proceed if LLM provided suggestions
                    # Create new individual with transformations applied
                    new_individual = Individual()
                    new_individual.chromosome = apply_transformations(
                        individual.chromosome, transformations, problem)
                    new_individual.cal_fitness(problem)
                    new_individuals.append(new_individual)
                
            # Integrate transformed individuals into population
            if new_individuals:
                population.inds.extend(new_individuals)
                population.inds.sort(key=lambda x: x.fitness, reverse=True)
                population.inds = population.inds[:pop_size]  # Maintain population size
                
                # Update best if transformation found better solution
                if population.inds[0].fitness > best.fitness:
                    best = population.inds[0]
            
            no_improvement = 0  # Reset stagnation counter after transformation
            
        # Progress reporting
        sol = best.chromosome
        violations = problem.cal_violations(sol)
        revenue = problem.cal_revenue(sol)
        assigned_count = problem.cal_assigned_cnt(sol)
        budget_penalty = problem.cal_budget_penalty(sol)
        
        print(f'Gen {gen+1}, best fitness = {best.fitness:.2f}, '
              f'violations = {violations}, revenue = {revenue:.2f}, '
              f'budget penalty = {budget_penalty} '
              f'assigned slots = {assigned_count}')
        
    return best