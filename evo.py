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
from llm_support import LLMSupporter, PromptBuilder
import prompt_template as pt


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

def _mutation_unassign(p: Individual, problem: Problem) -> Individual:
    """
    Unassign a random slot in the individual.
    
    This mutation sets one random slot to -1 (unassigned).
    
    Args:
        p (Individual): Parent individual to mutate
        problem (Problem): Problem instance for context
        
    Returns:
        Individual: Mutated offspring individual
    """
    # Create copy of parent
    c = Individual()
    c.chromosome = p.chromosome.copy()
    
    # Select random slot to unassign
    slot_idx = random.choice(range(len(c.chromosome)))
    c.chromosome[slot_idx] = -1  # Unassign the slot
    
    return c

def _mutation_assign_new(p: Individual, problem: Problem) -> Individual:
    """
    Assign a random ad to a random slot in the individual.
    
    This mutation selects a random slot and assigns it a new ad (not currently assigned).
    
    Args:
        p (Individual): Parent individual to mutate
        problem (Problem): Problem instance for context
        
    Returns:
        Individual: Mutated offspring individual
    """
    # Create copy of parent
    c = Individual()
    c.chromosome = p.chromosome.copy()
    
    # Select random slot to assign
    slot_idx = random.choice(range(len(c.chromosome)))
    
    # Get all possible ads except the currently assigned one
    current_ad = c.chromosome[slot_idx]
    available_ads = [x for x in range(problem.num_ads) if x != current_ad]
    
    if available_ads:  # Make sure we have options
        new_ad = random.choice(available_ads)
        c.chromosome[slot_idx] = new_ad
    
    return c

def _mutation_swap_assign(p: Individual, problem: Problem) -> Individual:
    """
    Swap the assignment of two random slots in the individual.
    
    This mutation selects two random slots and swaps their assigned ads.
    
    Args:
        p (Individual): Parent individual to mutate
        problem (Problem): Problem instance for context
        
    Returns:
        Individual: Mutated offspring individual
    """
    # Create copy of parent
    c = Individual()
    c.chromosome = p.chromosome.copy()
    
    # Select two different random slots to swap
    slot_idx1, slot_idx2 = random.sample(range(len(c.chromosome)), 2)
    
    # Swap the assignments
    c.chromosome[slot_idx1], c.chromosome[slot_idx2] = (
        c.chromosome[slot_idx2], c.chromosome[slot_idx1])
    
    return c

def _muation_swap_billboard(p: Individual, problem: Problem) -> Individual:
    """
    Swap the assignment of all ads in two random billboards.
    This mutation selects two random billboards and swaps their assigned ads.
    Args:
        p (Individual): Parent individual to mutate
        problem (Problem): Problem instance for context
    Returns:
        Individual: Mutated offspring individual
    """
    # Create copy of parent
    c = Individual()
    c.chromosome = p.chromosome.copy()

    billboard1_id, billboard2_id = random.sample(range(problem.num_billboards), 2)
    billboard1_slots = [i for i, b in enumerate(problem.slots) if b == billboard1_id]
    billboard2_slots = [i for i, b in enumerate(problem.slots) if b == billboard2_id]
    
    min_slots = min(len(billboard1_slots), len(billboard2_slots))
    # Swap assignments for the minimum number of slots
    for i in range(min_slots):
        slot1 = billboard1_slots[i]
        slot2 = billboard2_slots[i]
        c.chromosome[slot1], c.chromosome[slot2] = c.chromosome[slot2], c.chromosome[slot1]
        
    return c

def mutation(p: Individual, problem: Problem) -> Individual:
    """
    Apply mutation to an individual with a random mutation strategy.
    
    The mutation process randomly selects one of the following strategies:
    1. Unassign a random slot
    2. Assign a new ad to a random slot
    3. Swap assignments of two random slots
    4. Swap all ads in two random billboards
    
    Args:
        p (Individual): Parent individual to mutate
        problem (Problem): Problem instance for context
        
    Returns:
        Individual: Mutated offspring individual
    """
    mutation_strategies = [
        _mutation_unassign,
        _mutation_assign_new,
        _mutation_swap_assign,
        _muation_swap_billboard
    ]
    
    # Randomly select a mutation strategy
    mutation_func = random.choice(mutation_strategies)
    
    # Apply the selected mutation
    return mutation_func(p, problem)


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

def llm_crossover(p1: Individual, p2: Individual,
                  prompt_builder: PromptBuilder,
                  llm_supporter: LLMSupporter,
                  problem: Problem) -> tuple[Individual, Individual]:
    """
    Perform LLM-based crossover between two parent individuals.
    
    Args:
        p1 (Individual): First parent individual
        p2 (Individual): Second parent individual
        prompt_builder (PromptBuilder): LLM prompt builder for generating crossover prompts
        llm_supporter (LLMSupporter): LLM interface for executing the crossover
        problem (Problem): Problem instance to solve
    Returns:
        tuple[Individual, Individual]: Two offspring individuals created by LLM
    """
    
    crossover_prompt = pt.CROSSOVER_TEMPLATE
    problem_context = prompt_builder.get_problem_context(problem)
    prompt = prompt_builder.build(solution_1=p1.chromosome,
                                  solution_2=p2.chromosome,
                                  problem_context=problem_context,
                                  template=crossover_prompt)
    json_response = llm_supporter.get_json_response(prompt)
    if json_response is None:
        return None, None
    if 'children' not in json_response or len(json_response['children']) != 2:
        print("LLM crossover response is invalid or does not contain two children.")
        return None, None
    
    if 'solution' not in json_response['children'][0] or \
       'solution' not in json_response['children'][1]:
        print("LLM crossover response does not contain solutions.")
        return None, None
    
    if not problem.check_sol(json_response['children'][0]['solution']) or \
       not problem.check_sol(json_response['children'][1]['solution']):
        print("LLM crossover produced invalid solutions.")
        return None, None
    
    c1 = Individual()
    c2 = Individual()
    
    c1.chromosome = json_response['children'][0]['solution']
    c2.chromosome = json_response['children'][1]['solution']
    
    return c1, c2

def llm_mutation(individual: Individual,
                 prompt_builder: PromptBuilder,
                 llm_supporter: LLMSupporter,
                 problem: Problem) -> Individual:
    """
    Perform LLM-based mutation on an individual.

    Args:
        individual (Individual): Cá thể cần đột biến
        prompt_builder (PromptBuilder): LLM prompt builder for generating crossover prompts
        llm_supporter (LLMSupporter): LLM interface for executing the crossover
        problem (Problem): Problem instance to solve

    Returns:
        Individual: Cá thể mới sau khi đột biến
    """
    mutation_prompt = pt.MUTATION_TEMPLATE
    problem_context = prompt_builder.get_problem_context(problem)
    prompt = prompt_builder.build(solution=individual.chromosome,
                                  problem_context=problem_context,
                                  template=mutation_prompt)
    
    json_response = llm_supporter.get_json_response(prompt)
    if json_response is None:
        return None
    if 'solution' not in json_response:
        print("LLM mutation response is invalid or does not contain a solution.")
        return None
    if problem.check_sol(json_response['solution']) is False:
        print("LLM mutation produced an invalid solution.")
        return None
    new_individual = Individual()
    new_individual.chromosome = json_response['solution']
    
    return new_individual
    

def llm_ga(num_gen: int, pop_size: int, 
           problem: Problem, llm_supporter: LLMSupporter, prompt_builder: PromptBuilder, 
           pc: float = 0.8, pm: float = 0.1, elite_ratio: float = 0.1,
           max_no_improve: int = 40, llm_pop_size: int = 20) -> Individual:
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
        prompt_builder (PromptBuilder): LLM prompt builder for generating prompts
        pc (float): Crossover probability
        pm (float): Mutation probability  
        elite_ratio (float): Elite preservation ratio
        max_no_improve (int): Maximum generations without improvement before LLM intervention
        llm_pop_size (int): Population size for LLM-enhanced solutions
        
    Returns:
        Individual: Best solution found
    """
    
    # Initialize population
    population = Population(pop_size)
    population.random_generate(problem)
    elite_cnt = int(max(1, pop_size * elite_ratio))
    best = None
    
    no_improve_count = 0  # Counter for stagnation detection
    use_llm_flag = False  # Flag to indicate if LLM should be used
    
    # Evolution loop
    for gen in range(num_gen):
        offspring = []
        
        # Standard GA operations (same as regular GA)
        if not use_llm_flag:
            while len(offspring) < pop_size:
                if random.random() < pc:
                    parent1, parent2 = topk_selection(population, k=6, select_size=2)
                    # Standard crossover
                    child1, child2 = crossover(parent1, parent2)
                    
                    # Apply mutation with probability pm
                    if random.random() < pm:
                        child1 = mutation(child1, problem)
                    if random.random() < pm:
                        child2 = mutation(child2, problem)
                        
                    
                    child1.cal_fitness(problem)
                    child2.cal_fitness(problem)
                    
                    offspring.extend([child1, child2])
                    
        else:
            print("Using LLM to create offspring...")
            call_cnt = 0
            while len(offspring) < pop_size and call_cnt < 50:  # Limit LLM calls to avoid excessive costs
                if random.random() < pc:
                    call_cnt += 1
                    parent1, parent2 = topk_selection(population, k=6, select_size=2)
                    c1, c2 = llm_crossover(parent1, parent2, prompt_builder, llm_supporter, problem)
                    
                    if c1 is None or c2 is None:
                        print("Failed in crossover")
                        continue
                    
                    if random.random() < pm:
                        c1_llm = llm_mutation(c1, prompt_builder, llm_supporter, problem)
                        if c1_llm is not None:
                            c1 = mutation(c1_llm, problem)
                    if random.random() < pm:
                        c2_llm = llm_mutation(c2, prompt_builder, llm_supporter, problem)
                        if c2_llm is not None:
                            c2 = mutation(c2_llm, problem) 
                            
                    c1.cal_fitness(problem)
                    c2.cal_fitness(problem)
                    
                    offspring.extend([c1, c2])
                    
                
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
            no_improve_count = 0
        else:
            no_improve_count += 1
            
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
        
        if use_llm_flag:
            no_improve_count = 0
            use_llm_flag = False
        
        if no_improve_count >= max_no_improve:
            # If no improvement for too long, use LLM to suggest transformations
            print(f'No improvement for {no_improve_count} generations, next gen will use LLM...')
            use_llm_flag = True
        
    return best