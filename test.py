"""
Test Module - Entry Point for the Advertising Distribution Optimization

This module demonstrates how to use the LLM-enhanced genetic algorithm
to solve advertising distribution problems. It shows the complete workflow
from problem loading to solution optimization.

Usage:
    python test.py

Make sure to:
1. Install required dependencies (see requirements.txt)
2. Set up your Google API key in a .env file
3. Prepare your problem data file (e.g., sample.txt)
"""

# Import genetic algorithm implementations
from evo import ga, llm_ga

# Import problem handling utilities
from problem import read_file, random_generate

# Import standard libraries
import random
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This should contain your GOOGLE_API_KEY
load_dotenv()

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the LLM model
# Using Gemini 2.0 Flash for fast, cost-effective responses
model = genai.GenerativeModel('gemini-2.0-flash')

# Import LLM support components
from llm_support import PromptBuilder, LLMSupporter

# Set up LLM integration
prompt_builder = PromptBuilder()
llm_supporter = LLMSupporter(model)

ga_config = {
    "num_gen": 500,
    "pop_size": 100,
    "pc": 0.8,
    "pm": 0.1,
    "elite_ratio": 0.1,
}

llm_ga_config = ga_config.copy()
llm_ga_config['max_no_improve'] = 30
llm_ga_config['llm_pop_size'] = 30  # Number of solutions to send to LLM for improvement

# Set random seed for reproducible results
random.seed(42)

# Init common problem data
problem = read_file("sample-max-budget.txt")  # Load problem data from file
# If use random problem data, uncomment the next line
# problem_data = random_generate(30, 150) # 30 billboards, 150 ads

# If use a specific problem data file, uncomment the next line
'''with open("sample-max-budget.txt", "w") as f:
    f.write(f'{problem.num_billboards} {problem.num_slots} {problem.num_ads}\n')
    f.write(" ".join(map(str, problem.slots)) + "\n")
    f.write(" ".join(map(str, problem.ads_base_price)) + "\n")
    f.write(" ".join(map(str, problem.ads_max_budget)) + "\n")
    conflict_ads_pair = set()
    for i in range(problem.num_ads - 1):
        for j in range(i + 1, problem.num_ads):
            if problem.conflict_ads[i].get(j, False):
                conflict_ads_pair.add((i, j))
    f.write(str(len(conflict_ads_pair)) + "\n")
    for i, j in conflict_ads_pair:
        f.write(f"{i} {j}\n")'''
    
def test_std_ga():
    global problem, ga_config
    """
    Optional function to compare LLM-enhanced GA with standard GA.
    
    This can be useful for research or to understand the impact
    of LLM enhancement on solution quality.
    """
    print("\n=== Running Standard GA for Comparison ===")
    best_solution = ga(
        num_gen=ga_config["num_gen"],  # Fewer generations for quick comparison
        pop_size=ga_config["pop_size"],
        problem=problem,
        pc=ga_config["pc"],
        pm=ga_config["pm"],
        elite_ratio=ga_config["elite_ratio"]
    )
    
    print("\n=== Optimization Complete ===")
        
    # Display final results
    print("\nFinal Solution:")
    print(f"Chromosome: {best_solution.chromosome}")
    
    # Calculate and display solution quality metrics
    violations = problem.cal_violations(best_solution.chromosome)
    revenue = problem.cal_revenue(best_solution.chromosome)
    assigned_count = problem.cal_assigned_cnt(best_solution.chromosome)
    
    print(f"\nSolution Quality:")
    print(f"  - Total fitness: {best_solution.fitness:.2f}")
    print(f"  - Constraint violations: {violations}")
    print(f"  - Total revenue: ${revenue:.2f}")
    print(f"  - Assigned slots: {assigned_count}/{problem.num_slots}")
    print(f"  - Utilization: {assigned_count/problem.num_slots*100:.1f}%")
    
    # Interpret the solution
    print(f"\nSolution Interpretation:")
    print("Slot -> Ad assignments:")
    for slot_idx, ad_id in enumerate(best_solution.chromosome):
        billboard_id = problem.slots[slot_idx]
        if ad_id >= 0:
            revenue_multiplier = problem.get_slot_factor_of(billboard_id)
            base_price = problem.ads_base_price[ad_id]
            slot_revenue = revenue_multiplier * base_price
            print(f"  Slot {slot_idx} (Billboard {billboard_id}): "
                    f"Ad {ad_id} -> ${slot_revenue:.2f} revenue")
        else:
            print(f"  Slot {slot_idx} (Billboard {billboard_id}): Unassigned")
            
def test_llm_ga():
    global problem, llm_ga_config, llm_supporter
    """
    Optional function to compare LLM-enhanced GA with standard GA.
    
    This can be useful for research or to understand the impact
    of LLM enhancement on solution quality.
    """
    print("\n=== Running Standard GA for Comparison ===")
    best_solution = llm_ga(
        num_gen=llm_ga_config["num_gen"],  # Fewer generations for quick comparison
        pop_size=llm_ga_config["pop_size"],
        problem=problem,
        pc=llm_ga_config["pc"],
        pm=llm_ga_config["pm"],
        elite_ratio=llm_ga_config["elite_ratio"],
        max_no_improve=llm_ga_config['max_no_improve'],
        llm_pop_size=llm_ga_config["llm_pop_size"],
        llm_supporter=llm_supporter,
        prompt_builder=prompt_builder
    )
    
    print("\n=== Optimization Complete ===")
        
    # Display final results
    print("\nFinal Solution:")
    print(f"Chromosome: {best_solution.chromosome}")
    
    # Calculate and display solution quality metrics
    violations = problem.cal_violations(best_solution.chromosome)
    revenue = problem.cal_revenue(best_solution.chromosome)
    assigned_count = problem.cal_assigned_cnt(best_solution.chromosome)
    
    print(f"\nSolution Quality:")
    print(f"  - Total fitness: {best_solution.fitness:.2f}")
    print(f"  - Constraint violations: {violations}")
    print(f"  - Total revenue: ${revenue:.2f}")
    print(f"  - Assigned slots: {assigned_count}/{problem.num_slots}")
    print(f"  - Utilization: {assigned_count/problem.num_slots*100:.1f}%")
    
    # Interpret the solution
    print(f"\nSolution Interpretation:")
    print("Slot -> Ad assignments:")
    for slot_idx, ad_id in enumerate(best_solution.chromosome):
        billboard_id = problem.slots[slot_idx]
        if ad_id >= 0:
            revenue_multiplier = problem.get_slot_factor_of(billboard_id)
            base_price = problem.ads_base_price[ad_id]
            slot_revenue = revenue_multiplier * base_price
            print(f"  Slot {slot_idx} (Billboard {billboard_id}): "
                    f"Ad {ad_id} -> ${slot_revenue:.2f} revenue")
        else:
            print(f"  Slot {slot_idx} (Billboard {billboard_id}): Unassigned")