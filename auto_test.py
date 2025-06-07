import os
from dotenv import load_dotenv
import google.generativeai as genai

import random

from problem import read_file, Problem
from llm_support import LLMSupporter, PromptBuilder
from exact_alg import branch_and_bound, pure_backtracking
from evo import ga, llm_ga
from co_evo import co_evo_llm
from heuristic import hill_climbing

# Load enviroment variables
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

prompt_builder = PromptBuilder()
llm_supporter = LLMSupporter(model)

random.seed(42)

def test_pure_backtracking(problem: Problem, problem_name: str=None):
    sol, stats = pure_backtracking(problem, time_limit=2000.0)
    print(f"Pure backtracking Solution for problem {problem_name}:", sol)
    print(stats)

def test_branch_and_bound(problem: Problem, problem_name: str=None):
    sol, stats = branch_and_bound(problem, time_limit=2000.0)
    print(f"Branch and Bound Solution for problem {problem_name}:", sol)
    print(stats)
    
def test_std_ga(problem: Problem, problem_name: str=None):
    best, stats = ga(num_gen=500, pop_size=100, problem=problem,
                     pc=0.8, pm=0.2, elite_ratio=0.1,
                     debug=False)
    print(f"GA best solution for problem {problem_name}", best.chromosome)
    print(stats)
    
def test_llm_ga(problem: Problem, problem_name: str=None):
    best, stats = llm_ga(num_gen=500, pop_size=100, problem=problem,
                         pc=0.8, pm=0.2, elite_ratio=0.1,
                         llm_supporter=llm_supporter,
                         prompt_builder=prompt_builder,
                         max_no_improve=90, max_llm_call=5,
                         debug=False)
    print(f"LLM GA best solution for problem {problem_name}", best.chromosome)
    print(stats)
    
def test_hill_climbing(problem: Problem, problem_name: str=None):
    sol, stats = hill_climbing(problem, max_iterations=40000, 
                               num_restarts=8, time_limit=1000)
    print(f'HillClimbing best solution for problem {problem_name}', sol)
    print(stats)
    
def test_co_evo(problem: Problem, problem_name: str=None):
    best, stats = co_evo_llm(500, 100, 16, problem, llm_supporter,
                             prompt_builder, 
                             pc=0.8, pm=0.2, elite_ratio=0.1,
                             heuristic_evo_cycle=50, apply_heuristic_cycle=50,
                             early_stopping_gen=300, appliable_heuristics=3,
                             problem_code_filepath='safety_problem_code.txt',
                             debug=False)
    
    print(f"Co Evo with LLM best solution for problem {problem_name}", best.chromosome)
    print(stats)
    
NUM_OF_TEST = 10
TEST_NAME_TEMPLATE = "test/test_{id}_{test_type}.txt"

test_type = [
    'small_random',
    'medium_random',
    'large_random', 
    'small_clique', 
    'large_clique',
    'medium_star',
    'medium_robust',
    'medium_approx_budget',
    'medium_low_budget',
    'many_ads'
]

def test_all(test_from: int = 1, test_to: int = NUM_OF_TEST):
    for i in range(test_from, test_to + 1):
        print(f"====Test {i}====")
        test_file = TEST_NAME_TEMPLATE.format(id=i, test_type=test_type[i-1])
        problem = read_file(test_file)
        
        ''' if problem.num_slots <= 25:
            print("Test Pure Backtracking")
            test_pure_backtracking(problem, problem_name=str(i))
            
            print("\nTest Branch and Bound")
            test_branch_and_bound(problem, problem_name=str(i))'''
        
        print("\nTest Standard GA")
        test_std_ga(problem, problem_name=str(i))
        
        print('\nTest Hill Climbing')
        test_hill_climbing(problem, problem_name=str(i))
        
        print("\nTest LLM GA")
        test_llm_ga(problem, problem_name=str(i))
        
        print('\nTest Co-Evo')
        test_co_evo(problem, problem_name=str(i))