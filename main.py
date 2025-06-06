import os
from dotenv import load_dotenv
import google.generativeai as genai

import random

from problem import read_file, Problem
from llm_support import LLMSupporter, PromptBuilder
from exact_alg import branch_and_bound
from evo import ga, llm_ga
from heuristic import hill_climbing

# Load enviroment variables
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

prompt_builder = PromptBuilder()
llm_supporter = LLMSupporter(model)

random.seed(42)

problem = read_file('test/test_10_many_ads.txt')

from co_evo import co_evo_llm
best, stats = co_evo_llm(500, 100, 16, problem,llm_supporter,prompt_builder,pc=0.8,
                         pm=0.2,elite_ratio=0.1,heuristic_evo_cycle=50,apply_heuristic_cycle=50,
                         appliable_heuristics=4)
print(best)
print(stats)
