from problem import Problem
import random
from typing import Literal
random.seed(42)

def generate_no_conflict(num_billboards: int, num_ads: int,
                         base_price_dist: Literal['uniform', 'robust']='uniform',
                         max_budget_type: Literal['high', 'approx', 'low'] = 'high') -> Problem:
    billboards = []
    slots = []
    
    # Generate random billboard configurations
    for i in range(num_billboards):
        # Each billboard has 1-6 sides randomly
        num_sides = random.randint(1, 6)
        billboards.append(num_sides)
        
        # Add slots for this billboard
        slots.extend([i for _ in range(num_sides)])
        
    # Generate random ad base prices
    ads_base_price = []
    ads_max_budget = []

    if max_budget_type == 'high':
        max_budget_range = [1.3, 2.5]
    elif max_budget_type == 'approx':
        max_budget_range = [1.05, 2.05]
    else:
        max_budget_range = [1.05, 1.55]
    
    if base_price_dist == 'uniform':
        for _ in range(num_ads):
            price = random.randint(200, 300)
            ads_base_price.append(price)
            ads_max_budget.append(int(price * random.uniform(*max_budget_range)))
    else:
        p = 0.2
        for _ in range(num_ads):
            if random.random() < p:
                price = 500
            else:
                price = random.randint(200, 300)
            ads_base_price.append(price)
            ads_max_budget.append(int(price * random.uniform(*max_budget_range)))
    
    
    # Create and populate problem instance
    problem = Problem(num_billboards, len(slots), num_ads)
    problem.billboards = billboards
    problem.slots = slots
    problem.ads_base_price = ads_base_price
    problem.ads_max_budget = ads_max_budget
    
    return problem


def generate_randomly_conflict(num_billboards: int, num_ads: int, conflict_rate: float = 0.3,
                               base_price_dist: Literal['uniform', 'robust']='uniform',
                               max_budget_type: Literal['high', 'approx', 'low'] = 'high') -> Problem:
    problem = generate_no_conflict(num_billboards, num_ads,
                                   base_price_dist, max_budget_type)
    # Generate random conflict relationships
    conflict_ads = [{} for _ in range(num_ads)]
    for i in range(num_ads - 1):
        for j in range(i + 1, num_ads):  # Avoid self-conflicts and duplicates
            # 40% chance of conflict between any two ads
            if random.random() < conflict_rate:
                conflict_ads[i][j] = True
                conflict_ads[j][i] = True
                
    # Create and populate problem instance
    problem.conflict_ads = conflict_ads
    
    return problem

def generate_clique_conflict(num_billboards: int, num_ads: int, 
                             clique_size: int, max_num_cliques: int):
    problem = generate_no_conflict(num_billboards, num_ads)
    cliques = []
    remain_ads = list(range(num_ads))
    for _ in range(max_num_cliques):
        real_size = random.randint(clique_size // 2, clique_size * 3 // 2)
        chosen = random.sample(remain_ads, real_size)
        chosen.sort()
        cliques.append(chosen)
        remain_ads = [x for x in remain_ads if x not in chosen]
    
    conflict_ads = [{} for _ in range(num_ads)]
    for clique_ids in cliques:
        for i in range(len(clique_ids) - 1):
            for j in range(i + 1, len(clique_ids)):
                conflict_ads[clique_ids[i]][clique_ids[j]] = True
                conflict_ads[clique_ids[j]][clique_ids[i]] = True
    
    problem.conflict_ads = conflict_ads
    return problem

def generate_star_conflict(num_billboards: int, num_ads: int, num_centroids: int, satellites: int):
    problem = generate_no_conflict(num_billboards, num_ads)
    
    base_price_with_ids = list(enumerate(problem.ads_base_price))
    base_price_with_ids.sort(key=lambda x: x[1], reverse=True)
    
    remain_ads = list(range(num_ads))
    conflict_ads = [{} for _ in range(num_ads)]
    for i in range(num_centroids):
        centroid_id = base_price_with_ids[i][0]
        remain_ads.remove(centroid_id)
        sat_ids = random.sample(remain_ads, satellites)
        
        for idx in sat_ids:
            conflict_ads[idx][centroid_id] = True
            conflict_ads[centroid_id][idx] = True
    
    problem.conflict_ads = conflict_ads
    return problem





def write_problem(output_file: str, problem: Problem):
    with open(output_file, "w") as f:
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
            f.write(f"{i} {j}\n")
            
problem_args = [
    ['small_random', generate_randomly_conflict, [2, 10, 0.15]],
    ['medium_random', generate_randomly_conflict, [5, 20, 0.2]],
    ['large_random', generate_randomly_conflict, [15, 50, 0.25]],
    ['small_clique', generate_clique_conflict, [3, 16, 5, 2]],
    ['large_clique', generate_clique_conflict, [15, 50, 10, 4]],
    ['medium_star', generate_star_conflict, [7, 30, 5, 5]],
    ['medium_robust', generate_randomly_conflict, [7, 30, 0.3, 'robust']],
    ['medium_approx_budget', generate_randomly_conflict, [7, 30, 0.3, 'approx']],
    ['medium_low_budget', generate_randomly_conflict, [7, 30, 0.3, 'low']],
    ['many_ads', generate_randomly_conflict, [8, 70, 0.3]]
]

for i, args in enumerate(problem_args):
    test_name = args[0]
    test_func = args[1]
    test_params = args[2]
    output_file = f"test/test_{i+1}_{test_name}.txt"
    problem = test_func(*test_params)
    write_problem(output_file, problem)
    print(f"Generated problem file: {output_file} with args: {args}")