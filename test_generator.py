from problem import Problem, random_generate
import random
random.seed(42)

problem_args = [
    (2, 10, 0.1),
    (2, 15, 0.2),
    (3, 15, 0.2),
    (3, 20, 0.25),
    (5, 20, 0.3),
    (10, 30, 0.4),
    (15, 40, 0.4),
    (20, 60, 0.4),
    (30, 100, 0.45)
]

def generate_problem(output_file: str, *args):
    problem = random_generate(*args)
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
            
for i, args in enumerate(problem_args):
    output_file = f"test/test_{i+1}.txt"
    generate_problem(output_file, *args)
    print(f"Generated problem file: {output_file} with args: {args}")