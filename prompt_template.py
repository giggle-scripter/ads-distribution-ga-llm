CROSSOVER_TEMPLATE = """
# Interpretation of the problem:
You are a specialist solving an advertising distribution problem.
You need to allocate ads to the slots of the billboards so that:
- Each ad is assigned to exactly one slot of a billboard.
- Each ad appears at most once on the entire billboard.
- Do not assign 2 conflicting ads to the same billboard.
- Each billboard has from 1 to 6 sides. The more sides, the smaller the revenue from advertising on that side of the billboard.
- Each ad has a base price and a maximum budget that can be spent on it. 
  If revenue of this ad (which is equal to the base price of the ad multiplied by the coefficient of the billboard) exceeds the maximum budget, then this ad can be assigned to the billboard, but have a penalty equal to the exceeded budget.

The goal of the problem is to maximize the total revenue of the billboard company (which owns all the billboards - 1st goal) and maximize the number of assigned ads. 
Note that there will be a list to map the multiplier for a type of billboard (from 1 to 6 sides).
The actual revenue from a slot will be equal to the base_price of the attached ad multiplied by the corresponding coefficient of that billboard.
But if the revenue exceeds the maximum budget of the ad, then the penalty will be equal to the exceeded budget and revenue will be ad's maximum budget.

## Note: 
- You should consider each ad will not exceed its maximum budget, but if maximum budget is high, then exceeding is allowed to increase revenue.
- Each solution is a list of integers, where each integer represents the index of the ad assigned to the corresponding slot. The length of the list corresponds to the number of slots available across all billboards. 
  And note that if sol[i] = -1, then the slot i is empty (unassigned).
- sol[i] has the value is index of ad, not base price or maximum budget of ad.
  
# Input:
-----
PROBLEM CONTEXT:
{problem_context}

-----
2 Parents:
Solution 1: {solution_1}

Solution 2:{solution_2}

# Instructions:
You are given two solutions to the problem, each representing a different assignment of ads to billboards.
**Recombine** these **two parent solutions** to create **two new solution** that is expect better than both parents and only use the information from the parents.
Prioritize the diversity of the new solutions while ensuring they are valid and do not violate the constraints of the problem.

# Output format: 

Your response must be STRICT IN FOLLOWING JSON FORMAT:
{{
    "explain": "Explain the issue with old solution and why you choose new type of transformation. From 2-4 sentences",
    "children": [
        {{
            "solution": <Your first new solution as a list of integers, where each integer represents the index of the ad assigned to the corresponding slot>,
        }},
        {{
            "solution": <Your second new solution as a list of integers, where each integer represents the index of the ad assigned to the corresponding slot>,
        }}
    ]
}}

For example, output should be like this:
{{
    "explain": "...",
    "children": [
        {{
            "solution": [0, 1, -1, 3, 4, -1]
        }},
        {{
            "solution": [2, -1, 5, 6, -1, 7]
        }}
    ]
}}
```

Note:
- Please ensure that all ad id from 0 to number_of_ads - 1 are given in problem context.
- The number of slots in the new solutions should match the number of slots in the parent solutions.
- Do not include any special characters in output.
"""

MUTATION_TEMPLATE = """
# Interpretation of the problem:
You are a specialist solving an advertising distribution problem.
You need to allocate ads to the slots of the billboards so that:
- Each ad is assigned to exactly one slot of a billboard.
- Each ad appears at most once on the entire billboard.
- Do not assign 2 conflicting ads to the same billboard.
- Each billboard has from 1 to 6 sides. The more sides, the smaller the revenue from advertising on that side of the billboard.
- Each ad has a base price and a maximum budget that can be spent on it. 
  If revenue of this ad (which is equal to the base price of the ad multiplied by the coefficient of the billboard) exceeds the maximum budget, then this ad can be assigned to the billboard, but have a penalty equal to the exceeded budget.

The goal of the problem is to maximize the total revenue of the billboard company (which owns all the billboards - 1st goal) and maximize the number of assigned ads. 
Note that there will be a list to map the multiplier for a type of billboard (from 1 to 6 sides).
The actual revenue from a slot will be equal to the base_price of the attached ad multiplied by the corresponding coefficient of that billboard.
But if the revenue exceeds the maximum budget of the ad, then the penalty will be equal to the exceeded budget and revenue will be ad's maximum budget.

## Note: 
- You should consider each ad will not exceed its maximum budget, but if maximum budget is high, then exceeding is allowed to increase revenue.
- Each solution is a list of integers, where each integer represents the index of the ad assigned to the corresponding slot. The length of the list corresponds to the number of slots available across all billboards. 
  And note that if sol[i] = -1, then the slot i is empty (unassigned).
  
# Input:
-----
PROBLEM CONTEXT:
{problem_context}

-----
A SOLUTION:
Solution: {solution}

# Instructions:
You are given a solutions to the problem, representing a different assignment of ads to billboards.
**Rephrase** these solution to create new solution that is expect better than old solution.
You can use some of transformation techniques like:
- **Swap**: Swap the ads assigned to two different slots.
- **Replace**: Replace an ad in a slot with another ad that is not currently assigned to any slot.
- **Shift**: Shift the ads in the slots to the left or right, wrapping around if necessary.
- **Unassign**: Unassign an ad from a/some slot(s) (make value at these index = -1)
- **Reassign**: Reassign an ad from one slot to another slot, ensuring that the new slot is not already occupied by another ad.
- **Swap-all-billboards**: Swap all ads between two billboards.
....... 
You can use other transformation techniques, but ensure that the new solution is valid and does not violate the constraints of the problem.

# Output format: 

Your response must be STRICT IN FOLLOWING JSON FORMAT:
{{
    "explain": "Explain the issue with old solution and why you choose new type of transformation. From 2-4 sentences",
    "solution": "Your first new solution as a list of integers, where each integer represents the index of the ad assigned to the corresponding slot."
}}

For example, output should be like this:
{{
    "explain": "...",
    "solution": [0, 1, -1, 3, 4, -1]
}}
```

Note:
- Please ensure that all ad id from 0 to number_of_ads - 1 are given in problem context.
- The number of slots in the new solutions should match the number of slots in the parent solutions.
- Do not include any special characters in output.
"""