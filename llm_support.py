"""
LLM Support Module - Large Language Model Integration

This module provides integration with Large Language Models (LLMs) to enhance
the genetic algorithm with intelligent transformation suggestions.

Key Components:
- PromptBuilder: Creates structured prompts for the LLM
- LLMSupporter: Manages LLM interactions and response parsing
- Transformation functions: Apply LLM-suggested changes to solutions

The LLM acts as a domain expert that can analyze solutions and suggest
improvements when the genetic algorithm gets stuck in local optima.
"""

import json
import time
from typing import List
from problem import Problem
import re
from google.generativeai import GenerativeModel
import os
from dotenv import load_dotenv
load_dotenv()

# Template for prompting the LLM about solution improvements
SOL_PRO_TEMPLATE = """
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

Note: 
- You should consider each ad will not exceed its maximum budget, but if maximum budget is high, then exceeding is allowed to increase revenue.

-----
PROBLEM CONTEXT:
{problem_context}

-----
SPECIFIC SOLUTION (i-th element have value j meaning slots i assign ads j)
{solution}

Please suggest **one single simple transformation** or **a sequence of transformation (at most 10)** that could improve this solution under the two objectives (maximize revenue & maximize number of assigned slots) **and ensure no constraints are violated**. 
Avaiable transformation you can use:
1. {{"type": "unassigned", "indexes": [1, 3, 5]}}  (Unassigned some slots)
2. {{"type": "swap-assign", "indexes": [1, 2]}} (Swap assign of 2 slots)
3. {{"type": "swap-billboard", "indexes": [1, 3]}} (Swap all ads in 2 billboard. To know what ads in what billboard use given solution and given slot-billboard mapping).
4. {{"type": "assign-new", "index": 5, "value": 13}} (Assign new value of ads which from 0 to (num_of_ads-1) and have not appear on given solution)

Your response must be STRICT IN FOLLOWING JSON FORMAT:
{{
    "explain": "Explain the issue with old solution and why you choose new type of transformation. From 2-4 sentences",
    "transformation": [
        {{
        "type": "<your_type, choose from above>",
        .... (specify field base on type)
        }},
        {{
        "type": "<your_type, choose from above>",
        .... (specify field base on type)
        }},
        ...
    ]
}}
"""


class PromptBuilder:
    """
    Builds structured prompts for LLM interactions.
    
    This class creates detailed prompts that provide the LLM with:
    - Problem context and constraints
    - Current solution to analyze
    - Available transformation types
    - Expected response format
    """
    
    def __init__(self, template: str):
        """
        Initialize the prompt builder with a template.
        
        Args:
            template (str): Prompt template with placeholders for dynamic content
        """
        self.template = template
        
    def _get_problem_context(self, problem: Problem) -> str:
        """
        Generate a detailed problem context string for the LLM.
        
        This provides the LLM with all necessary information about:
        - Problem dimensions (billboards, slots, ads)
        - Slot-to-billboard mappings
        - Ad base prices
        - Conflict relationships
        
        Args:
            problem (Problem): Problem instance to describe
            
        Returns:
            str: Formatted problem context string
        """
        context = "This problem has the following data:\n"
        
        # Basic problem dimensions
        context += f"- Number of billboards: {problem.num_billboards} (indexed 0 to {problem.num_billboards-1})\n"
        context += f'- Number of slots: {problem.num_slots} (indexed 0 to {problem.num_slots-1})\n'
        context += f'- Number of ads: {problem.num_ads} (indexed 0 to {problem.num_ads-1})\n'
        
        # Slot-to-billboard mapping
        slot_billboard_mapping = [f"{slot_idx}:{billboard_id}" 
                                 for slot_idx, billboard_id in enumerate(problem.slots)]
        context += f'- Slot-to-billboard mapping (slot:billboard): {", ".join(slot_billboard_mapping)}\n'
        
        # Ad pricing information
        context += f'- Base prices of ads: {problem.ads_base_price}\n'
        
        # Ad maximum budgets
        context += f'- Maximum budgets of ads: {problem.ads_max_budget}\n'
        
        # Conflict relationships
        context += f'- Conflicting ads:\n'
        for ad_id in range(problem.num_ads):
            if problem.conflict_ads[ad_id]:  # Only show ads that have conflicts
                conflicting_ads = list(problem.conflict_ads[ad_id].keys())
                context += f'\tAd {ad_id} conflicts with: {", ".join(map(str, conflicting_ads))}\n'
            else:
                context += f'\tAd {ad_id} has no conflicts\n'
                
        return context
        
    def build(self, sol: List[int], problem: Problem) -> str:
        """
        Build a complete prompt for the LLM.
        
        Args:
            sol (List[int]): Current solution to analyze
            problem (Problem): Problem instance for context
            
        Returns:
            str: Complete formatted prompt ready for LLM
        """
        return self.template.format(
            solution=sol,
            problem_context=self._get_problem_context(problem)
        )


class LLMSupporter:
    """
    Manages interactions with Large Language Models for solution improvement.
    
    This class handles:
    - Sending prompts to LLM
    - Parsing JSON responses
    - Managing rate limits and error handling
    - Tracking transformation history
    """
    
    def __init__(self, model: GenerativeModel, prompt_builder: PromptBuilder):
        """
        Initialize the LLM supporter.
        
        Args:
            model (GenerativeModel): Google Generative AI model instance
            prompt_builder (PromptBuilder): Prompt builder for creating requests
        """
        self.model = model
        self.history_explains: List[str] = []  # Track LLM explanations for debugging
        self.prompt_builder = prompt_builder
        
    def get_json_response(self, prompt: str) -> dict:
        """
        Send a prompt to the LLM and parse the JSON response.
        
        This method handles:
        - API rate limiting with sleep delays
        - JSON extraction from potentially noisy responses
        - Error handling for malformed responses
        
        Args:
            prompt (str): The prompt to send to the LLM
            sleep_time (int): Seconds to sleep after API call (rate limiting)
            
        Returns:
            dict: Parsed JSON response, or None if parsing failed
        """
        try:
            # Send prompt to LLM
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response (may contain extra text)
            # Look for JSON-like content between potential markdown or other formatting
            json_match = re.search(r'(?:json\s*)?(\{[^`]+\})', response.text, re.DOTALL)
            if json_match is None:
                print('Cannot find JSON object in LLM response')
                print(f'Response was: {response.text}')
                return None
                
            # Parse the extracted JSON
            json_text = json_match.group(1)
            json_response: dict = json.loads(json_text)
            
            # Rate limiting - avoid hitting API limits
            time.sleep(int(os.getenv('SLEEP_TIME', 5)))
            return json_response
            
        except json.JSONDecodeError as e:
            print(f'JSON parsing error: {e}')
            print(f'Response text: {response.text}')
            return None
        except Exception as e:
            print(f'LLM API error: {e}')
            return None
        
    def get_transformation(self, sol: list[int], problem: Problem) -> list[dict]:
        """
        Get transformation suggestions from the LLM for a given solution.
        
        This is the main interface for getting LLM help. It:
        1. Builds a detailed prompt about the current solution
        2. Sends it to the LLM
        3. Parses the response for transformation suggestions
        4. Returns structured transformation data
        
        Args:
            sol (list[int]): Current solution to improve
            problem (Problem): Problem context
            
        Returns:
            list[dict]: List of transformation dictionaries, or None if failed
        """
        # Build detailed prompt
        prompt = self.prompt_builder.build(sol, problem)
        
        # Get LLM response
        json_response = self.get_json_response(prompt)
        if json_response is None:
            print('Failed to get valid response from LLM')
            return None
        
        # Extract and store explanation for debugging
        explanation = json_response.get('explain', None)
        if explanation is not None:
            self.history_explains.append(explanation)
            print(f'LLM explanation: {explanation}')
        
        # Extract transformation suggestions
        if 'transformation' not in json_response:
            print('LLM response missing transformation field')
            return None
            
        transformations = json_response['transformation']
        print(f'LLM suggested {len(transformations)} transformations')
        return transformations


def apply_transformation(sol: list[int], transformation: dict, problem: Problem) -> list[int]:
    """
    Apply a single transformation to a solution.
    
    Supported transformation types:
    1. "unassigned": Remove ads from specified slots
    2. "swap-assign": Swap the assignments of two slots  
    3. "swap-billboard": Swap all ads between two billboards
    4. "assign-new": Assign a specific ad to a specific slot
    
    Args:
        sol (list[int]): Solution to transform (modified in place)
        transformation (dict): Transformation specification
        problem (Problem): Problem context for validation
        
    Returns:
        list[int]: The modified solution
    """
    if transformation is None:
        print('Warning: Transformation is None')
        return sol
        
    if 'type' not in transformation:
        print('Warning: Transformation missing type field')
        return sol
        
    transform_type = transformation['type']
    
    # Validate transformation type
    valid_types = ['unassigned', 'swap-assign', 'swap-billboard', 'assign-new']
    if transform_type not in valid_types:
        print(f"Warning: Unsupported transformation type '{transform_type}'")
        return sol
    
    try:
        if transform_type == 'unassigned':
            # Remove ads from specified slots
            slot_indices = transformation['indexes']
            for slot_idx in slot_indices:
                # Validate slot index
                if 0 <= slot_idx < problem.num_slots:
                    sol[slot_idx] = -1  # -1 means unassigned
                else:
                    print(f"Warning: Slot index {slot_idx} out of range")
                    
        elif transform_type == 'swap-assign':
            # Swap assignments between two slots
            slot1_idx, slot2_idx = transformation['indexes']
            # Validate indices
            if (0 <= slot1_idx < problem.num_slots and 
                0 <= slot2_idx < problem.num_slots):
                sol[slot1_idx], sol[slot2_idx] = sol[slot2_idx], sol[slot1_idx]
            else:
                print(f"Warning: Invalid slot indices for swap: {slot1_idx}, {slot2_idx}")
                
        elif transform_type == 'swap-billboard':
            # Swap all ads between two billboards
            billboard1_id, billboard2_id = transformation['indexes']
            
            # Find all slots belonging to each billboard
            billboard1_slots = [i for i, b in enumerate(problem.slots) if b == billboard1_id]
            billboard2_slots = [i for i, b in enumerate(problem.slots) if b == billboard2_id]
            
            # Swap corresponding slots (up to minimum length)
            min_slots = min(len(billboard1_slots), len(billboard2_slots))
            for i in range(min_slots):
                slot1 = billboard1_slots[i]
                slot2 = billboard2_slots[i]
                sol[slot1], sol[slot2] = sol[slot2], sol[slot1]
                
        elif transform_type == 'assign-new':
            # Assign specific ad to specific slot
            slot_idx = transformation['index']
            ad_id = transformation['value']
            
            # Validate indices
            if (0 <= slot_idx < problem.num_slots and 
                0 <= ad_id < problem.num_ads):
                sol[slot_idx] = ad_id
            else:
                print(f"Warning: Invalid assignment - slot {slot_idx}, ad {ad_id}")
                
    except KeyError as e:
        print(f"Warning: Transformation missing required field: {e}")
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid transformation data: {e}")
        
    return sol


def apply_transformations(sol: list[int], transformations: List[dict], problem: Problem) -> list[int]:
    """
    Apply a sequence of transformations to a solution.
    
    Transformations are applied in order, with each transformation
    modifying the result of the previous one.
    
    Args:
        sol (list[int]): Original solution to transform
        transformations (List[dict]): List of transformation specifications
        problem (Problem): Problem context for validation
        
    Returns:
        list[int]: Solution after applying all transformations
    """
    if transformations is None or len(transformations) == 0:
        print('No transformations to apply')
        return sol
    
    # Make a copy to avoid modifying the original
    result_sol = sol.copy()
    
    print(f'Applying {len(transformations)} transformations...')
    
    # Apply each transformation in sequence
    for i, transformation in enumerate(transformations):
        print(f'Applying transformation {i+1}: {transformation.get("type", "unknown")}')
        result_sol = apply_transformation(result_sol, transformation, problem)
    
    return result_sol