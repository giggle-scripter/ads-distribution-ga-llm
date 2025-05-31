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


class PromptBuilder:
    """
    Builds structured prompts for LLM interactions.
    
    This class creates detailed prompts that provide the LLM with:
    - Problem context and constraints
    - Current solution to analyze
    - Available transformation types
    - Expected response format
    """
        
    def get_problem_context(self, problem: Problem) -> str:
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
        context += f"- Number of billboards: {problem.num_billboards} (from 0 to {problem.num_billboards-1})\n"
        context += f'- Number of slots: {problem.num_slots} (from 0 to {problem.num_slots-1})\n'
        context += f'- Number of ads: {problem.num_ads} (from 0 to {problem.num_ads-1})\n'
        
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
        
    def build(self, template: str, **kwargs) -> str:
        """
        Build a complete prompt for the LLM.
        
        Args:
            template (str): Template string with placeholders for formatting
            **kwargs: Keyword arguments to fill in the template
            
        Returns:
            str: Complete formatted prompt ready for LLM
        """
        return template.format(
            **kwargs
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
    
    def __init__(self, model: GenerativeModel):
        """
        Initialize the LLM supporter.
        
        Args:
            model (GenerativeModel): Google Generative AI model instance
            prompt_builder (PromptBuilder): Prompt builder for creating requests
        """
        self.model = model
        self.history_explains: List[str] = []  # Track LLM explanations for debugging
        
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