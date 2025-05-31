
"""
Heuristic Algorithm Module - Hill Climbing

This module implements a hill climbing algorithm for the advertising distribution problem.
The algorithm uses local search to iteratively improve solutions by making small changes
that increase revenue while maintaining feasibility.

Key Features:
- Multiple restart capability to escape local optima
- Smart neighbor generation focusing on profitable moves
- Constraint-aware solution generation
- Revenue-based objective with penalty for violations
"""

import random
import time
from typing import List, Tuple, Optional
from problem import Problem


class HillClimbingAlgorithm:
    """
    Hill climbing algorithm implementation for advertising distribution.
    
    The algorithm starts with an initial solution and iteratively improves it
    by exploring neighboring solutions and moving to better ones.
    """
    
    def __init__(self, problem: Problem):
        """
        Initialize the hill climbing algorithm.
        
        Args:
            problem (Problem): The advertising distribution problem to solve
        """
        self.problem = problem
        self.best_solution = None
        self.best_score = float('-inf')
        self.iterations = 0
        
    def solve(self, max_iterations: int = 10000, num_restarts: int = 10, 
              time_limit: float = 1000.0) -> Tuple[List[int], dict]:
        """
        Solve the problem using hill climbing with multiple restarts.
        
        Args:
            max_iterations (int): Maximum iterations per restart
            num_restarts (int): Number of random restarts
            time_limit (float): Maximum time to spend solving
            
        Returns:
            Tuple[List[int], dict]: Best solution, score, and statistics
        """
        print("Starting hill climbing algorithm...")
        
        start_time = time.time()
        self.best_solution = None
        self.best_score = float('-inf')
        total_iterations = 0
        restarts_completed = 0
        
        for restart in range(num_restarts):
            if time.time() - start_time >= time_limit:
                break
            
            # Generate initial solution
            current_solution = self._generate_initial_solution()
            current_score = self._evaluate_solution(current_solution)
            
            # Hill climbing from this starting point
            iterations_this_restart = 0
            improvements = 0
            
            for iteration in range(max_iterations):
                if time.time() - start_time >= time_limit:
                    break
                    
                # Generate neighbors and find the best one
                best_neighbor = None
                best_neighbor_score = current_score
                
                neighbors = self._generate_neighbors(current_solution)
                
                for neighbor in neighbors:
                    neighbor_score = self._evaluate_solution(neighbor)
                    if neighbor_score > best_neighbor_score:
                        best_neighbor = neighbor
                        best_neighbor_score = neighbor_score
                
                # Move to better neighbor if found
                if best_neighbor is not None:
                    current_solution = best_neighbor
                    current_score = best_neighbor_score
                    improvements += 1
                else:
                    # No improvement found, local optimum reached
                    break
                
                iterations_this_restart += 1
                total_iterations += 1
            
            # Update global best
            if current_score > self.best_score:
                self.best_solution = current_solution.copy()
                self.best_score = current_score
                
            restarts_completed += 1
        
        # Calculate final statistics
        total_time = time.time() - start_time
        revenue = self.problem.cal_revenue(self.best_solution) if self.best_solution else 0
        
        stats = {
            'total_iterations': total_iterations,
            'restarts_completed': restarts_completed,
            'time_taken': total_time,
            'violations': self.problem.cal_violations(self.best_solution) if self.best_solution else 0,
            'assigned_slots': self.problem.cal_assigned_cnt(self.best_solution) if self.best_solution else 0,
            'revenue': revenue
        }
        
        return self.best_solution, stats
    
    def _generate_initial_solution(self) -> List[int]:
        """
        Generate an initial feasible solution using greedy construction.
        
        Returns:
            List[int]: Initial solution
        """
        solution = [-1] * self.problem.num_slots
        used_ads = set()
        
        # Calculate revenue per ad for each slot
        slot_revenues = []
        for slot_id in range(self.problem.num_slots):
            billboard_id = self.problem.slots[slot_id]
            slot_factor = self.problem.get_slot_factor_of(billboard_id)
            
            for ad_id in range(self.problem.num_ads):
                if ad_id not in used_ads:
                    base_price = self.problem.ads_base_price[ad_id]
                    max_budget = self.problem.ads_max_budget[ad_id]
                    revenue = min(slot_factor * base_price, max_budget)
                    slot_revenues.append((revenue, slot_id, ad_id))
        
        # Sort by revenue (descending) and assign greedily
        slot_revenues.sort(reverse=True)
        
        for revenue, slot_id, ad_id in slot_revenues:
            if ad_id in used_ads:
                continue
            
            # Check if assignment is feasible
            if self._is_feasible_assignment(solution, slot_id, ad_id):
                solution[slot_id] = ad_id
                used_ads.add(ad_id)
        
        return solution
    
    def _generate_neighbors(self, solution: List[int]) -> List[List[int]]:
        """
        Generate neighboring solutions by making small modifications.
        
        Args:
            solution (List[int]): Current solution
            
        Returns:
            List[List[int]]: List of neighboring solutions
        """
        neighbors = []
        
        # Type 1: Swap two assignments
        assigned_slots = [i for i in range(len(solution)) if solution[i] != -1]
        for i in range(len(assigned_slots)):
            for j in range(i + 1, len(assigned_slots)):
                slot1, slot2 = assigned_slots[i], assigned_slots[j]
                neighbor = solution.copy()
                neighbor[slot1], neighbor[slot2] = neighbor[slot2], neighbor[slot1]
                
                # Only add if feasible
                if self._is_solution_feasible(neighbor):
                    neighbors.append(neighbor)
        
        # Type 2: Move assignment to empty slot
        assigned_slots = [i for i in range(len(solution)) if solution[i] != -1]
        empty_slots = [i for i in range(len(solution)) if solution[i] == -1]
        
        for assigned_slot in assigned_slots:
            for empty_slot in empty_slots:
                neighbor = solution.copy()
                neighbor[empty_slot] = neighbor[assigned_slot]
                neighbor[assigned_slot] = -1
                
                if self._is_solution_feasible(neighbor):
                    neighbors.append(neighbor)
        
        # Type 3: Replace assignment with unused ad
        used_ads = {ad for ad in solution if ad != -1}
        unused_ads = [ad for ad in range(self.problem.num_ads) if ad not in used_ads]
        
        for slot_id in range(len(solution)):
            if solution[slot_id] != -1:  # Only replace existing assignments
                for unused_ad in unused_ads:
                    neighbor = solution.copy()
                    neighbor[slot_id] = unused_ad
                    
                    if self._is_feasible_assignment(neighbor, slot_id, unused_ad):
                        neighbors.append(neighbor)
        
        # Type 4: Add new assignment to empty slot
        for empty_slot in empty_slots:
            for unused_ad in unused_ads:
                neighbor = solution.copy()
                neighbor[empty_slot] = unused_ad
                
                if self._is_feasible_assignment(neighbor, empty_slot, unused_ad):
                    neighbors.append(neighbor)
        
        # Type 5: Remove assignment (set to -1)
        for assigned_slot in assigned_slots:
            neighbor = solution.copy()
            neighbor[assigned_slot] = -1
            neighbors.append(neighbor)
        
        return neighbors
    
    def _is_feasible_assignment(self, solution: List[int], slot_id: int, ad_id: int) -> bool:
        """
        Check if assigning an ad to a specific slot is feasible.
        
        Args:
            solution (List[int]): Current solution
            slot_id (int): Slot to assign to
            ad_id (int): Ad to assign
            
        Returns:
            bool: True if assignment is feasible
        """
        billboard_id = self.problem.slots[slot_id]
        
        # Check conflicts with other ads on same billboard
        for other_slot in range(len(solution)):
            if (other_slot != slot_id and 
                solution[other_slot] != -1 and
                self.problem.slots[other_slot] == billboard_id):
                
                other_ad = solution[other_slot]
                if (ad_id in self.problem.conflict_ads and
                    self.problem.conflict_ads[ad_id].get(other_ad, False)):
                    return False
        
        # Check budget constraint
        slot_factor = self.problem.get_slot_factor_of(billboard_id)
        real_price = slot_factor * self.problem.ads_base_price[ad_id]
        if real_price > self.problem.ads_max_budget[ad_id]:
            return False
        
        # Check for duplicate assignments
        for other_slot in range(len(solution)):
            if other_slot != slot_id and solution[other_slot] == ad_id:
                return False
        
        return True
    
    def _is_solution_feasible(self, solution: List[int]) -> bool:
        """
        Check if a complete solution is feasible.
        
        Args:
            solution (List[int]): Solution to check
            
        Returns:
            bool: True if solution is feasible
        """
        violations = self.problem.cal_violations(solution)
        budget_penalty = self.problem.cal_budget_penalty(solution)
        return violations == 0
    
    def _evaluate_solution(self, solution: List[int]) -> float:
        """
        Evaluate a solution and return its score.
        
        The score combines revenue with penalties for violations.
        
        Args:
            solution (List[int]): Solution to evaluate
            
        Returns:
            float: Solution score (higher is better)
        """
        revenue = self.problem.cal_revenue(solution)
        violations = self.problem.cal_violations(solution)
        budget_penalty = self.problem.cal_budget_penalty(solution)
        
        # Heavy penalty for violations and budget violations
        penalty = violations * 10000
        
        return revenue - penalty


def hill_climbing(problem: Problem, max_iterations: int = 10000, 
                   num_restarts: int = 10, time_limit: float = 60.0) -> Tuple[List[int], dict]:
    """
    Solve the advertising distribution problem using hill climbing.
    
    Args:
        problem (Problem): Problem instance to solve
        max_iterations (int): Maximum iterations per restart
        num_restarts (int): Number of random restarts
        time_limit (float): Maximum time to spend solving
        
    Returns:
        Tuple[List[int], float, dict]: Best solution, revenue, and statistics
    """
    algorithm = HillClimbingAlgorithm(problem)
    return algorithm.solve(max_iterations, num_restarts, time_limit)