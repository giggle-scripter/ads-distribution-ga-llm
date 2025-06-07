"""
Exact Algorithm Module - Backtracking with and without Branch and Bound

This module implements exact algorithms for the advertising distribution problem
including both pure backtracking and backtracking with branch and bound optimization.

Key Features:
- Pure backtracking: Complete exploration without pruning
- Branch and bound backtracking: Optimized exploration with pruning
- Constraint violation checking
- Revenue maximization with budget constraints
"""

import time
from typing import List, Tuple, Optional
from problem import Problem


class PureBacktrackingSupporter:
    """
    Pure backtracking algorithm implementation without branch and bound.
    
    This algorithm explores all possible ad-to-slot assignments systematically
    without any pruning optimizations, guaranteeing to find the optimal solution
    but potentially taking longer than the branch and bound version.
    """
    
    def __init__(self, problem: Problem):
        """
        Initialize the pure backtracking algorithm with a problem instance.
        
        Args:
            problem (Problem): The advertising distribution problem to solve
        """
        self.problem = problem
        self.best_solution = None
        self.best_revenue = -1
        self.nodes_explored = 0
        
    def solve(self, time_limit: float = 300.0) -> Tuple[List[int], dict]:
        """
        Solve the problem using pure backtracking.
        
        Args:
            time_limit (float): Maximum time to spend solving (seconds)
            
        Returns:
            Tuple[List[int], dict]: Best solution found, its revenue, and statistics
        """
        
        start_time = time.time()
        self.best_solution = [-1] * self.problem.num_slots
        self.best_revenue = 0  # Start with revenue of empty solution
        self.nodes_explored = 0
        
        # Initialize current solution and tracking structures
        current_solution = [-1] * self.problem.num_slots
        used_ads = set()
        
        # Start backtracking from slot 0
        self._backtrack(0, current_solution, used_ads, start_time, time_limit)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        stats = {
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': 0,  # No pruning in pure backtracking
            'time_taken': total_time,
            'time_limit_reached': total_time >= time_limit,
            'best_revenue': self.best_revenue
        }
        
        return self.best_solution.copy(), stats
    
    def _backtrack(self, slot_idx: int, current_solution: List[int], used_ads: set, 
                  start_time: float, time_limit: float) -> bool:
        """
        Recursive backtracking function without pruning.
        
        Args:
            slot_idx (int): Current slot index to assign
            current_solution (List[int]): Current partial solution
            used_ads (set): Set of already used ad IDs
            start_time (float): Algorithm start time
            time_limit (float): Maximum time allowed
            
        Returns:
            bool: True if time limit exceeded, False otherwise
        """
        # Check time limit
        if time.time() - start_time >= time_limit:
            return True
            
        self.nodes_explored += 1
        
        # Base case: all slots assigned
        if slot_idx == self.problem.num_slots:
            # Calculate revenue and update best solution if better
            revenue = self._calculate_feasible_revenue(current_solution)
            if revenue > self.best_revenue:
                self.best_revenue = revenue
                self.best_solution = current_solution.copy()
            return False
        
        # Try assigning each available ad to current slot
        billboard_id = self.problem.slots[slot_idx]
        
        # Option 1: Leave slot unassigned
        current_solution[slot_idx] = -1
        if self._backtrack(slot_idx + 1, current_solution, used_ads, start_time, time_limit):
            return True
        
        # Option 2: Try assigning each available ad
        for ad_id in range(self.problem.num_ads):
            if ad_id in used_ads:
                continue
                
            # Check if assignment is feasible
            if self._is_feasible_assignment(slot_idx, ad_id, current_solution, used_ads):
                # Make assignment
                current_solution[slot_idx] = ad_id
                used_ads.add(ad_id)
                
                # Recurse
                if self._backtrack(slot_idx + 1, current_solution, used_ads, start_time, time_limit):
                    return True
                
                # Backtrack
                used_ads.remove(ad_id)
                current_solution[slot_idx] = -1
        
        return False
    
    def _is_feasible_assignment(self, slot_idx: int, ad_id: int, 
                              current_solution: List[int], used_ads: set) -> bool:
        """
        Check if assigning an ad to a slot is feasible.
        
        Args:
            slot_idx (int): Slot index to check
            ad_id (int): Ad ID to assign
            current_solution (List[int]): Current partial solution
            used_ads (set): Set of used ad IDs
            
        Returns:
            bool: True if assignment is feasible
        """
        billboard_id = self.problem.slots[slot_idx]
        
        # Check conflicts with ads already assigned to same billboard
        for i in range(slot_idx):
            if (current_solution[i] != -1 and 
                self.problem.slots[i] == billboard_id and
                ad_id in self.problem.conflict_ads and
                self.problem.conflict_ads[ad_id].get(current_solution[i], False)):
                return False
        
        # Check budget constraint
        slot_factor = self.problem.get_slot_factor_of(billboard_id)
        real_price = slot_factor * self.problem.ads_base_price[ad_id]
        
        # Uncomment the following line if you want to allow exceeding budget
        #if real_price > self.problem.ads_max_budget[ad_id]:
        #    return False
        
        return True
    
    def _calculate_feasible_revenue(self, solution: List[int]) -> float:
        """
        Calculate revenue for a solution, considering only feasible assignments.
        
        Args:
            solution (List[int]): Solution to evaluate
            
        Returns:
            float: Total feasible revenue
        """
        # Check for violations first
        violations = self.problem.cal_violations(solution)
        if violations > 0:
            return -1  # Invalid solution
        
        return self.problem.cal_revenue(solution)


class BranchAndBoundSupporter:
    """
    Exact algorithm implementation using backtracking with branch and bound.
    
    The algorithm systematically explores all possible ad-to-slot assignments
    while maintaining feasibility constraints and pruning suboptimal branches.
    """
    
    def __init__(self, problem: Problem):
        """
        Initialize the exact algorithm with a problem instance.
        
        Args:
            problem (Problem): The advertising distribution problem to solve
        """
        self.problem = problem
        self.best_solution = None
        self.best_revenue = -1
        self.nodes_explored = 0
        self.nodes_pruned = 0
        
    def solve(self, time_limit: float = 300.0) -> Tuple[List[int], dict]:
        """
        Solve the problem using backtracking with branch and bound.
        
        Args:
            time_limit (float): Maximum time to spend solving (seconds)
            
        Returns:
            Tuple[List[int], dict]: Best solution found, its revenue, and statistics
        """
        
        start_time = time.time()
        self.best_solution = [-1] * self.problem.num_slots
        self.best_revenue = 0  # Start with revenue of empty solution
        self.nodes_explored = 0
        self.nodes_pruned = 0
        
        # Initialize current solution and tracking structures
        current_solution = [-1] * self.problem.num_slots
        used_ads = set()
        
        # Start backtracking from slot 0
        self._backtrack(0, current_solution, used_ads, start_time, time_limit)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        stats = {
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'time_taken': total_time,
            'time_limit_reached': total_time >= time_limit,
            'best_revenue': self.best_revenue
        }
        
        return self.best_solution.copy(), stats
    
    def _backtrack(self, slot_idx: int, current_solution: List[int], used_ads: set, 
                  start_time: float, time_limit: float) -> bool:
        """
        Recursive backtracking function with branch and bound pruning.
        
        Args:
            slot_idx (int): Current slot index to assign
            current_solution (List[int]): Current partial solution
            used_ads (set): Set of already used ad IDs
            start_time (float): Algorithm start time
            time_limit (float): Maximum time allowed
            
        Returns:
            bool: True if time limit exceeded, False otherwise
        """
        # Check time limit
        if time.time() - start_time >= time_limit:
            return True
            
        self.nodes_explored += 1
        
        # Base case: all slots assigned
        if slot_idx == self.problem.num_slots:
            # Calculate revenue and update best solution if better
            revenue = self._calculate_feasible_revenue(current_solution)
            if revenue > self.best_revenue:
                self.best_revenue = revenue
                self.best_solution = current_solution.copy()
            return False
        
        # Branch and bound: estimate upper bound for remaining slots
        upper_bound = self._estimate_upper_bound(slot_idx, current_solution, used_ads)
        current_revenue = self._calculate_feasible_revenue(current_solution[:slot_idx])
        
        if current_revenue + upper_bound <= self.best_revenue:
            self.nodes_pruned += 1
            return False
        
        # Try assigning each available ad to current slot
        billboard_id = self.problem.slots[slot_idx]
        
        # Option 1: Leave slot unassigned
        current_solution[slot_idx] = -1
        if self._backtrack(slot_idx + 1, current_solution, used_ads, start_time, time_limit):
            return True
        
        # Option 2: Try assigning each available ad
        for ad_id in range(self.problem.num_ads):
            if ad_id in used_ads:
                continue
                
            # Check if assignment is feasible
            if self._is_feasible_assignment(slot_idx, ad_id, current_solution, used_ads):
                # Make assignment
                current_solution[slot_idx] = ad_id
                used_ads.add(ad_id)
                
                # Recurse
                if self._backtrack(slot_idx + 1, current_solution, used_ads, start_time, time_limit):
                    return True
                
                # Backtrack
                used_ads.remove(ad_id)
                current_solution[slot_idx] = -1
        
        return False
    
    def _is_feasible_assignment(self, slot_idx: int, ad_id: int, 
                              current_solution: List[int], used_ads: set) -> bool:
        """
        Check if assigning an ad to a slot is feasible.
        
        Args:
            slot_idx (int): Slot index to check
            ad_id (int): Ad ID to assign
            current_solution (List[int]): Current partial solution
            used_ads (set): Set of used ad IDs
            
        Returns:
            bool: True if assignment is feasible
        """
        billboard_id = self.problem.slots[slot_idx]
        
        # Check conflicts with ads already assigned to same billboard
        for i in range(slot_idx):
            if (current_solution[i] != -1 and 
                self.problem.slots[i] == billboard_id and
                ad_id in self.problem.conflict_ads and
                self.problem.conflict_ads[ad_id].get(current_solution[i], False)):
                return False
        
        # Check budget constraint
        slot_factor = self.problem.get_slot_factor_of(billboard_id)
        real_price = slot_factor * self.problem.ads_base_price[ad_id]
        
        # Uncomment the following line if you want to allow exceeding budget
        #if real_price > self.problem.ads_max_budget[ad_id]:
        #    return False
        
        return True
    
    def _estimate_upper_bound(self, slot_idx: int, current_solution: List[int], 
                            used_ads: set) -> float:
        """
        Estimate upper bound for revenue from remaining unassigned slots.
        
        This provides an optimistic estimate by assuming the best possible
        assignments for remaining slots without considering all constraints.
        
        Args:
            slot_idx (int): Starting slot index for estimation
            current_solution (List[int]): Current partial solution
            used_ads (set): Set of used ad IDs
            
        Returns:
            float: Estimated upper bound for remaining revenue
        """
        if slot_idx >= self.problem.num_slots:
            return 0
        
        # Get remaining ads and slots
        remaining_ads = [ad for ad in range(self.problem.num_ads) if ad not in used_ads]
        remaining_slots = list(range(slot_idx, self.problem.num_slots))
        
        if not remaining_ads or not remaining_slots:
            return 0
        
        # Calculate potential revenues for remaining ad-slot combinations
        potential_revenues = []
        for slot in remaining_slots:
            billboard_id = self.problem.slots[slot]
            slot_factor = self.problem.get_slot_factor_of(billboard_id)
            
            for ad_id in remaining_ads:
                base_price = self.problem.ads_base_price[ad_id]
                max_budget = self.problem.ads_max_budget[ad_id]
                revenue = min(slot_factor * base_price, max_budget)
                potential_revenues.append(revenue)
        
        # Sort revenues in descending order and take top min(ads, slots)
        potential_revenues.sort(reverse=True)
        max_assignments = min(len(remaining_ads), len(remaining_slots))
        
        return sum(potential_revenues[:max_assignments])
    
    def _calculate_feasible_revenue(self, solution: List[int]) -> float:
        """
        Calculate revenue for a solution, considering only feasible assignments.
        
        Args:
            solution (List[int]): Solution to evaluate
            
        Returns:
            float: Total feasible revenue
        """
        # Check for violations first
        violations = self.problem.cal_violations(solution)
        if violations > 0:
            return -1  # Invalid solution
        
        return self.problem.cal_revenue(solution)


def pure_backtracking(problem: Problem, time_limit: float = 300.0) -> Tuple[List[int], dict]:
    """
    Solve the advertising distribution problem using pure backtracking.
    
    Args:
        problem (Problem): Problem instance to solve
        time_limit (float): Maximum time to spend solving
        
    Returns:
        Tuple[List[int], dict]: Best solution, statistics
    """
    algorithm = PureBacktrackingSupporter(problem)
    return algorithm.solve(time_limit)


def branch_and_bound(problem: Problem, time_limit: float = 300.0) -> Tuple[List[int], dict]:
    """
    Solve the advertising distribution problem using exact algorithm with branch and bound.
    
    Args:
        problem (Problem): Problem instance to solve
        time_limit (float): Maximum time to spend solving
        
    Returns:
        Tuple[List[int], dict]: Best solution, statistics
    """
    algorithm = BranchAndBoundSupporter(problem)
    return algorithm.solve(time_limit)