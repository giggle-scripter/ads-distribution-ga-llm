"""
Problem Module - Advertising Distribution Problem

This module defines the core problem structure for the advertising distribution optimization.
The problem involves assigning advertisements to billboard slots while maximizing revenue
and respecting various constraints.

Key Concepts:
- Billboards: Physical advertising structures with 1-6 sides (slots)
- Slots: Individual advertising spaces on billboards
- Ads: Advertisements with base prices that need to be assigned
- Conflicts: Some ads cannot be placed on the same billboard
"""

import random


class Problem:
    """
    Represents the advertising distribution optimization problem.
    
    This class encapsulates all the data and constraints for the problem:
    - Billboard configurations (how many sides each has)
    - Slot-to-billboard mappings
    - Advertisement base prices
    - Conflict relationships between ads
    
    The revenue model uses multiplier factors based on billboard sides:
    - 1 side: 2.0x multiplier (premium single-sided billboard)
    - 2 sides: 1.7x multiplier
    - 3 sides: 1.5x multiplier
    - 4 sides: 1.3x multiplier
    - 5 sides: 1.1x multiplier
    - 6 sides: 1.0x multiplier (base price, most common)
    """
    
    # Revenue multiplier factors based on number of billboard sides
    # Fewer sides = higher multiplier (more premium placement)
    SLOTS_FACTOR = {
        1: 2.0,  # Single-sided billboard - highest revenue
        2: 1.7,  # Two-sided billboard
        3: 1.5,  # Three-sided billboard
        4: 1.3,  # Four-sided billboard
        5: 1.1,  # Five-sided billboard
        6: 1.0   # Six-sided billboard - base revenue
    }
    
    def __init__(self, num_billboards: int, num_slots: int, num_ads: int):
        """
        Initialize the problem instance.
        
        Args:
            num_billboards (int): Total number of billboards available
            num_slots (int): Total number of advertising slots across all billboards
            num_ads (int): Total number of advertisements to be assigned
        """
        self.num_billboards = num_billboards  # Total billboards in the system
        self.num_slots = num_slots            # Total slots across all billboards
        self.num_ads = num_ads                # Total ads to be placed
        
        # Initialize data structures
        self.billboards = [0] * num_billboards    # Number of sides for each billboard
        self.slots = [-1] * num_slots             # Which billboard each slot belongs to
        self.ads_base_price = [0] * num_ads       # Base price for each advertisement
        self.ads_max_budget = [0] * num_ads
        self.conflict_ads = [{} for _ in range(num_ads)]  # Conflict relationships
        
    def get_num_slots_of(self, billboard_id: int) -> int:
        """
        Get the number of slots (sides) of a specific billboard.
        
        Args:
            billboard_id (int): The ID of the billboard (0-indexed)
            
        Returns:
            int: Number of slots on the specified billboard
        """
        return self.billboards[billboard_id]
    
    def get_slot_factor_of(self, billboard_id: int) -> float:
        """
        Get the revenue multiplier factor for a specific billboard.
        
        Args:
            billboard_id (int): The ID of the billboard (0-indexed)
            
        Returns:
            float: Revenue multiplier factor for the billboard
        """
        num_sides = self.billboards[billboard_id]
        return Problem.SLOTS_FACTOR[num_sides]
    
    def cal_violations(self, sol: list) -> int:
        """
        Calculate the total number of constraint violations in a solution.
        
        This function checks for two types of violations:
        1. Conflict violations: Conflicting ads placed on the same billboard
        2. Duplicate violations: Same ad assigned to multiple slots
        
        Args:
            sol (list): Solution array where sol[i] = ad_id assigned to slot i
                       (-1 means slot is unassigned)
        
        Returns:
            int: Total number of violations (higher is worse)
        """
        total_violations = 0
        
        # Check all pairs of slots for violations
        for i in range(len(sol) - 1):
            for j in range(i + 1, len(sol)):
                # Skip if either slot is unassigned
                if sol[i] == -1 or sol[j] == -1:
                    continue
                
                # Check for conflict violations (conflicting ads on same billboard)
                if (self.slots[i] == self.slots[j] and  # Same billboard
                    sol[i] in self.conflict_ads and     # First ad has conflicts
                    self.conflict_ads[sol[i]].get(sol[j], False)):  # Conflicts with second ad
                    total_violations += 2  # Heavy penalty for conflict violations
                
                # Check for duplicate violations (same ad in multiple slots)
                if sol[i] == sol[j] and sol[i] >= 0:
                    total_violations += 1  # Penalty for duplicate assignment
                    
        return total_violations
    
    def cal_budget_penalty(self, sol: list[int]) -> float:
        total_penalty = 0.0
        for i in range(len(sol)):
            slot_id = i
            ad_id = sol[i]
            
            billboard_id = self.slots[slot_id]
            real_price = self.get_slot_factor_of(billboard_id) * self.ads_base_price[ad_id]
            
            total_penalty += max(0, real_price - self.ads_max_budget[ad_id])
            
        return total_penalty
    
    def cal_revenue(self, sol: list) -> float:
        """
        Calculate the total revenue generated by a solution.
        
        Revenue = sum of (ad_base_price * billboard_multiplier) for all assigned slots
        
        Args:
            sol (list): Solution array where sol[i] = ad_id assigned to slot i
                       (-1 means slot is unassigned)
        
        Returns:
            float: Total revenue generated by the solution
        """
        total_revenue = 0
        
        for i in range(len(sol)):
            # Skip unassigned slots
            if sol[i] == -1:
                continue
                
            # Get billboard information for this slot
            billboard_id = self.slots[i]
            multiplier_factor = self.get_slot_factor_of(billboard_id)
            
            # Get ad information
            ad_id = sol[i]
            ad_base_price = self.ads_base_price[ad_id]
            
            # Calculate revenue for this slot
            slot_revenue = multiplier_factor * ad_base_price
            slot_revenue = min(self.ads_max_budget[ad_id], slot_revenue)
            total_revenue += slot_revenue
            
        return total_revenue
    
    def cal_assigned_cnt(self, sol: list) -> int:
        """
        Calculate the number of assigned slots in a solution.
        
        Args:
            sol (list): Solution array where sol[i] = ad_id assigned to slot i
                       (-1 means slot is unassigned)
        
        Returns:
            int: Number of slots that have been assigned ads
        """
        return sum(1 for x in sol if x >= 0)
    
    def check_sol(self, sol: list) -> bool:
        if len(sol) != self.num_slots:
            print(f"Solution length {len(sol)} does not match number of slots {self.num_slots}")
            return False
        if any(ad_id < -1 or ad_id >= self.num_ads for ad_id in sol if ad_id != -1):
            print(f"Invalid ad_id in solution: {sol}")
            return False
        return True


def read_console() -> Problem:
    """
    Read problem data from console input.
    
    Input format:
    Line 1: num_billboards num_slots num_ads
    Line 2: slot assignments (which billboard each slot belongs to)
    Line 3: ad base prices
    Line 4: number of conflicts
    Next lines: conflict pairs (ad1_id ad2_id)
    
    Returns:
        Problem: Initialized problem instance with data from console
    """
    # Read basic problem dimensions
    num_billboards, num_slots, num_ads = map(int, input().split())
    
    # Initialize billboard slot counts
    billboards = [0] * num_billboards
    
    # Read slot-to-billboard mapping and count slots per billboard
    slots = list(map(int, input().split()))
    for billboard_id in slots:
        billboards[billboard_id] += 1
        
    # Read advertisement base prices
    ads_base_price = list(map(int, input().split()))
    
    # Read max budget
    ads_max_budgets = list(map(int, input().split()))
    
    # Initialize conflict relationships
    conflict_ads = [{} for _ in range(num_ads)]
    
    # Read conflict relationships
    num_conflicts = int(input())
    for _ in range(num_conflicts):
        ad1, ad2 = map(int, input().split())
        # Create bidirectional conflict relationship
        conflict_ads[ad1][ad2] = True
        conflict_ads[ad2][ad1] = True
        
    # Create and populate problem instance
    problem = Problem(num_billboards, num_slots, num_ads)
    problem.billboards = billboards
    problem.slots = slots
    problem.ads_base_price = ads_base_price
    problem.ads_max_budget = ads_max_budgets
    problem.conflict_ads = conflict_ads
    
    return problem


def read_file(filename: str) -> Problem:
    """
    Read problem data from a file.
    
    File format is the same as console input format.
    
    Args:
        filename (str): Path to the input file
        
    Returns:
        Problem: Initialized problem instance with data from file
    """
    with open(filename, 'r') as f:
        # Read basic problem dimensions
        num_billboards, num_slots, num_ads = map(int, f.readline().strip().split())
        
        # Initialize billboard slot counts
        billboards = [0] * num_billboards
        
        # Read slot-to-billboard mapping and count slots per billboard
        slots = list(map(int, f.readline().strip().split()))
        for billboard_id in slots:
            billboards[billboard_id] += 1
            
        # Read advertisement base prices
        ads_base_price = list(map(int, f.readline().strip().split()))
        
        # Read ad max budget
        ads_max_budget = list(map(int, f.readline().strip().split()))
        
        # Initialize conflict relationships
        conflict_ads = [{} for _ in range(num_ads)]
        
        # Read conflict relationships
        num_conflicts = int(f.readline().strip())
        for _ in range(num_conflicts):
            ad1, ad2 = map(int, f.readline().strip().split())
            # Create bidirectional conflict relationship
            conflict_ads[ad1][ad2] = True
            conflict_ads[ad2][ad1] = True
            
    # Create and populate problem instance
    problem = Problem(num_billboards, num_slots, num_ads)
    problem.billboards = billboards
    problem.slots = slots
    problem.ads_base_price = ads_base_price
    problem.ads_max_budget = ads_max_budget
    problem.conflict_ads = conflict_ads
    
    return problem