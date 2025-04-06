import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta
import os
from scipy.optimize import linprog
import time
import json
from tqdm import tqdm
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Create output directory
output_dir = "ev_charging_results"
os.makedirs(output_dir, exist_ok=True)

# Constants
NUM_STATIONS = 10  # Number of charging stations
TIME_SLOTS = 24    # Number of time slots (hours in a day)
POWER_CAPACITY = 150  # kW, total power capacity of the charging station
MAX_POWER_PER_VEHICLE = 22  # kW, maximum power per vehicle
MIN_POWER_PER_VEHICLE = 3.3  # kW, minimum power required for a vehicle

# Enhanced features
RENEWABLE_INTEGRATION = True    # Consider renewable energy availability
DYNAMIC_PRICING = True          # Implement dynamic pricing based on grid load
PRIORITY_CLASSES = True         # Support priority classes for emergency vehicles
BATTERY_HEALTH_AWARE = True     # Consider battery health in charging decisions

class Vehicle:
    """Class representing an electric vehicle with charging requirements"""
    
    def __init__(self, vehicle_id, arrival_time, departure_time, initial_soc, target_soc, 
                 battery_capacity, max_charging_rate, priority=0):
        self.id = vehicle_id
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.initial_soc = initial_soc  # State of charge (0-1)
        self.target_soc = target_soc    # Target state of charge (0-1)
        self.battery_capacity = battery_capacity  # kWh
        self.max_charging_rate = max_charging_rate  # kW
        self.priority = priority  # 0: normal, 1: priority, 2: emergency
        
        # Calculate energy needed
        self.energy_needed = (target_soc - initial_soc) * battery_capacity  # kWh
        
        # Calculate flexible time window
        self.time_window = [max(0, arrival_time), min(TIME_SLOTS-1, departure_time)]
        self.flexibility = departure_time - arrival_time
        
        # Battery health parameters
        self.optimal_charging_rate = 0.5 * max_charging_rate  # kW - optimal rate for battery health
        self.charging_history = []
        
        # Cost related
        self.willingness_to_pay = np.random.uniform(0.2, 0.5) * (1 + 0.5 * priority)  # $/kWh
        self.utility_function = self._generate_utility_function()
        
    def _generate_utility_function(self):
        """Generate a non-linear utility function based on charging time preferences"""
        # Base utility is higher for charging during preferred times
        base_utility = np.zeros(TIME_SLOTS)
        
        # Generally prefer charging during off-peak hours (if not urgent)
        if self.flexibility > 6:  # If vehicle has flexibility
            off_peak_start = 22  # 10 PM
            off_peak_end = 6     # 6 AM
            
            for t in range(TIME_SLOTS):
                hour = t % 24
                if off_peak_start <= hour or hour < off_peak_end:
                    base_utility[t] = 1.0  # Higher utility for off-peak
                else:
                    base_utility[t] = 0.5  # Lower utility for peak hours
        else:
            # Less flexible vehicles have higher utility for immediate charging
            window_start, window_end = self.time_window
            window_length = window_end - window_start + 1
            
            for t in range(TIME_SLOTS):
                if window_start <= t <= window_end:
                    # Higher utility for earlier slots in the window (want to charge ASAP)
                    position_in_window = t - window_start
                    base_utility[t] = 1.0 - (position_in_window / window_length) * 0.5
        
        # Priority vehicles have more urgent utility functions
        if self.priority > 0:
            # Shift utility to prefer immediate charging
            for t in range(TIME_SLOTS):
                if t >= self.time_window[0]:
                    base_utility[t] *= (1 + 0.5 * self.priority)
        
        return base_utility
    
    def calculate_charging_benefit(self, time_slot, power_allocated, electricity_price):
        """Calculate the benefit of charging at a specific time slot"""
        if time_slot < self.time_window[0] or time_slot > self.time_window[1]:
            return -float('inf')  # Cannot charge outside time window
        
        # Base benefit from utility function
        benefit = self.utility_function[time_slot] * power_allocated
        
        # Subtract cost
        cost = power_allocated * electricity_price
        benefit -= cost
        
        # Battery health consideration
        if BATTERY_HEALTH_AWARE:
            # Penalty for charging rates far from optimal
            rate_difference = abs(power_allocated - self.optimal_charging_rate)
            health_penalty = 0.1 * rate_difference * power_allocated
            benefit -= health_penalty
        
        return benefit
    
    def get_desired_power(self, time_slot, remaining_energy):
        """Get desired charging power for a time slot"""
        if time_slot < self.time_window[0] or time_slot > self.time_window[1]:
            return 0
        
        # Calculate remaining time slots
        remaining_slots = self.time_window[1] - time_slot + 1
        
        if remaining_slots <= 0:
            return 0
        
        # Calculate minimum required power to meet target
        min_required_power = remaining_energy / remaining_slots
        
        # Adjust for battery health if needed
        if BATTERY_HEALTH_AWARE and remaining_slots > 1:
            desired_power = min(self.optimal_charging_rate, self.max_charging_rate)
            desired_power = max(desired_power, min_required_power)
        else:
            # Last chance to charge, use maximum available
            desired_power = min(remaining_energy, self.max_charging_rate)
        
        return desired_power
    
    def __str__(self):
        return f"Vehicle {self.id}: Arrival={self.arrival_time}, Departure={self.departure_time}, " \
               f"Energy needed={self.energy_needed:.2f} kWh, Priority={self.priority}"


class ChargingStation:
    """Class representing a charging station with multiple charging points"""
    
    def __init__(self, station_id, num_chargers, max_power):
        self.id = station_id
        self.num_chargers = num_chargers
        self.max_power = max_power  # Maximum power capacity of the station (kW)
        self.chargers_occupied = np.zeros(TIME_SLOTS)  # Track charger occupancy over time
        self.power_consumption = np.zeros(TIME_SLOTS)  # Track power consumption over time
        self.vehicle_assignments = {t: [] for t in range(TIME_SLOTS)}  # Vehicles assigned per time slot
        self.revenue = 0.0  # Track station revenue
    
    def can_accommodate(self, time_slot, num_chargers_needed=1):
        """Check if the station can accommodate additional chargers at a time slot"""
        return self.chargers_occupied[time_slot] + num_chargers_needed <= self.num_chargers
    
    def get_available_power(self, time_slot):
        """Get available power at a time slot"""
        return max(0, self.max_power - self.power_consumption[time_slot])
    
    def assign_vehicle(self, vehicle, time_slot, power_allocated, price):
        """Assign a vehicle to a charging slot"""
        if self.can_accommodate(time_slot):
            self.chargers_occupied[time_slot] += 1
            self.power_consumption[time_slot] += power_allocated
            self.vehicle_assignments[time_slot].append((vehicle.id, power_allocated))
            self.revenue += power_allocated * price
            return True
        return False
    
    def get_utilization(self):
        """Calculate station utilization metrics"""
        charger_utilization = np.mean(self.chargers_occupied / self.num_chargers)
        power_utilization = np.mean(self.power_consumption / self.max_power)
        return {
            'charger_utilization': charger_utilization,
            'power_utilization': power_utilization,
            'revenue': self.revenue
        }


class GridModel:
    """Model representing the electricity grid with renewable integration"""
    
    def __init__(self, time_slots=TIME_SLOTS):
        self.time_slots = time_slots
        self.base_load = self._generate_base_load()
        self.renewable_generation = self._generate_renewable_generation() if RENEWABLE_INTEGRATION else np.zeros(time_slots)
        self.base_prices = self._generate_base_prices()
        
    def _generate_base_load(self):
        """Generate a realistic base load profile for the grid"""
        # Morning peak: 7-9 AM, Evening peak: 6-8 PM
        base_load = np.zeros(self.time_slots)
        for t in range(self.time_slots):
            hour = t % 24
            if 7 <= hour < 9:  # Morning peak
                base_load[t] = np.random.uniform(0.7, 0.9)
            elif 18 <= hour < 20:  # Evening peak
                base_load[t] = np.random.uniform(0.8, 1.0)
            elif 9 <= hour < 17:  # Working hours
                base_load[t] = np.random.uniform(0.5, 0.7)
            elif 20 <= hour < 23:  # Evening
                base_load[t] = np.random.uniform(0.6, 0.8)
            else:  # Night
                base_load[t] = np.random.uniform(0.3, 0.5)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, self.time_slots)
        base_load += noise
        base_load = np.clip(base_load, 0.2, 1.0)
        return base_load
    
    def _generate_renewable_generation(self):
        """Generate renewable energy availability profile"""
        # Solar generation: peaks at noon
        solar = np.zeros(self.time_slots)
        for t in range(self.time_slots):
            hour = t % 24
            if 6 <= hour < 20:  # Daylight hours
                # Bell curve with peak at noon (hour 12)
                solar[t] = 0.8 * np.exp(-0.5 * ((hour - 13) / 3) ** 2)
                # Add some noise for cloud cover, etc.
                solar[t] *= np.random.uniform(0.7, 1.0)
            
        # Wind generation: more random but with some daily patterns
        wind = np.zeros(self.time_slots)
        for t in range(self.time_slots):
            hour = t % 24
            # More wind at night in this model
            base_wind = 0.2 + 0.3 * (1 if hour < 6 or hour >= 18 else 0.5)
            wind[t] = base_wind * np.random.uniform(0.5, 1.5)
        
        # Combine solar and wind (with different weights)
        renewable = 0.7 * solar + 0.3 * wind
        renewable = np.clip(renewable, 0, 1.0)  # Normalize to [0, 1]
        return renewable
    
    def _generate_base_prices(self):
        """Generate base electricity prices"""
        # Base prices follow load to some extent
        base_prices = 0.15 + 0.3 * self.base_load
        
        # Add time-of-use pricing structure
        for t in range(self.time_slots):
            hour = t % 24
            if 7 <= hour < 10 or 17 <= hour < 21:  # Peak hours
                base_prices[t] *= 1.5
            elif 0 <= hour < 5:  # Super off-peak
                base_prices[t] *= 0.7
        
        return base_prices
    
    def get_price(self, time_slot, additional_load=0):
        """Get electricity price for a time slot"""
        if not DYNAMIC_PRICING:
            return self.base_prices[time_slot]
        
        # Dynamic pricing based on current load
        current_load = self.base_load[time_slot]
        renewable_factor = 1.0
        
        if RENEWABLE_INTEGRATION:
            # Discount price when renewable generation is high
            renewable_available = self.renewable_generation[time_slot]
            renewable_factor = max(0.6, 1.0 - 0.4 * renewable_available)
        
        # Price increases with additional load (quadratic relationship)
        load_factor = 1.0 + 0.5 * (additional_load / POWER_CAPACITY) ** 2
        
        # Combined price factors
        price = self.base_prices[time_slot] * load_factor * renewable_factor
        
        return price
    
    def get_grid_carbon_intensity(self, time_slot):
        """Get carbon intensity of electricity at a time slot"""
        if RENEWABLE_INTEGRATION:
            # Lower carbon intensity when renewable penetration is high
            renewable_factor = self.renewable_generation[time_slot]
            # Base carbon intensity 400 g/kWh, can go down to 100 g/kWh with 100% renewables
            return 400 * (1 - 0.75 * renewable_factor)
        else:
            # Fixed average carbon intensity
            return 400  # g CO2/kWh


class DataGenerator:
    """Generate realistic EV charging demand datasets"""
    
    def __init__(self, num_vehicles=100):
        self.num_vehicles = num_vehicles
        # Vehicle types: (battery_capacity, max_charging_rate)
        self.vehicle_types = {
            'compact': (40, 7.4),   # 40 kWh, 7.4 kW (e.g., Nissan Leaf)
            'sedan': (70, 11),      # 70 kWh, 11 kW (e.g., Tesla Model 3)
            'suv': (100, 16.5),     # 100 kWh, 16.5 kW (e.g., Tesla Model X)
            'luxury': (120, 22),    # 120 kWh, 22 kW (e.g., Lucid Air)
            'commercial': (80, 22)  # 80 kWh, 22 kW (e.g., delivery van)
        }
        # Arrival patterns for different user groups
        self.user_groups = {
            'commuter': {
                'arrival_peak': 8,      # 8 AM peak arrival
                'departure_peak': 17,   # 5 PM peak departure
                'arrival_std': 1,       # Standard deviation of 1 hour
                'departure_std': 1.5,   # Standard deviation of 1.5 hours
                'min_stay': 7,          # Minimum stay duration of 7 hours
                'initial_soc_mean': 0.4,  # Mean initial SOC
                'target_soc_mean': 0.9   # Mean target SOC
            },
            'shopper': {
                'arrival_peak': 12,     # 12 PM peak arrival
                'departure_peak': 15,   # 3 PM peak departure
                'arrival_std': 2,       # Standard deviation of 2 hours
                'departure_std': 2,     # Standard deviation of 2 hours
                'min_stay': 1,          # Minimum stay duration of 1 hour
                'initial_soc_mean': 0.6,  # Mean initial SOC
                'target_soc_mean': 0.8   # Mean target SOC
            },
            'resident': {
                'arrival_peak': 18,     # 6 PM peak arrival
                'departure_peak': 8,    # 8 AM peak departure
                'arrival_std': 3,       # Standard deviation of 3 hours
                'departure_std': 2,     # Standard deviation of 2 hours  
                'min_stay': 10,         # Minimum stay duration of 10 hours
                'initial_soc_mean': 0.3,  # Mean initial SOC
                'target_soc_mean': 0.95  # Mean target SOC
            }
        }
    
    def generate_dataset(self):
        """Generate a dataset of EV charging requests"""
        vehicles = []
        
        # Distribute vehicles across user groups
        group_distribution = {
            'commuter': 0.5,   # 50% commuters
            'shopper': 0.3,    # 30% shoppers
            'resident': 0.2    # 20% residents
        }
        
        vehicle_types_dist = {
            'compact': 0.3,    # 30% compact cars
            'sedan': 0.4,      # 40% sedans
            'suv': 0.2,        # 20% SUVs
            'luxury': 0.05,    # 5% luxury cars
            'commercial': 0.05 # 5% commercial vehicles
        }
        
        for i in range(self.num_vehicles):
            # Assign user group based on distribution
            user_group = np.random.choice(
                list(group_distribution.keys()),
                p=list(group_distribution.values())
            )
            group_params = self.user_groups[user_group]
            
            # Assign vehicle type based on distribution
            vehicle_type = np.random.choice(
                list(vehicle_types_dist.keys()),
                p=list(vehicle_types_dist.values())
            )
            battery_capacity, max_charging_rate = self.vehicle_types[vehicle_type]
            
            # Generate arrival and departure times based on normal distributions
            arrival_time = int(np.random.normal(
                group_params['arrival_peak'],
                group_params['arrival_std']
            ))
            
            # Ensure minimum stay duration and reasonable departure time
            min_departure = arrival_time + group_params['min_stay']
            departure_mean = max(min_departure, group_params['departure_peak'])
            
            departure_time = int(np.random.normal(
                departure_mean,
                group_params['departure_std']
            ))
            
            # Ensure times are within bounds and arrival < departure
            arrival_time = max(0, min(arrival_time, TIME_SLOTS - 2))
            departure_time = max(arrival_time + 1, min(departure_time, TIME_SLOTS - 1))
            
            # Generate initial and target SOC
            initial_soc = np.random.normal(
                group_params['initial_soc_mean'],
                0.1
            )
            initial_soc = max(0.1, min(initial_soc, 0.9))
            
            target_soc = np.random.normal(
                group_params['target_soc_mean'],
                0.05
            )
            target_soc = max(initial_soc + 0.1, min(target_soc, 0.99))
            
            # Assign priority (emergency vehicles, etc.)
            priority = 0
            if np.random.random() < 0.05:  # 5% priority vehicles
                priority = 1
            if np.random.random() < 0.01:  # 1% emergency vehicles
                priority = 2
            
            # Create vehicle
            vehicle = Vehicle(
                vehicle_id=i,
                arrival_time=arrival_time,
                departure_time=departure_time,
                initial_soc=initial_soc,
                target_soc=target_soc,
                battery_capacity=battery_capacity,
                max_charging_rate=max_charging_rate,
                priority=priority
            )
            
            vehicles.append(vehicle)
            
        return vehicles


class CentralizedScheduler:
    """Enhanced centralized scheduler for EV charging"""
    
    def __init__(self, stations, grid_model):
        self.stations = stations
        self.grid_model = grid_model
        self.schedule = {}  # Final schedule
        self.vehicle_assignments = {}  # Track which station each vehicle is assigned to
        self.time_slot_power = np.zeros(TIME_SLOTS)  # Track power consumption per time slot
        self.solution_metrics = {
            'social_welfare': 0,
            'user_satisfaction': 0,
            'revenue': 0,
            'carbon_emissions': 0,
            'unfulfilled_requests': 0,
            'fairness_index': 0
        }
        
    def optimize_schedule(self, vehicles):
        """Optimize the charging schedule for a set of vehicles"""
        start_time = time.time()
        
        print(f"Optimizing schedule for {len(vehicles)} vehicles...")
        
        # Sort vehicles by priority first, then by flexibility (less flexible first)
        sorted_vehicles = sorted(
            vehicles, 
            key=lambda v: (-v.priority, v.flexibility, v.arrival_time)
        )
        
        # First pass: allocate for emergency and priority vehicles
        high_priority_vehicles = [v for v in sorted_vehicles if v.priority > 0]
        regular_vehicles = [v for v in sorted_vehicles if v.priority == 0]
        
        fulfilled_requests = 0
        unfulfilled_requests = 0
        user_utilities = []
        
        # Handle priority vehicles first
        for vehicle in tqdm(high_priority_vehicles, desc="Processing priority vehicles"):
            if self._schedule_vehicle(vehicle):
                fulfilled_requests += 1
                user_utilities.append(self._calculate_user_utility(vehicle))
            else:
                unfulfilled_requests += 1
        
        # Then handle regular vehicles
        for vehicle in tqdm(regular_vehicles, desc="Processing regular vehicles"):
            if self._schedule_vehicle(vehicle):
                fulfilled_requests += 1
                user_utilities.append(self._calculate_user_utility(vehicle))
            else:
                unfulfilled_requests += 1
        
        # Second pass: try to optimize the schedule further
        self._refine_schedule()
        
        # Calculate final metrics
        self._calculate_metrics(sorted_vehicles)
        
        end_time = time.time()
        self.solution_metrics['runtime'] = end_time - start_time
        
        print(f"Schedule optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Fulfilled requests: {fulfilled_requests}/{len(vehicles)}")
        print(f"Social welfare: {self.solution_metrics['social_welfare']:.2f}")
        
        return self.schedule
    
    def _schedule_vehicle(self, vehicle):
        """Schedule a single vehicle using optimization approach"""
        # Find best charging strategy for this vehicle
        best_utility = -float('inf')
        best_station = None
        best_schedule = None
        
        window_start, window_end = vehicle.time_window
        
        # Calculate remaining energy needed
        remaining_energy = vehicle.energy_needed
        
        # Try each station
        for station in self.stations:
            # Calculate charging schedule using optimization
            schedule = self._optimize_vehicle_charging(vehicle, station)
            
            if schedule:
                # Calculate utility for this schedule
                utility = 0
                for t in range(window_start, window_end + 1):
                    if t in schedule:
                        power = schedule[t]
                        price = self.grid_model.get_price(t, self.time_slot_power[t])
                        utility += vehicle.calculate_charging_benefit(t, power, price)
                
                if utility > best_utility:
                    best_utility = utility
                    best_station = station
                    best_schedule = schedule
        
        # If a feasible schedule was found, implement it
        if best_schedule and best_station:
            # Update the schedule
            vehicle_id = vehicle.id
            self.schedule[vehicle_id] = {'station': best_station.id, 'charging_profile': {}}
            
            total_energy = 0
            
            for t, power in best_schedule.items():
                if power > 0:
                    price = self.grid_model.get_price(t, self.time_slot_power[t])
                    best_station.assign_vehicle(vehicle, t, power, price)
                    self.time_slot_power[t] += power
                    self.schedule[vehicle_id]['charging_profile'][t] = power
                    energy = power  # Simplified: 1 hour slots, so energy (kWh) = power (kW)
                    total_energy += energy
            
            # Record which station this vehicle is assigned to
            self.vehicle_assignments[vehicle_id] = best_station.id
            
            # Check if the vehicle receives enough energy
            if total_energy >= 0.95 * vehicle.energy_needed:  # Allow for small rounding errors
                return True
            else:
                # Not enough energy was allocated, remove the partial allocation
                for t in best_schedule:
                    if t in self.schedule[vehicle_id]['charging_profile']:
                        power = self.schedule[vehicle_id]['charging_profile'][t]
                        self.time_slot_power[t] -= power
                
                del self.schedule[vehicle_id]
                if vehicle_id in self.vehicle_assignments:
                    del self.vehicle_assignments[vehicle_id]
                return False
        
        return False
    
    def _optimize_vehicle_charging(self, vehicle, station):
        """Optimize charging profile for a vehicle at a station"""
        window_start, window_end = vehicle.time_window
        window_duration = window_end - window_start + 1
        
        if window_duration <= 0:
            return None
        
        # Check if there are any available chargers in the time window
        available_slots = 0
        for t in range(window_start, window_end + 1):
            if station.can_accommodate(t):
                available_slots += 1
        
        if available_slots == 0:
            return None
        
        # Calculate energy needed
        energy_needed = vehicle.energy_needed
        
        # Prepare for optimization
        # We'll use linear programming to maximize utility while meeting constraints
        
        # Decision variables: power allocated at each time slot
        # Constraints:
        # 1. Power allocated <= min(station available power, vehicle max charging rate)
        # 2. Sum of energy charged >= energy needed
        # 3. Power allocated >= 0
        
        # Objective: maximize utility (considering time preferences, prices, battery health)
        
        # Initialize variables
        power_allocations = {}
        remaining_energy = energy_needed
        
        # Simple greedy algorithm based on utility
        utilities = []
        for t in range(window_start, window_end + 1):
            if station.can_accommodate(t):
                available_power = min(station.get_available_power(t), vehicle.max_charging_rate)
                price = self.grid_model.get_price(t, self.time_slot_power[t])
                utility = vehicle.calculate_charging_benefit(t, available_power, price)
                utilities.append((t, utility, available_power))
        
        # Sort by utility (highest first)
        utilities.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate power greedily
        for t, utility, available_power in utilities:
            if remaining_energy <= 0:
                break
                
            # Determine power to allocate
            power = min(available_power, remaining_energy)
            
            # Ensure minimum charging rate if charging occurs
            if power < MIN_POWER_PER_VEHICLE:
                if available_power >= MIN_POWER_PER_VEHICLE:
                    power = MIN_POWER_PER_VEHICLE
                else:
                    continue  # Skip if we can't meet minimum power
            
            power_allocations[t] = power
            remaining_energy -= power  # Simplification: 1-hour slots, so energy = power
        
        # Check if all energy needs are met
        if remaining_energy > 0.05 * energy_needed:  # Allow for small rounding errors
            # Try again with a more aggressive approach if we're not meeting needs
            # Focus on filling slots sequentially to ensure we get enough energy
            power_allocations = {}
            remaining_energy = energy_needed
            
            for t in range(window_start, window_end + 1):
                if remaining_energy <= 0:
                    break
                    
                if station.can_accommodate(t):
                    available_power = min(station.get_available_power(t), vehicle.max_charging_rate)
                    
                    # Determine power to allocate
                    power = min(available_power, remaining_energy)
                    
                    # Ensure minimum charging rate if charging occurs
                    if power < MIN_POWER_PER_VEHICLE:
                        if available_power >= MIN_POWER_PER_VEHICLE:
                            power = MIN_POWER_PER_VEHICLE
                        else:
                            continue  # Skip if we can't meet minimum power
                    
                    power_allocations[t] = power
                    remaining_energy -= power
        
        # Return the schedule if we managed to allocate enough energy
        if remaining_energy <= 0.05 * energy_needed:  # Allow for small rounding errors
            return power_allocations
        else:
            return None
    
    def _refine_schedule(self):
        """Refine the schedule to improve efficiency"""
        # Look for opportunities to shift charging to times with more renewable energy
        # or better prices, without compromising vehicle charging needs
        
        if not RENEWABLE_INTEGRATION and not DYNAMIC_PRICING:
            return  # No need to refine if neither feature is enabled
        
        for vehicle_id, assignment in self.schedule.items():
            station_id = assignment['station']
            station = next(s for s in self.stations if s.id == station_id)
            charging_profile = assignment['charging_profile'].copy()
            
            # Calculate total energy allocated
            total_energy = sum(charging_profile.values())
            
            # Get time slots with charging
            charging_slots = sorted(charging_profile.keys())
            
            # Look for potential improvements
            improvements_made = False
            
            for t_from in charging_slots:
                power_from = charging_profile[t_from]
                
                # Current cost/benefit metrics
                price_from = self.grid_model.get_price(t_from, self.time_slot_power[t_from] - power_from)
                carbon_from = self.grid_model.get_grid_carbon_intensity(t_from)
                
                # Find better time slots
                for t_to in range(min(charging_slots), max(charging_slots) + 1):
                    # Skip if it's the same slot or already has charging
                    if t_to == t_from or t_to in charging_profile:
                        continue
                    
                    # Check if station can accommodate at the new time
                    if not station.can_accommodate(t_to):
                        continue
                    
                    # Get new price and carbon intensity
                    price_to = self.grid_model.get_price(t_to, self.time_slot_power[t_to] + power_from)
                    carbon_to = self.grid_model.get_grid_carbon_intensity(t_to)
                    
                    # Calculate improvement (consider both price and carbon)
                    price_improvement = price_from - price_to
                    carbon_improvement = carbon_from - carbon_to
                    
                    # Weighted score for improvement (prioritize price over carbon)
                    improvement_score = 0.7 * price_improvement + 0.3 * carbon_improvement / 100
                    
                    if improvement_score > 0.01:  # Significant improvement threshold
                        # Make the change
                        # Remove from old slot
                        station.power_consumption[t_from] -= power_from
                        self.time_slot_power[t_from] -= power_from
                        del charging_profile[t_from]
                        
                        # Add to new slot
                        station.power_consumption[t_to] += power_from
                        self.time_slot_power[t_to] += power_from
                        charging_profile[t_to] = power_from
                        
                        # Update revenue (price may have changed)
                        station.revenue -= power_from * price_from
                        station.revenue += power_from * price_to
                        
                        improvements_made = True
                        break  # Only make one change per iteration
                
                if improvements_made:
                    break
            
            # Update the schedule if changes were made
            if improvements_made:
                self.schedule[vehicle_id]['charging_profile'] = charging_profile
    
    def _calculate_user_utility(self, vehicle):
        """Calculate total utility for a vehicle based on its charging schedule"""
        if vehicle.id not in self.schedule:
            return 0
        
        utility = 0
        charging_profile = self.schedule[vehicle.id]['charging_profile']
        
        for t, power in charging_profile.items():
            price = self.grid_model.get_price(t, self.time_slot_power[t] - power)  # Remove vehicle's contribution to get price
            utility += vehicle.calculate_charging_benefit(t, power, price)
        
        return utility
    
    def _calculate_metrics(self, vehicles):
        """Calculate comprehensive metrics for the schedule"""
        # Calculate social welfare (sum of utilities)
        total_utility = sum(self._calculate_user_utility(v) for v in vehicles if v.id in self.schedule)
        self.solution_metrics['social_welfare'] = total_utility
        
        # Calculate user satisfaction
        successful_vehicles = [v for v in vehicles if v.id in self.schedule]
        if successful_vehicles:
            satisfaction_scores = []
            for vehicle in successful_vehicles:
                # Calculate energy received vs energy needed
                profile = self.schedule[vehicle.id]['charging_profile']
                energy_received = sum(profile.values())  # Assuming 1-hour slots
                satisfaction = min(energy_received / vehicle.energy_needed, 1.0)
                satisfaction_scores.append(satisfaction)
            
            self.solution_metrics['user_satisfaction'] = np.mean(satisfaction_scores)
        else:
            self.solution_metrics['user_satisfaction'] = 0
        
        # Calculate revenue and carbon emissions
        total_revenue = 0
        total_carbon = 0
        
        for t in range(TIME_SLOTS):
            power_consumed = self.time_slot_power[t]
            if power_consumed > 0:
                price = self.grid_model.get_price(t, power_consumed)
                carbon = self.grid_model.get_grid_carbon_intensity(t)
                
                total_revenue += power_consumed * price
                total_carbon += power_consumed * carbon
        
        self.solution_metrics['revenue'] = total_revenue
        self.solution_metrics['carbon_emissions'] = total_carbon
        
        # Count unfulfilled requests
        self.solution_metrics['unfulfilled_requests'] = len([v for v in vehicles if v.id not in self.schedule])
        
        # Calculate Jain's fairness index
        if successful_vehicles:
            # For fairness, use ratio of energy_received/energy_needed
            allocations = []
            for vehicle in successful_vehicles:
                profile = self.schedule[vehicle.id]['charging_profile']
                energy_received = sum(profile.values())
                allocations.append(energy_received / vehicle.energy_needed)
            
            # Jain's fairness index: (sum(x_i))² / (n * sum(x_i²))
            sum_allocations = sum(allocations)
            sum_squares = sum(x**2 for x in allocations)
            n = len(allocations)
            
            if sum_squares > 0:
                fairness = (sum_allocations**2) / (n * sum_squares)
                self.solution_metrics['fairness_index'] = fairness
            else:
                self.solution_metrics['fairness_index'] = 0
        else:
            self.solution_metrics['fairness_index'] = 0


class Visualizer:
    """Class for visualizing results from the EV charging scheduler"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        # Use a modern, visually appealing style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_power_consumption(self, scheduler, grid_model):
        """Plot power consumption over time alongside grid load and renewable generation"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        time_slots = range(TIME_SLOTS)
        
        # Plot EV charging power consumption
        ax1.bar(time_slots, scheduler.time_slot_power, color='#4285F4', alpha=0.7, label='EV Charging Load')
        
        # Plot grid base load (scaled to kW for comparison)
        scaled_base_load = grid_model.base_load * POWER_CAPACITY * 0.8
        ax1.plot(time_slots, scaled_base_load, 'r--', linewidth=2, label='Grid Base Load (scaled)')
        
        # Format x-axis as hours
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Power (kW)')
        ax1.set_title('EV Charging Power Consumption Over Time', fontsize=14)
        
        # Set x-axis ticks to represent hours
        ax1.set_xticks(range(0, TIME_SLOTS, 2))
        ax1.set_xticklabels([f'{h%24:02d}:00' for h in range(0, TIME_SLOTS, 2)], rotation=45)
        
        if RENEWABLE_INTEGRATION:
            # Create second y-axis for renewable generation
            ax2 = ax1.twinx()
            
            # Plot renewable generation (scaled to match power scale)
            scaled_renewable = grid_model.renewable_generation * POWER_CAPACITY
            ax2.plot(time_slots, scaled_renewable, color='#34A853', linewidth=2, label='Renewable Generation')
            ax2.set_ylabel('Renewable Generation (kW)')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/power_consumption.png", dpi=300)
        plt.close()
    
    def plot_station_utilization(self, stations):
        """Plot utilization metrics for each charging station"""
        num_stations = len(stations)
        
        # Collect utilization data
        charger_util = [s.get_utilization()['charger_utilization'] * 100 for s in stations]
        power_util = [s.get_utilization()['power_utilization'] * 100 for s in stations]
        revenue = [s.get_utilization()['revenue'] for s in stations]
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot charger utilization
        axs[0].bar(range(num_stations), charger_util, color='#4285F4')
        axs[0].set_ylabel('Charger Utilization (%)')
        axs[0].set_title('Charging Station Performance Metrics', fontsize=14)
        
        # Plot power utilization
        axs[1].bar(range(num_stations), power_util, color='#EA4335')
        axs[1].set_ylabel('Power Utilization (%)')
        
        # Plot revenue
        axs[2].bar(range(num_stations), revenue, color='#34A853')
        axs[2].set_xlabel('Station ID')
        axs[2].set_ylabel('Revenue ($)')
        
        # Set x-ticks
        axs[2].set_xticks(range(num_stations))
        axs[2].set_xticklabels([f'Station {s.id}' for s in stations], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/station_utilization.png", dpi=300)
        plt.close()
    
    def plot_charging_profiles(self, scheduler, vehicles, sample_size=5):
        """Plot charging profiles for a sample of vehicles"""
        # Select a sample of vehicles that have charging schedules
        scheduled_vehicles = [v for v in vehicles if v.id in scheduler.schedule]
        if not scheduled_vehicles:
            print("No vehicles with schedules to plot")
            return
        
        # Take a representative sample or all if sample_size > available
        sample_size = min(sample_size, len(scheduled_vehicles))
        # Try to include different priority levels
        priorities = sorted(set(v.priority for v in scheduled_vehicles))
        
        # Ensure we have at least one vehicle from each priority if possible
        sample_vehicles = []
        for priority in priorities:
            priority_vehicles = [v for v in scheduled_vehicles if v.priority == priority]
            if priority_vehicles:
                sample_vehicles.append(random.choice(priority_vehicles))
        
        # Fill the rest of the sample randomly if needed
        remaining_vehicles = [v for v in scheduled_vehicles if v not in sample_vehicles]
        if remaining_vehicles and len(sample_vehicles) < sample_size:
            sample_vehicles.extend(random.sample(remaining_vehicles, 
                                              min(sample_size - len(sample_vehicles), len(remaining_vehicles))))
        
        # Create subplot for each vehicle
        fig, axs = plt.subplots(len(sample_vehicles), 1, figsize=(12, 3*len(sample_vehicles)), sharex=True)
        
        # Handle case with only one vehicle
        if len(sample_vehicles) == 1:
            axs = [axs]
        
        for i, vehicle in enumerate(sample_vehicles):
            profile = scheduler.schedule[vehicle.id]['charging_profile']
            times = sorted(profile.keys())
            powers = [profile[t] for t in times]
            
            # Plot charging profile as a step function
            axs[i].step(times, powers, where='post', linewidth=2, 
                       color=['#4285F4', '#EA4335', '#FBBC05'][min(vehicle.priority, 2)])
            
            # Fill area under the curve
            axs[i].fill_between(times, powers, step='post', alpha=0.3, 
                              color=['#4285F4', '#EA4335', '#FBBC05'][min(vehicle.priority, 2)])
            
            # Show vehicle arrival and departure with vertical lines
            axs[i].axvline(vehicle.arrival_time, color='green', linestyle='--', alpha=0.7, label='Arrival')
            axs[i].axvline(vehicle.departure_time, color='red', linestyle='--', alpha=0.7, label='Departure')
            
            # Show max charging rate as a horizontal line
            axs[i].axhline(vehicle.max_charging_rate, color='black', linestyle=':', alpha=0.5, 
                          label='Max Rate')
            
            # Add vehicle info
            priority_labels = {0: 'Standard', 1: 'Priority', 2: 'Emergency'}
            axs[i].set_title(f"Vehicle {vehicle.id} ({priority_labels[vehicle.priority]}): " + 
                           f"Needs {vehicle.energy_needed:.1f} kWh, " +
                           f"Battery: {vehicle.battery_capacity:.1f} kWh", fontsize=10)
            
            axs[i].set_ylabel('Power (kW)')
            
            if i == 0:
                axs[i].legend(loc='upper right')
        
        # Set common x-axis properties
        axs[-1].set_xlabel('Hour of Day')
        axs[-1].set_xticks(range(0, TIME_SLOTS, 2))
        axs[-1].set_xticklabels([f'{h%24:02d}:00' for h in range(0, TIME_SLOTS, 2)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/charging_profiles.png", dpi=300)
        plt.close()
    
    def plot_price_vs_renewable(self, grid_model):
        """Plot electricity price vs renewable generation over time"""
        if not DYNAMIC_PRICING or not RENEWABLE_INTEGRATION:
            return
        
        time_slots = range(TIME_SLOTS)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot prices
        prices = [grid_model.get_price(t, 0) for t in time_slots]  # Base prices without additional load
        ax1.plot(time_slots, prices, 'b-', linewidth=2, label='Electricity Price')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Price ($/kWh)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Set x-axis ticks to represent hours
        ax1.set_xticks(range(0, TIME_SLOTS, 2))
        ax1.set_xticklabels([f'{h%24:02d}:00' for h in range(0, TIME_SLOTS, 2)], rotation=45)
        
        # Create second y-axis for renewable generation
        ax2 = ax1.twinx()
        ax2.plot(time_slots, grid_model.renewable_generation, 'g-', linewidth=2, 
                label='Renewable Generation')
        ax2.set_ylabel('Renewable Generation (normalized)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Add title
        plt.title('Electricity Price vs Renewable Generation Over Time', fontsize=14)
        
        # Add legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/price_vs_renewable.png", dpi=300)
        plt.close()
    
    def plot_solution_metrics(self, scheduler_metrics):
        """Plot key metrics from the scheduling solution"""
        # Extract metrics
        metrics = scheduler_metrics.copy()
        
        # Remove numerical metrics that shouldn't be on the same scale
        excluded_metrics = ['runtime', 'unfulfilled_requests', 'carbon_emissions', 'revenue']
        plot_metrics = {k: v for k, v in metrics.items() if k not in excluded_metrics}
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create bar chart
        bars = ax.bar(plot_metrics.keys(), plot_metrics.values(), color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.set_ylim(0, max(1.0, max(plot_metrics.values()) * 1.1))  # Set y-limit with some headroom
        ax.set_ylabel('Score (normalized)')
        ax.set_title('Key Performance Metrics', fontsize=14)
        
        # Make x-axis labels more readable
        ax.set_xticklabels([' '.join(k.split('_')).title() for k in plot_metrics.keys()], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/solution_metrics.png", dpi=300)
        plt.close()
        
        # Create a separate chart for excluded metrics
        excluded_values = {k: metrics[k] for k in excluded_metrics if k in metrics}
        
        if excluded_values:
            fig, axes = plt.subplots(len(excluded_values), 1, figsize=(8, 3*len(excluded_values)))
            
            # Handle case with only one metric
            if len(excluded_values) == 1:
                axes = [axes]
            
            for i, (metric, value) in enumerate(excluded_values.items()):
                axes[i].bar([metric], [value], color='#4285F4')
                axes[i].set_title(f"{' '.join(metric.split('_')).title()}: {value:.2f}")
                
                # Format y-axis based on metric
                if metric == 'runtime':
                    axes[i].set_ylabel('Seconds')
                elif metric == 'unfulfilled_requests':
                    axes[i].set_ylabel('Count')
                elif metric == 'carbon_emissions':
                    axes[i].set_ylabel('kg CO2')
                elif metric == 'revenue':
                    axes[i].set_ylabel('$')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/additional_metrics.png", dpi=300)
            plt.close()
    
    def plot_fairness_distribution(self, scheduler, vehicles):
        """Plot distribution of charging allocations to visualize fairness"""
        # Get vehicles that received charging allocation
        scheduled_vehicles = [v for v in vehicles if v.id in scheduler.schedule]
        
        if not scheduled_vehicles:
            print("No vehicles with schedules to plot fairness distribution")
            return
        
        # Calculate allocation ratios (energy received / energy needed)
        allocation_ratios = []
        for vehicle in scheduled_vehicles:
            profile = scheduler.schedule[vehicle.id]['charging_profile']
            energy_received = sum(profile.values())  # Assuming 1-hour slots
            allocation_ratio = energy_received / vehicle.energy_needed
            allocation_ratios.append((vehicle.id, allocation_ratio, vehicle.priority))
        
        # Sort by allocation ratio
        allocation_ratios.sort(key=lambda x: x[1])
        
        # Extract data for plotting
        vehicle_ids = [str(x[0]) for x in allocation_ratios]
        ratios = [x[1] for x in allocation_ratios]
        priorities = [x[2] for x in allocation_ratios]
        
        # Create color map based on priority
        colors = ['#4285F4', '#EA4335', '#FBBC05']  # blue, red, yellow
        bar_colors = [colors[min(p, 2)] for p in priorities]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar chart
        bars = ax.bar(range(len(vehicle_ids)), ratios, color=bar_colors)
        
        # Set labels and title
        ax.set_xlabel('Vehicle ID (ordered by allocation ratio)')
        ax.set_ylabel('Allocation Ratio (energy received / energy needed)')
        ax.set_title('Distribution of Charging Allocations Among Vehicles', fontsize=14)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[0], label='Standard Priority'),
            Patch(facecolor=colors[1], label='Priority'),
            Patch(facecolor=colors[2], label='Emergency')
        ]
        ax.legend(handles=legend_elements)
        
        # Add fairness index as text
        fairness_index = scheduler.solution_metrics.get('fairness_index', 0)
        ax.text(0.02, 0.95, f"Jain's Fairness Index: {fairness_index:.4f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set x-ticks (show only a subset for readability if many vehicles)
        if len(vehicle_ids) > 20:
            step = max(1, len(vehicle_ids) // 10)
            ax.set_xticks(range(0, len(vehicle_ids), step))
            ax.set_xticklabels([vehicle_ids[i] for i in range(0, len(vehicle_ids), step)], rotation=45)
        else:
            ax.set_xticks(range(len(vehicle_ids)))
            ax.set_xticklabels(vehicle_ids, rotation=45)
        
        # Add horizontal line at ratio = 1.0
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fairness_distribution.png", dpi=300)
        plt.close()
    
    def plot_demand_distribution(self, vehicles):
        """Plot distribution of vehicle arrivals and energy demands"""
        # Extract data
        arrival_times = [v.arrival_time for v in vehicles]
        departure_times = [v.departure_time for v in vehicles]
        energy_needed = [v.energy_needed for v in vehicles]
        flexibility = [v.flexibility for v in vehicles]
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot arrival distribution
        axs[0, 0].hist(arrival_times, bins=24, alpha=0.7, color='#4285F4')
        axs[0, 0].set_title('Vehicle Arrival Distribution', fontsize=12)
        axs[0, 0].set_xlabel('Hour of Day')
        axs[0, 0].set_ylabel('Number of Vehicles')
        axs[0, 0].set_xticks(range(0, 24, 2))
        axs[0, 0].set_xticklabels([f'{h%24:02d}:00' for h in range(0, 24, 2)])
        
        # Plot departure distribution
        axs[0, 1].hist(departure_times, bins=24, alpha=0.7, color='#EA4335')
        axs[0, 1].set_title('Vehicle Departure Distribution', fontsize=12)
        axs[0, 1].set_xlabel('Hour of Day')
        axs[0, 1].set_ylabel('Number of Vehicles')
        axs[0, 1].set_xticks(range(0, 24, 2))
        axs[0, 1].set_xticklabels([f'{h%24:02d}:00' for h in range(0, 24, 2)])
        
        # Plot energy demand distribution
        axs[1, 0].hist(energy_needed, bins=20, alpha=0.7, color='#FBBC05')
        axs[1, 0].set_title('Energy Demand Distribution', fontsize=12)
        axs[1, 0].set_xlabel('Energy Needed (kWh)')
        axs[1, 0].set_ylabel('Number of Vehicles')
        
        # Plot flexibility distribution
        axs[1, 1].hist(flexibility, bins=15, alpha=0.7, color='#34A853')
        axs[1, 1].set_title('Charging Flexibility Distribution', fontsize=12)
        axs[1, 1].set_xlabel('Flexibility (hours)')
        axs[1, 1].set_ylabel('Number of Vehicles')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/demand_distribution.png", dpi=300)
        plt.close()
    
    def save_metrics_to_json(self, scheduler_metrics, filename="metrics.json"):
        """Save all metrics to a JSON file"""
        with open(f"{self.output_dir}/{filename}", 'w') as f:
            # Round values for better readability
            rounded_metrics = {k: round(v, 4) if isinstance(v, float) else v 
                           for k, v in scheduler_metrics.items()}
            json.dump(rounded_metrics, f, indent=4)


def run_simulation(num_vehicles=100, num_stations=NUM_STATIONS):
    """Run a complete simulation of the EV charging scheduling system"""
    print(f"\n{'='*50}")
    print(f"Starting EV Charging Simulation with {num_vehicles} vehicles and {num_stations} stations")
    print(f"{'='*50}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate stations
    stations = []
    for i in range(num_stations):
        # Varying number of chargers and power capacity per station
        num_chargers = random.randint(3, 8)
        max_power = random.randint(40, 150)  # kW
        stations.append(ChargingStation(i, num_chargers, max_power))
    
    # Create grid model
    grid_model = GridModel()
    
    # Generate vehicle data
    data_generator = DataGenerator(num_vehicles=num_vehicles)
    vehicles = data_generator.generate_dataset()
    
    print(f"Generated {len(vehicles)} vehicles with charging requests")
    
    # Create scheduler
    scheduler = CentralizedScheduler(stations, grid_model)
    
    # Run optimization
    scheduler.optimize_schedule(vehicles)
    
    # Create visualizer and generate plots
    visualizer = Visualizer(output_dir)
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    visualizer.plot_power_consumption(scheduler, grid_model)
    visualizer.plot_station_utilization(stations)
    visualizer.plot_charging_profiles(scheduler, vehicles)
    visualizer.plot_price_vs_renewable(grid_model)
    visualizer.plot_solution_metrics(scheduler.solution_metrics)
    visualizer.plot_fairness_distribution(scheduler, vehicles)
    visualizer.plot_demand_distribution(vehicles)
    
    # Save metrics to JSON
    visualizer.save_metrics_to_json(scheduler.solution_metrics)
    
    # Print summary statistics
    print("\nSimulation Complete! Summary Statistics:")
    print(f"Total vehicles processed: {len(vehicles)}")
    print(f"Successfully scheduled: {len(vehicles) - scheduler.solution_metrics['unfulfilled_requests']} vehicles")
    print(f"Social welfare: {scheduler.solution_metrics['social_welfare']:.2f}")
    print(f"User satisfaction: {scheduler.solution_metrics['user_satisfaction']:.2f}")
    print(f"Jain's fairness index: {scheduler.solution_metrics['fairness_index']:.4f}")
    print(f"Total revenue: ${scheduler.solution_metrics['revenue']:.2f}")
    print(f"Total carbon emissions: {scheduler.solution_metrics['carbon_emissions']/1000:.2f} kg CO2")
    print(f"Runtime: {scheduler.solution_metrics['runtime']:.2f} seconds")
    
    print(f"\nResults saved to {output_dir}/")
    
    return scheduler, vehicles, stations, grid_model


if __name__ == "__main__":
    # Run simulation with different vehicle quantities for comparison
    scenario_sizes = [50, 100, 250]
    
    results = {}
    
    for size in scenario_sizes:
        print(f"\nRunning scenario with {size} vehicles...")
        scheduler, vehicles, stations, grid_model = run_simulation(num_vehicles=size)
        
        # Store results for comparison
        results[size] = scheduler.solution_metrics.copy()
    
    # Create comparative visualization
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot metrics across different scenario sizes
    metrics_to_plot = [
        ('user_satisfaction', 'User Satisfaction', axs[0, 0]),
        ('social_welfare', 'Social Welfare', axs[0, 1]),
        ('fairness_index', 'Fairness Index', axs[1, 0]),
        ('runtime', 'Runtime (s)', axs[1, 1])
    ]
    
    for metric_name, title, ax in metrics_to_plot:
        sizes = list(results.keys())
        values = [results[size][metric_name] for size in sizes]
        
        ax.plot(sizes, values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Vehicles')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Problem Size')
        
        # Set x-ticks to match scenario sizes
        ax.set_xticks(sizes)
        
        # Add value labels
        for i, value in enumerate(values):
            ax.annotate(f'{value:.2f}', (sizes[i], value), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparative_analysis.png", dpi=300)
    plt.close()
    
    # Save comparative metrics to JSON
    with open(f"{output_dir}/comparative_metrics.json", 'w') as f:
        json.dump({str(k): {m: round(v[m], 4) if isinstance(v[m], float) else v[m] 
                         for m in v} 
                for k, v in results.items()}, f, indent=4)
    
    print(f"\nComparative analysis saved to {output_dir}/comparative_analysis.png")
    
    # Run one larger simulation for final dataset
    print("\nRunning final large-scale simulation...")
    scheduler, vehicles, stations, grid_model = run_simulation(num_vehicles=300)