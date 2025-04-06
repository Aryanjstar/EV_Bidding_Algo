import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import math
from scipy.stats import norm
import networkx as nx
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Create output directory for saving results
output_dir = "ev_charging_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Enhanced data generation with realistic patterns
class EVChargingDataGenerator:
    def __init__(self, num_users=200, num_stations=10, time_slots=24, days=7, seed=42):
        """
        Generate realistic EV charging request data
        
        Parameters:
        - num_users: Number of EV users
        - num_stations: Number of charging stations
        - time_slots: Number of time slots per day (default: 24 hours)
        - days: Number of days to simulate
        - seed: Random seed for reproducibility
        """
        self.num_users = num_users
        self.num_stations = num_stations
        self.time_slots = time_slots
        self.days = days
        self.total_slots = time_slots * days
        np.random.seed(seed)
        random.seed(seed)
        
        # Station capacity: different stations have different capacity
        self.station_capacity = np.random.randint(3, 8, size=num_stations)
        
        # Location coordinates for stations (2D grid)
        self.station_locations = np.random.uniform(0, 100, size=(num_stations, 2))
        
        # User home locations
        self.user_locations = np.random.uniform(0, 100, size=(num_users, 2))
        
        # User types (commuter, random, night charger, etc.)
        self.user_types = np.random.choice(['commuter', 'random', 'night_charger', 'work_charger', 'weekend_user'], 
                                         size=num_users, 
                                         p=[0.4, 0.2, 0.15, 0.15, 0.1])
        
        # EV battery capacity and current charge level (kWh)
        self.battery_capacity = np.random.uniform(40, 100, size=num_users)  # From compact to large EVs
        
        # User utility functions - different users value electricity differently
        # This determines price sensitivity
        self.price_sensitivity = np.random.uniform(0.5, 2.0, size=num_users)
        
        # Weekly patterns (1: weekday, 2: weekend)
        self.day_types = np.array([1, 1, 1, 1, 1, 2, 2] * (days // 7 + 1))[:days]
        
        # Time of day price factors (electricity pricing)
        self.base_price = 0.15  # base price per kWh
        self.time_price_factors = self._generate_time_price_factors()
        
        # Station preference matrix: users prefer certain stations over others (e.g., closer to work/home)
        self.calculate_station_preferences()
        
    def _generate_time_price_factors(self):
        """Generate price factors for different times of day (peak/off-peak)"""
        factors = np.ones((self.days, self.time_slots))
        
        # Morning peak (7-9 AM)
        factors[:, 7:10] = 1.5
        
        # Evening peak (5-8 PM)
        factors[:, 17:21] = 1.8
        
        # Night off-peak (11 PM - 6 AM)
        factors[:, 23:] = 0.7
        factors[:, :6] = 0.7
        
        # Weekend has lower peak factors
        weekend_idx = np.where(self.day_types == 2)[0]
        factors[weekend_idx, :] *= 0.85
        
        return factors
    
    def calculate_station_preferences(self):
        """Calculate user preference for each station based on distance and randomized preference"""
        # Calculate distances between each user and station
        distances = np.zeros((self.num_users, self.num_stations))
        
        for u in range(self.num_users):
            for s in range(self.num_stations):
                distances[u, s] = np.sqrt(
                    (self.user_locations[u, 0] - self.station_locations[s, 0])**2 + 
                    (self.user_locations[u, 1] - self.station_locations[s, 1])**2
                )
        
        # Normalize distances (smaller is better)
        max_distance = np.max(distances)
        normalized_distances = 1 - (distances / max_distance)
        
        # Add random preference factor
        random_preference = np.random.uniform(0.7, 1.3, size=(self.num_users, self.num_stations))
        
        # Final preference is a combination of distance and random preference
        self.station_preferences = normalized_distances * random_preference
        
        # Normalize to [0, 1] range
        for u in range(self.num_users):
            self.station_preferences[u] = self.station_preferences[u] / np.sum(self.station_preferences[u])
    
    def generate_charge_requests(self):
    #"""Generate charging requests for all users across the time period"""
        requests = []
    
        for u in range(self.num_users):
            user_type = self.user_types[u]
            num_requests = self._get_num_requests_for_user(user_type)

            for _ in range(num_requests):
                # Choose day and hour based on user type
                day, hour = self._get_charge_time(user_type)

                if day >= self.days:
                    continue
                    
                # Charge needed - normally distributed around their battery capacity
                charge_needed = np.random.normal(0.6 * self.battery_capacity[u], 0.2 * self.battery_capacity[u])
                charge_needed = max(5, min(charge_needed, self.battery_capacity[u]))

                # Duration needed (hours) - depends on charging speed
                charging_speed = np.random.uniform(7, 22)  # kW
                duration_needed = math.ceil(charge_needed / charging_speed)
                duration_needed = max(1, min(duration_needed, 8))  # Cap at 8 hours

                # Flexibility (how many hours they're willing to stay beyond minimum required)
                flexibility = self._get_flexibility(user_type)

                # Max duration
                max_duration = min(duration_needed + flexibility, 12)  # Cap at 12 hours

                # Deadline
                deadline = hour + max_duration
                if deadline >= self.time_slots:
                    deadline = self.time_slots

                # Price willing to pay (per kWh) - based on time of day and user's price sensitivity
                time_idx = day * self.time_slots + hour
                day_idx = time_idx // self.time_slots
                hour_idx = time_idx % self.time_slots

                base_price_factor = self.time_price_factors[day_idx % self.days, hour_idx]
                price_willing = self.base_price * base_price_factor * (1 / self.price_sensitivity[u])

                # Station preferences for this user
                preferred_stations = np.random.choice(
                    self.num_stations, 
                    size=min(3, self.num_stations), 
                    p=self.station_preferences[u], 
                    replace=False
                )

                # Utility function parameters (quadratic function)
                # Utility = a - b * (t - preferred_time)^2
                a = np.random.uniform(50, 150)  # base utility
                b = np.random.uniform(1, 5)     # time preference strength
                preferred_time = hour           # preferred start time

                # Create request
                request = {
                    'user_id': u,
                    'day': day,
                    'arrival_hour': hour,
                    'charge_needed': charge_needed,
                    'min_duration': duration_needed,
                    'max_duration': max_duration,
                    'deadline': deadline,
                    'price_willing': price_willing,
                    'preferred_stations': preferred_stations.tolist(),
                    'utility_a': a,
                    'utility_b': b,
                    'preferred_time': preferred_time,
                    'user_type': user_type
                }

                requests.append(request)

        # Convert to DataFrame
        df = pd.DataFrame(requests)

        # Add absolute time slots
        df['earliest_slot'] = df['day'] * self.time_slots + df['arrival_hour']
        df['latest_slot'] = df['earliest_slot'] + df['max_duration']
        df['ideal_slot'] = df['day'] * self.time_slots + df['preferred_time']

        # Calculate utility for different time slots and store as dictionary
        time_utilities_list = []
        for _, row in df.iterrows():
            utilities = {}
            earliest = int(row['earliest_slot'])
            latest = int(min(row['latest_slot'], self.total_slots))
            ideal = int(row['ideal_slot'])

            for t in range(earliest, latest):
                time_diff = abs(t - ideal)
                utility = row['utility_a'] - row['utility_b'] * (time_diff ** 2)
                utilities[t] = max(0, utility)
            time_utilities_list.append(utilities)

        # Add time utilities as a new column
        df['time_utilities'] = time_utilities_list

        return df
    
    def _get_num_requests_for_user(self, user_type):
        """Determine how many charging requests a user makes based on their type"""
        if user_type == 'commuter':
            return np.random.randint(3, 6)  # Regular commuter charges frequently
        elif user_type == 'night_charger':
            return np.random.randint(2, 4)  # Night chargers charge less frequently
        elif user_type == 'work_charger':
            return np.random.randint(3, 5)  # Work chargers charge at work regularly
        elif user_type == 'weekend_user':
            return np.random.randint(1, 3)  # Weekend users charge less often
        else:  # random user
            return np.random.randint(1, 4)
    
    def _get_charge_time(self, user_type):
        """Determine when a user wants to charge based on their type"""
        day = np.random.randint(0, self.days)
        
        if user_type == 'commuter':
            if self.day_types[day % len(self.day_types)] == 1:  # Weekday
                hour = np.random.choice([
                    np.random.randint(7, 10),    # Morning (work arrival)
                    np.random.randint(17, 20)    # Evening (returning home)
                ])
            else:  # Weekend
                hour = np.random.randint(10, 20)  # More random on weekends
        elif user_type == 'night_charger':
            hour = np.random.randint(19, 24)  # Evening or night
        elif user_type == 'work_charger':
            if self.day_types[day % len(self.day_types)] == 1:  # Weekday
                hour = np.random.randint(8, 14)  # During work hours
            else:  # Weekend
                hour = np.random.randint(10, 20)  # More random on weekends
        elif user_type == 'weekend_user':
            if self.day_types[day % len(self.day_types)] == 2:  # Weekend
                hour = np.random.randint(9, 21)  # Any time during weekend day
            else:
                # Try to find a weekend day
                weekend_days = np.where(self.day_types == 2)[0]
                if len(weekend_days) > 0:
                    day = np.random.choice(weekend_days)
                hour = np.random.randint(9, 21)
        else:  # random user
            hour = np.random.randint(7, 22)  # Any time during waking hours
            
        return day, hour
    
    def _get_flexibility(self, user_type):
        """Determine how flexible a user is with charging time based on their type"""
        if user_type == 'commuter':
            return np.random.randint(1, 4)  # Moderate flexibility
        elif user_type == 'night_charger':
            return np.random.randint(5, 9)  # High flexibility (overnight)
        elif user_type == 'work_charger':
            return np.random.randint(2, 6)  # Moderate to high flexibility
        elif user_type == 'weekend_user':
            return np.random.randint(3, 7)  # High flexibility
        else:  # random user
            return np.random.randint(1, 5)  # Variable flexibility


# Enhanced Iterative Bidding Algorithm
class EnhancedIterativeBidding:
    def __init__(self, requests_df, num_stations, station_capacity, time_slots, 
                 max_rounds=20, min_price_increment=0.5, learning_rate=0.8, 
                 congestion_factor=1.2, dynamic_price_adjustment=True):
        """
        Enhanced Iterative Bidding for EV Charging Scheduling
        
        Parameters:
        - requests_df: DataFrame with charging requests
        - num_stations: Number of charging stations
        - station_capacity: Array of capacity for each station
        - time_slots: Total number of time slots
        - max_rounds: Maximum bidding rounds
        - min_price_increment: Minimum price increment in congested slots
        - learning_rate: Price adjustment learning rate
        - congestion_factor: How aggressively prices increase with congestion
        - dynamic_price_adjustment: Whether to use dynamic price adjustment
        """
        self.requests_df = requests_df.copy()
        self.num_stations = num_stations
        self.station_capacity = station_capacity
        self.time_slots = time_slots
        self.max_rounds = max_rounds
        self.min_price_increment = min_price_increment
        self.learning_rate = learning_rate
        self.congestion_factor = congestion_factor
        self.dynamic_price_adjustment = dynamic_price_adjustment
        
        # Initialize prices for each (station, time slot) pair
        self.prices = np.zeros((num_stations, time_slots))
        
        # Initialize base demand tracker for dynamic pricing
        self.base_demand = np.zeros((num_stations, time_slots))
        self.demand_history = []
        
        # Metrics to track
        self.social_welfare_history = []
        self.revenue_history = []
        self.allocation_rate_history = []
        self.price_evolution = np.zeros((max_rounds, num_stations, time_slots))
        self.congestion_history = []
        self.preference_revelation = []  # Measure of how much users reveal preferences
        
        # Enhanced tracking
        self.user_satisfaction = []
        self.station_utilization = []
        self.price_volatility = []
        self.unfairness_metric = []
        
    def calculate_utility(self, request, station_id, start_time):
        """Calculate utility for a user given station and start time"""
        # Basic utility from user's time preference
        time_utilities = request['time_utilities']
        if isinstance(time_utilities, dict) and start_time in time_utilities:
            time_utility = time_utilities[start_time]
        else:
            return -float('inf')  # Invalid time slot
            
        # Cost of charging
        duration = request['min_duration']
        cost = sum(self.prices[station_id, start_time:start_time+duration])
        
        # Distance penalty (based on station preference)
        if station_id in request['preferred_stations']:
            station_pref_idx = request['preferred_stations'].index(station_id)
            station_penalty = station_pref_idx * 2  # Penalty increases for less preferred stations
        else:
            station_penalty = 10  # High penalty for non-preferred stations
            
        # The final utility is time utility minus cost and penalties
        utility = time_utility - cost - station_penalty
        
        return utility
    
    def find_best_allocation(self, request):
        """Find best (station, time) allocation for a user given current prices"""
        best_utility = -float('inf')
        best_allocation = None
        
        # Only consider preferred stations
        for station_id in request['preferred_stations']:
            # Check each possible start time
            for start_time in range(int(request['earliest_slot']), int(min(request['latest_slot'], self.time_slots))):
                # Ensure the charging duration fits within the time window
                if start_time + request['min_duration'] <= min(request['latest_slot'], self.time_slots):
                    utility = self.calculate_utility(request, station_id, start_time)
                    
                    if utility > best_utility:
                        best_utility = utility
                        best_allocation = (station_id, start_time, start_time + request['min_duration'], utility)
        
        return best_allocation
    
    def update_prices(self, demand, round_num):
        """
        Enhanced price updating mechanism with adaptive adjustments
        """
        price_changes = np.zeros_like(self.prices)
        congestion = np.zeros_like(self.prices)
        
        # Calculate congestion levels
        for s in range(self.num_stations):
            for t in range(self.time_slots):
                congestion[s, t] = max(0, demand[s, t] - self.station_capacity[s])
        
        # Calculate adaptive price increments
        for s in range(self.num_stations):
            for t in range(self.time_slots):
                if demand[s, t] > self.station_capacity[s]:
                    # More aggressive price increase for heavily congested spots
                    congestion_ratio = demand[s, t] / self.station_capacity[s]
                    
                    # Exponential increase based on congestion severity
                    if self.dynamic_price_adjustment:
                        # Use demand history to adjust price increment
                        if round_num > 2 and s < len(self.demand_history[-1]) and t < len(self.demand_history[-1][s]):
                            prev_demand = self.demand_history[-1][s][t]
                            demand_change = demand[s, t] - prev_demand
                            
                            # If demand isn't decreasing, increase price more aggressively
                            if demand_change >= 0:
                                price_increment = self.min_price_increment * (congestion_ratio ** self.congestion_factor)
                            else:
                                # Smaller increment if demand is already decreasing
                                adjustment_factor = max(0.1, abs(demand_change) / self.station_capacity[s])
                                price_increment = self.min_price_increment * adjustment_factor
                        else:
                            price_increment = self.min_price_increment * (congestion_ratio ** self.congestion_factor)
                    else:
                        price_increment = self.min_price_increment * congestion_ratio
                    
                    # Adaptive learning rate: reduce as rounds progress
                    effective_lr = self.learning_rate * (1 - 0.5 * round_num / self.max_rounds)
                    
                    price_changes[s, t] = effective_lr * price_increment
                
                elif demand[s, t] < self.station_capacity[s] * 0.5 and self.prices[s, t] > 0:
                    # Price decrease for underutilized spots
                    utilization_ratio = demand[s, t] / self.station_capacity[s]
                    price_decrement = self.min_price_increment * (1 - utilization_ratio)
                    
                    # Prevent prices from going negative
                    price_changes[s, t] = -min(self.prices[s, t] * 0.5, price_decrement * self.learning_rate)
        
        # Update prices
        self.prices += price_changes
        
        # Ensure prices are non-negative
        self.prices = np.maximum(0, self.prices)
        
        # Store price volatility metric
        self.price_volatility.append(np.mean(np.abs(price_changes)))
        
        return congestion
        
    def run_iterative_bidding(self):
        """Run the iterative bidding process"""
        num_requests = len(self.requests_df)
        
        for round_num in tqdm(range(self.max_rounds), desc="Bidding Rounds"):
            # Current demand for each (station, time) pair
            demand = np.zeros((self.num_stations, self.time_slots))
            
            # Current allocations
            allocations = []
            
            # Total utility in this round
            total_utility = 0
            total_revenue = 0
            
            # Track user preference revelation
            revealed_preferences = 0
            
            # For calculating user satisfaction
            user_utilities = []
            
            # Bid collection phase
            for idx, request in self.requests_df.iterrows():
                # Find best allocation given current prices
                best_allocation = self.find_best_allocation(request)
                
                if best_allocation:
                    station_id, start_time, end_time, utility = best_allocation
                    
                    if utility > 0:  # Only allocate if utility is positive
                        # Add to demand for each time slot
                        for t in range(start_time, end_time):
                            if t < self.time_slots:
                                demand[station_id, t] += 1
                        
                        # Save allocation
                        allocations.append({
                            'user_id': request['user_id'],
                            'station_id': station_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'utility': utility,
                            'charge_needed': request['charge_needed']
                        })
                        
                        # Add to total utility and metrics
                        total_utility += utility
                        total_revenue += sum(self.prices[station_id, start_time:end_time])
                        user_utilities.append(utility)
                        
                        # Count preference revelation
                        if 'time_utilities' in request and isinstance(request['time_utilities'], dict) and len(request['time_utilities']) > 0:
                            revealed_preferences += len(request['time_utilities'])
            
            # Save demand history
            self.demand_history.append(demand.copy())
            
            # Update prices based on demand
            congestion = self.update_prices(demand, round_num)
            self.price_evolution[round_num] = self.prices.copy()
            
            # Calculate metrics
            allocation_rate = len(allocations) / num_requests if num_requests > 0 else 0
            self.social_welfare_history.append(total_utility)
            self.revenue_history.append(total_revenue)
            self.allocation_rate_history.append(allocation_rate)
            self.congestion_history.append(np.sum(congestion))
            self.preference_revelation.append(revealed_preferences / num_requests if num_requests > 0 else 0)
            
            # Calculate enhanced metrics
            self.user_satisfaction.append(np.mean(user_utilities) if user_utilities else 0)
            
            # Station utilization (percentage of capacity used)
            station_util = []
            for s in range(self.num_stations):
                utilization = np.mean(demand[s]) / self.station_capacity[s]
                station_util.append(utilization)
            self.station_utilization.append(np.mean(station_util))
            
            # Unfairness metric (standard deviation of utilities across users)
            self.unfairness_metric.append(np.std(user_utilities) if len(user_utilities) > 1 else 0)
            
            # Check for convergence
            if round_num > 1:
                welfare_change = abs(self.social_welfare_history[-1] - self.social_welfare_history[-2])
                if welfare_change < 0.1 and allocation_rate > 0.9:
                    print(f"Converged at round {round_num+1} with welfare change {welfare_change:.4f}")
                    break
        
        # Return the final allocation and metrics
        return {
            'final_allocations': allocations,
            'final_prices': self.prices,
            'social_welfare': self.social_welfare_history,
            'revenue': self.revenue_history,
            'allocation_rate': self.allocation_rate_history,
            'congestion': self.congestion_history,
            'preference_revelation': self.preference_revelation,
            'user_satisfaction': self.user_satisfaction,
            'station_utilization': self.station_utilization,
            'price_volatility': self.price_volatility,
            'unfairness': self.unfairness_metric,
            'price_evolution': self.price_evolution
        }


# Enhanced visualization functions
def plot_social_welfare_comparison(results_dict, title="Social Welfare Comparison"):
    """Plot social welfare comparison between different algorithm variants"""
    plt.figure(figsize=(12, 7))
    
    for label, result in results_dict.items():
        plt.plot(result['social_welfare'], marker='o', label=label)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Bidding Round', fontsize=12)
    plt.ylabel('Social Welfare', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/social_welfare_comparison.png", dpi=300)
    plt.close()

def plot_metrics_evolution(results, title="Algorithm Performance Metrics"):
    """Plot multiple metrics evolution over bidding rounds"""
    metrics = ['social_welfare', 'revenue', 'allocation_rate', 
              'congestion', 'user_satisfaction', 'price_volatility']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Define custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Normalize metrics for better visualization
    normalized_metrics = {}
    for i, metric in enumerate(metrics):
        values = results[metric]
        if max(values) - min(values) > 0:
            normalized_metrics[metric] = [(v - min(values)) / (max(values) - min(values)) for v in values]
        else:
            normalized_metrics[metric] = [0.5 for _ in values]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = range(1, len(results[metric]) + 1)
        
        # Create smooth curve using cubic interpolation
        ax.plot(x, results[metric], marker='o', color=colors[i], alpha=0.7, label=f'Raw {metric}')
        
        # Add trend line
        z = np.polyfit(x, results[metric], 3)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(x), max(x), 100)
        ax.plot(x_smooth, p(x_smooth), '--', color=colors[i], linewidth=2, 
                label=f'{metric} trend')
        
        # Format plot
        metric_name = ' '.join(w.capitalize() for w in metric.split('_'))
        ax.set_title(f'{metric_name} Evolution', fontsize=12)
        ax.set_xlabel('Bidding Round', fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_evolution.png", dpi=300)
    plt.close()

def plot_price_heatmap(results, time_slots_per_day=24, title="Price Evolution Heatmap"):
    """Plot price evolution as heatmap for a selected station"""
    # Select a station with interesting price evolution
    station_id = np.argmax(np.std(results['price_evolution'][:, :, :], axis=(0, 2)))
    
    # Get price data for this station
    price_data = results['price_evolution'][:, station_id, :]
    
    # Reshape for daily view if multiple days
    days = time_slots_per_day
    plt.figure(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(price_data, cmap='YlOrRd', 
                xticklabels=[f"{h}:00" for h in range(0, 24, 2)], 
                yticklabels=[f"Round {r+1}" for r in range(price_data.shape[0])],
                cbar_kws={'label': 'Price'})
    
    plt.title(f"{title} - Station {station_id+1}", fontsize=14)
    plt.ylabel('Bidding Round', fontsize=12)
    plt.xlabel('Time of Day', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_heatmap_station_{station_id+1}.png", dpi=300)
    plt.close()

def plot_station_congestion_network(results, station_locations, final_allocations, num_stations):
    """Plot stations as a network with congestion visualized"""
    # Create graph
    G = nx.Graph()
    
    # Calculate average congestion for each station
    station_congestion = np.zeros(num_stations)
    for alloc in final_allocations:
        station_id = alloc['station_id']
        station_congestion[station_id] += 1
    
    # Normalize congestion values
    max_congestion = max(station_congestion) if max(station_congestion) > 0 else 1
    normalized_congestion = station_congestion / max_congestion
    
    # Add nodes
    for i in range(num_stations):
        G.add_node(i, pos=station_locations[i], congestion=normalized_congestion[i])
    
    # Add edges between nearby stations
    for i in range(num_stations):
        for j in range(i+1, num_stations):
            dist = np.sqrt(np.sum((station_locations[i] - station_locations[j])**2))
            if dist < 30:  # Add edge if stations are within certain distance
                G.add_edge(i, j, weight=1/dist)
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Color map based on congestion
    node_colors = [G.nodes[i]['congestion'] for i in G.nodes]
    
    # Node sizes based on congestion
    node_sizes = [2000 * (0.3 + G.nodes[i]['congestion']) for i in G.nodes]
    
    # Draw nodes with congestion-based color and size
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=node_sizes, alpha=0.8,
                                  cmap=plt.cm.YlOrRd)
    
    # Draw edges
    edges = nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    labels = {i: f"S{i+1}" for i in range(num_stations)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    # Add colorbar
    plt.colorbar(nodes, ax=plt.gca(), label='Normalized Congestion')
    
    plt.title('Charging Station Network with Congestion Visualization', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/station_congestion_network.png", dpi=300)
    plt.close()

def plot_user_station_allocation(results, user_locations, station_locations, num_users, num_stations):
    """Plot user to station allocations as a bipartite graph"""
    plt.figure(figsize=(14, 10))
    
    # Plot users
    plt.scatter(user_locations[:, 0], user_locations[:, 1], 
                c='blue', marker='o', s=50, alpha=0.6, label='EV Users')
    
    # Plot stations
    plt.scatter(station_locations[:, 0], station_locations[:, 1], 
                c='red', marker='s', s=200, alpha=0.8, label='Charging Stations')
    
    # Plot allocations
    allocations = results['final_allocations']
    user_allocated = set()
    
    for alloc in allocations:
        user_id = alloc['user_id']
        station_id = alloc['station_id']
        
        if user_id < len(user_locations) and station_id < len(station_locations):
            user_pos = user_locations[user_id]
            station_pos = station_locations[station_id]
            
            # Draw line from user to allocated station
            plt.plot([user_pos[0], station_pos[0]], [user_pos[1], station_pos[1]], 
                     'g-', alpha=0.2)
            
            user_allocated.add(user_id)
    
    # Highlight unallocated users
    unallocated_users = set(range(num_users)) - user_allocated
    if unallocated_users:
        unallocated_pos = np.array([user_locations[i] for i in unallocated_users])
        plt.scatter(unallocated_pos[:, 0], unallocated_pos[:, 1], 
                    c='gray', marker='x', s=100, alpha=0.7, label='Unallocated Users')
    
    # Label stations
    for i, pos in enumerate(station_locations):
        plt.annotate(f'S{i+1}', (pos[0], pos[1]), fontsize=12, ha='center', va='center')
    
    plt.title('User-Station Allocation Map', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/user_station_allocation.png", dpi=300)
    plt.close()

def plot_time_slot_allocation_heatmap(results, num_stations, time_slots, time_slots_per_day=24):
    """Create a heatmap of allocations across stations and time slots"""
    # Initialize allocation matrix
    allocation_matrix = np.zeros((num_stations, time_slots))
    
    # Fill matrix with allocations
    for alloc in results['final_allocations']:
        station_id = alloc['station_id']
        start = alloc['start_time']
        end = alloc['end_time']
        
        # Mark all time slots used by this allocation
        for t in range(start, min(end, time_slots)):
            if station_id < num_stations and t < time_slots:
                allocation_matrix[station_id, t] += 1
    
    # Plot heatmap
    plt.figure(figsize=(15, 8))
    
    # Days on x-axis, hours within day on y-axis
    days = time_slots // time_slots_per_day
    station_data_reshaped = allocation_matrix.reshape(num_stations, days, time_slots_per_day)
    
    # Plot one heatmap per station
    fig, axes = plt.subplots(num_stations, 1, figsize=(16, 3*num_stations), sharex=True)
    
    if num_stations == 1:
        axes = [axes]
    
    for s in range(num_stations):
        # Create 2D heatmap (days x hours)
        station_data = station_data_reshaped[s]
        
        # Plot
        im = axes[s].imshow(station_data.T, aspect='auto', cmap='YlGnBu', 
                         interpolation='nearest', vmin=0)
        
        # Format
        axes[s].set_title(f'Station {s+1} Allocation', fontsize=12)
        axes[s].set_ylabel('Hour of Day', fontsize=10)
        axes[s].set_yticks(range(0, 24, 3))
        axes[s].set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
        
        if s == num_stations - 1:  # Only add x-label to bottom subplot
            axes[s].set_xlabel('Day', fontsize=10)
        
        axes[s].set_xticks(range(days))
        axes[s].set_xticklabels([f'Day {d+1}' for d in range(days)])
        
        # Add colorbar
        plt.colorbar(im, ax=axes[s], label='Number of EVs', shrink=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_slot_allocation_heatmap.png", dpi=300)
    plt.close()

def plot_price_evolution_3d(results, station_id=0, title="Price Evolution in 3D"):
    """Create a 3D surface plot of price evolution for a single station"""
    # Get price data for this station
    price_data = results['price_evolution'][:, station_id, :]
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create X, Y coordinates
    X, Y = np.meshgrid(range(price_data.shape[1]), range(price_data.shape[0]))
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, price_data, cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Time Slot', fontsize=12)
    ax.set_ylabel('Bidding Round', fontsize=12)
    ax.set_zlabel('Price', fontsize=12)
    ax.set_title(f'{title} - Station {station_id+1}', fontsize=14)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Price')
    
    # Set better viewing angle
    ax.view_init(elev=35, azim=-65)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_evolution_3d_station_{station_id+1}.png", dpi=300)
    plt.close()

def plot_user_type_satisfaction(results, user_types):
    """Plot satisfaction levels by user type"""
    # Extract user types and utilities from final allocation
    user_types_list = []
    utilities = []
    
    for alloc in results['final_allocations']:
        user_id = alloc['user_id']
        user_type = user_types[user_id]
        utility = alloc['utility']
        
        user_types_list.append(user_type)
        utilities.append(utility)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame({'user_type': user_types_list, 'utility': utilities})
    
    # Group by user type and calculate statistics
    type_stats = df.groupby('user_type').agg(
        mean_utility=('utility', 'mean'),
        median_utility=('utility', 'median'),
        min_utility=('utility', 'min'),
        max_utility=('utility', 'max'),
        count=('utility', 'count')
    ).reset_index()
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 8))
    
    # Bar positions
    x = np.arange(len(type_stats))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, type_stats['mean_utility'], width, label='Mean Utility', color='skyblue')
    plt.bar(x + width/2, type_stats['median_utility'], width, label='Median Utility', color='salmon')
    
    # Add error bars for min/max
    plt.errorbar(x - width/2, type_stats['mean_utility'], 
                yerr=[type_stats['mean_utility'] - type_stats['min_utility'], 
                      type_stats['max_utility'] - type_stats['mean_utility']],
                fmt='none', ecolor='black', capsize=5)
    
    # Add counts as text
    for i, row in enumerate(type_stats.itertuples()):
        plt.text(i, row.max_utility + 2, f'n={row.count}', ha='center')
    
    # Add labels and formatting
    plt.xlabel('User Type', fontsize=12)
    plt.ylabel('Utility', fontsize=12)
    plt.title('Satisfaction by User Type', fontsize=14)
    plt.xticks(x, type_stats['user_type'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/user_type_satisfaction.png", dpi=300)
    plt.close()

def plot_radar_comparison(results_dict):
    """Create a radar chart comparing different algorithms"""
    # Metrics to compare
    metrics = ['final_social_welfare', 'final_revenue', 'final_allocation_rate', 
              'final_user_satisfaction', 'final_station_utilization', 'fairness']
    
    # Calculate fairness as inverse of unfairness
    for alg, results in results_dict.items():
        results['fairness'] = 1 / (1 + results['unfairness'][-1])
        results['final_social_welfare'] = results['social_welfare'][-1]
        results['final_revenue'] = results['revenue'][-1]
        results['final_allocation_rate'] = results['allocation_rate'][-1]
        results['final_user_satisfaction'] = results['user_satisfaction'][-1]
        results['final_station_utilization'] = results['station_utilization'][-1]
    
    # Normalize metrics
    normalized_results = {}
    for metric in metrics:
        metric_values = [results[metric] for results in results_dict.values()]
        min_val = min(metric_values)
        max_val = max(metric_values)
        
        if max_val - min_val > 0:
            normalized_results[metric] = {alg: (results[metric] - min_val) / (max_val - min_val) 
                                         for alg, results in results_dict.items()}
        else:
            normalized_results[metric] = {alg: 0.5 for alg in results_dict.keys()}
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot each algorithm
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for i, (alg, results) in enumerate(results_dict.items()):
        values = [normalized_results[metric][alg] for metric in metrics]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=alg, alpha=0.8)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # Add labels
    metric_labels = [m.replace('_', ' ').replace('final ', '').title() for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    
    # Add grid and legend
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Algorithm Performance Comparison', fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/radar_comparison.png", dpi=300)
    plt.close()

def save_metrics_to_csv(results_dict):
    """Save all metrics to CSV files for further analysis"""
    # Save overall metrics
    metrics_df = pd.DataFrame()
    
    for alg, results in results_dict.items():
        alg_df = pd.DataFrame({
            'algorithm': alg,
            'round': range(len(results['social_welfare'])),
            'social_welfare': results['social_welfare'],
            'revenue': results['revenue'],
            'allocation_rate': results['allocation_rate'],
            'congestion': results['congestion'],
            'user_satisfaction': results['user_satisfaction'],
            'station_utilization': results['station_utilization'],
            'price_volatility': results['price_volatility'],
            'unfairness': results['unfairness']
        })
        metrics_df = pd.concat([metrics_df, alg_df], ignore_index=True)
    
    metrics_df.to_csv(f"{output_dir}/algorithm_metrics.csv", index=False)
    
    # Save final allocation details
    for alg, results in results_dict.items():
        alloc_df = pd.DataFrame(results['final_allocations'])
        alloc_df['algorithm'] = alg
        alloc_df.to_csv(f"{output_dir}/{alg}_allocations.csv", index=False)

# New algorithm variants for comparison

class PriorityBasedBidding(EnhancedIterativeBidding):
    """Modified bidding algorithm that prioritizes certain user types"""
    
    def __init__(self, requests_df, num_stations, station_capacity, time_slots, 
                 max_rounds=20, min_price_increment=0.5, learning_rate=0.8, 
                 congestion_factor=1.2, user_type_priorities=None):
        
        super().__init__(requests_df, num_stations, station_capacity, time_slots, 
                        max_rounds, min_price_increment, learning_rate, congestion_factor)
        
        # User type priorities (e.g., {'commuter': 1.2, 'night_charger': 0.9})
        self.user_type_priorities = user_type_priorities or {
            'commuter': 1.3,       # Give priority to commuters
            'work_charger': 1.2,   # Work chargers also get priority
            'night_charger': 0.8,  # Night chargers get discounts
            'weekend_user': 0.9,   # Weekend users get small discounts
            'random': 1.0          # No adjustment for random users
        }
    
    def calculate_utility(self, request, station_id, start_time):
        """Calculate utility with user type priority adjustments"""
        # Get base utility from parent class
        utility = super().calculate_utility(request, station_id, start_time)
        
        # Apply user type priority adjustment
        user_type = request['user_type']
        priority_factor = self.user_type_priorities.get(user_type, 1.0)
        
        # Adjust utility based on priority - higher priority users get utility boost
        adjusted_utility = utility * priority_factor
        
        return adjusted_utility

class TimeWindowConstrainedBidding(EnhancedIterativeBidding):
    """Modified bidding algorithm with stricter time window constraints"""
    
    def __init__(self, requests_df, num_stations, station_capacity, time_slots, 
                 max_rounds=20, min_price_increment=0.5, learning_rate=0.8, 
                 congestion_factor=1.2, peak_hour_penalty=2.0):
        
        super().__init__(requests_df, num_stations, station_capacity, time_slots, 
                        max_rounds, min_price_increment, learning_rate, congestion_factor)
        
        self.peak_hour_penalty = peak_hour_penalty
        
        # Define peak hours (e.g., morning rush 7-9 AM, evening rush 5-7 PM)
        self.peak_hours = []
        
        # Morning peak: 7-9 AM
        for day in range(time_slots // 24):
            self.peak_hours.extend(range(day*24 + 7, day*24 + 10))
        
        # Evening peak: 5-7 PM
        for day in range(time_slots // 24):
            self.peak_hours.extend(range(day*24 + 17, day*24 + 20))
    
    def calculate_utility(self, request, station_id, start_time):
        """Calculate utility with peak hour penalties"""
        # Get base utility
        utility = super().calculate_utility(request, station_id, start_time)
        
        # Check if charging period overlaps with peak hours
        duration = request['min_duration']
        charging_slots = range(start_time, start_time + duration)
        
        # Count overlapping peak hours
        peak_hour_count = sum(1 for slot in charging_slots if slot in self.peak_hours)
        
        # Apply penalty for peak hours
        peak_penalty = peak_hour_count * self.peak_hour_penalty
        
        # Adjust utility
        adjusted_utility = utility - peak_penalty
        
        return adjusted_utility

class DeadlineConstrainedBidding(EnhancedIterativeBidding):
    """Modified bidding algorithm with strict deadline constraints"""
    
    def find_best_allocation(self, request):
        """Find best allocation with deadline constraints"""
        best_utility = -float('inf')
        best_allocation = None
        
        # Deadline pressure factor - increases utility for earlier slots when close to deadline
        remaining_slots = request['latest_slot'] - request['earliest_slot']
        deadline_pressure = max(1.0, 1.5 * (1.0 / max(1, remaining_slots)))
        
        # Only consider preferred stations
        for station_id in request['preferred_stations']:
            # Check each possible start time
            for start_time in range(int(request['earliest_slot']), int(min(request['latest_slot'], self.time_slots))):
                # Ensure charging can complete before deadline
                if start_time + request['min_duration'] <= min(request['latest_slot'], self.time_slots):
                    # Base utility
                    utility = self.calculate_utility(request, station_id, start_time)
                    
                    # Time pressure bonus for scheduling earlier
                    time_pressure_bonus = (request['latest_slot'] - start_time) * deadline_pressure
                    
                    # Adjust utility
                    adjusted_utility = utility + time_pressure_bonus * 0.2  # Scale factor to avoid dominating
                    
                    if adjusted_utility > best_utility:
                        best_utility = adjusted_utility
                        # Store original utility, not the adjusted one with deadline pressure
                        best_allocation = (station_id, start_time, start_time + request['min_duration'], utility)
        
        return best_allocation

# Main execution function
def run_ev_charging_experiment(num_users=500, num_stations=15, days=14, 
                              time_slots_per_day=24, max_rounds=30):
    """Run comprehensive EV charging experiment with multiple algorithms"""
    print("Starting EV Charging Scheduling Experiment")
    start_time = time.time()
    
    # Generate data
    print("Generating realistic EV charging data...")
    data_gen = EVChargingDataGenerator(
        num_users=num_users, 
        num_stations=num_stations, 
        time_slots=time_slots_per_day,
        days=days,
        seed=42
    )
    requests_df = data_gen.generate_charge_requests()
    
    # Save generated data
    requests_df.to_csv(f"{output_dir}/charging_requests.csv", index=False)
    print(f"Generated {len(requests_df)} charging requests from {num_users} users.")
    
    # Define algorithms to compare
    algorithms = {
        'Standard Bidding': EnhancedIterativeBidding(
            requests_df, num_stations, data_gen.station_capacity, 
            days * time_slots_per_day, max_rounds=max_rounds
        ),
        'Priority Based': PriorityBasedBidding(
            requests_df, num_stations, data_gen.station_capacity, 
            days * time_slots_per_day, max_rounds=max_rounds
        ),
        'Time Window Constrained': TimeWindowConstrainedBidding(
            requests_df, num_stations, data_gen.station_capacity, 
            days * time_slots_per_day, max_rounds=max_rounds,
            peak_hour_penalty=2.5
        ),
        'Deadline Constrained': DeadlineConstrainedBidding(
            requests_df, num_stations, data_gen.station_capacity, 
            days * time_slots_per_day, max_rounds=max_rounds,
            learning_rate=0.9
        )
    }
    
    # Run all algorithms and collect results
    results_dict = {}
    
    for name, alg in algorithms.items():
        print(f"\nRunning {name} algorithm...")
        results = alg.run_iterative_bidding()
        results_dict[name] = results
        print(f"  Final welfare: {results['social_welfare'][-1]:.2f}")
        print(f"  Final revenue: {results['revenue'][-1]:.2f}")
        print(f"  Allocation rate: {results['allocation_rate'][-1]:.2%}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Social welfare comparison
    plot_social_welfare_comparison(results_dict)
    
    # 2. Performance metrics for each algorithm
    for name, results in results_dict.items():
        plot_metrics_evolution(results, title=f"{name} Performance Metrics")
    
    # 3. Price heatmaps
    for name, results in results_dict.items():
        plot_price_heatmap(results, time_slots_per_day, title=f"{name} - Price Evolution")
    
    # 4. Price 3D evolution (for selected algorithm and stations)
    for station_id in range(min(3, num_stations)):
        plot_price_evolution_3d(results_dict['Standard Bidding'], station_id)
    
    # 5. Station congestion network
    best_alg = max(results_dict.items(), key=lambda x: x[1]['social_welfare'][-1])[0]
    plot_station_congestion_network(
        results_dict[best_alg], 
        data_gen.station_locations, 
        results_dict[best_alg]['final_allocations'],
        num_stations
    )
    
    # 6. User-station allocation map
    plot_user_station_allocation(
        results_dict[best_alg],
        data_gen.user_locations,
        data_gen.station_locations,
        num_users,
        num_stations
    )
    
    # 7. Time slot allocation heatmap
    plot_time_slot_allocation_heatmap(
        results_dict[best_alg],
        num_stations,
        days * time_slots_per_day,
        time_slots_per_day
    )
    
    # 8. User type satisfaction
    plot_user_type_satisfaction(results_dict[best_alg], data_gen.user_types)
    
    # 9. Radar chart comparing all algorithms
    plot_radar_comparison(results_dict)
    
    # Save metrics to CSV
    save_metrics_to_csv(results_dict)
    
    elapsed_time = time.time() - start_time
    print(f"\nExperiment completed in {elapsed_time:.2f} seconds.")
    print(f"Results saved to {output_dir}/")
    
    return results_dict

# Run the experiment
if __name__ == "__main__":
    # Large experiment with many users and stations
    results = run_ev_charging_experiment(
        num_users=500,
        num_stations=15,
        days=14,
        max_rounds=30
    )
    
    print("\nExperiment results summary:")
    for alg, result in results.items():
        print(f"\n{alg}:")
        print(f"  - Final social welfare: {result['social_welfare'][-1]:.2f}")
        print(f"  - Final allocation rate: {result['allocation_rate'][-1]:.2%}")
        print(f"  - User satisfaction: {result['user_satisfaction'][-1]:.2f}")
        print(f"  - Station utilization: {result['station_utilization'][-1]:.2%}")