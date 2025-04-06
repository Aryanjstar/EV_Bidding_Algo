import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from typing import Dict, List, Tuple, Union
import json
import time
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

class EVChargingMarketSimulator:
    """
    Enhanced Electric Vehicle Charging Market Simulator with Realistic Market Dynamics
    
    Features:
    - Multiple user behavior profiles based on real-world charging patterns
    - Dynamic time-of-day and day-of-week pricing
    - Grid congestion modeling and demand response
    - Advanced bidding strategies based on game theory
    - Geographical distribution of charging stations
    - Renewable energy integration effects
    """
    
    def __init__(self, 
                 num_users: int = 10000, 
                 num_stations: int = 500,
                 max_rounds: int = 15,
                 time_window_days: int = 7,
                 renewable_integration: float = 0.3,
                 congestion_factor: float = 0.6,
                 base_price: float = 0.22):  # Average US electricity price per kWh
        
        self.num_users = num_users
        self.num_stations = num_stations
        self.max_rounds = max_rounds
        self.time_window_days = time_window_days
        self.renewable_integration = renewable_integration
        self.congestion_factor = congestion_factor
        self.base_price = base_price
        
        # Create output directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f'ev_market_simulation_{self.timestamp}'
        self.data_dir = f'{self.output_dir}/data'
        self.graph_dir = f'{self.output_dir}/graphs'
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        
        # Initialize simulation components
        self.users = None
        self.stations = None
        self.time_grid = None
        self.price_model = None
        self.scaler = StandardScaler()
        
        # Market metrics tracking
        self.market_history = {
            'price': [],
            'demand': [],
            'supply': [],
            'transactions': [],
            'unmet_demand': [],
            'revenue': [],
            'grid_load': [],
            'renewable_contribution': [],
            'price_volatility': [],
            'user_satisfaction': [],
            'satisfaction_rate': []
        }
        
        # Grid load patterns (24 hours × 7 days)
        self.grid_base_load = self._create_grid_load_pattern()
        
        # Generate simulation data
        self._initialize_simulation()
    
    def _create_grid_load_pattern(self) -> np.ndarray:
        """Create realistic grid load patterns based on time of day and day of week"""
        # 24 hours × 7 days grid base load pattern
        hourly_pattern = np.array([
            0.65, 0.60, 0.58, 0.55, 0.60, 0.70,  # 12am-6am
            0.85, 1.00, 1.10, 1.05, 1.00, 0.95,  # 6am-12pm
            0.90, 0.95, 1.00, 1.05, 1.15, 1.20,  # 12pm-6pm
            1.25, 1.20, 1.10, 1.00, 0.85, 0.70   # 6pm-12am
        ])
        
        # Day of week factors (weekday vs weekend)
        day_factors = np.array([
            1.05, 1.05, 1.05, 1.05, 1.05,  # Weekdays
            0.85, 0.80                     # Weekend
        ])
        
        # Combine into 7-day, 24-hour grid
        grid = np.outer(day_factors, hourly_pattern)
        return grid
    
    def _initialize_simulation(self):
        """Initialize all simulation components"""
        print("Initializing simulation environment...")
        
        # Create time grid spanning the specified window
        self.start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.time_grid = [self.start_time + timedelta(hours=h) 
                          for h in range(self.time_window_days * 24)]
        
        # Generate charging stations with geographical distribution
        self._generate_charging_stations()
        
        # Generate users with realistic profiles
        self._generate_users()
        
        # Train price prediction model
        self._train_price_model()
        
        # Save initial datasets
        self._save_datasets()
        
        print(f"Simulation initialized with {self.num_users} users and {self.num_stations} stations")
    
    def _generate_charging_stations(self):
        """Generate charging stations with geographical distribution and capacity variations"""
        print("Generating charging stations...")
        
        # Geographic regions with proportional distribution
        regions = {
            'urban_dense': {'proportion': 0.35, 'capacity_range': (8, 20)},
            'urban_sparse': {'proportion': 0.25, 'capacity_range': (4, 12)},
            'suburban': {'proportion': 0.25, 'capacity_range': (2, 8)},
            'rural': {'proportion': 0.10, 'capacity_range': (1, 4)},
            'highway': {'proportion': 0.05, 'capacity_range': (10, 30)}
        }
        
        stations = []
        id_counter = 0
        
        for region, params in regions.items():
            count = int(self.num_stations * params['proportion'])
            min_cap, max_cap = params['capacity_range']
            
            for _ in range(count):
                # Generate capacity (number of simultaneous charging ports)
                capacity = np.random.randint(min_cap, max_cap + 1)
                
                # Generate power levels based on region
                if region == 'highway':
                    # DC Fast charging
                    power_level = np.random.uniform(50, 150)
                elif region in ['urban_dense', 'urban_sparse']:
                    # Mix of Level 2 and DC Fast
                    power_level = np.random.choice([7, 11, 22, 50, 100], 
                                                 p=[0.2, 0.3, 0.3, 0.15, 0.05])
                else:
                    # Mostly Level 2
                    power_level = np.random.choice([7, 11, 22, 50], 
                                                 p=[0.4, 0.4, 0.15, 0.05])
                
                # Calculate operational costs based on power level and region
                base_op_cost = 0.02 + (power_level * 0.0008)  # Higher power = higher operational costs
                if region == 'urban_dense':
                    op_cost = base_op_cost * 1.3  # Higher rent/maintenance in dense urban areas
                elif region == 'rural':
                    op_cost = base_op_cost * 0.8  # Lower overhead in rural areas
                else:
                    op_cost = base_op_cost
                
                # Location factors for geographical distribution
                if region == 'urban_dense':
                    x, y = np.random.normal(0, 5, 2)
                elif region == 'urban_sparse':
                    angle = np.random.uniform(0, 2*np.pi)
                    distance = np.random.uniform(5, 15)
                    x, y = distance * np.cos(angle), distance * np.sin(angle)
                elif region == 'suburban':
                    angle = np.random.uniform(0, 2*np.pi)
                    distance = np.random.uniform(15, 25)
                    x, y = distance * np.cos(angle), distance * np.sin(angle)
                elif region == 'rural':
                    x, y = np.random.uniform(-50, 50, 2)
                else:  # highway
                    # Highways as lines radiating from center
                    angle = np.random.choice([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
                    distance = np.random.uniform(5, 40)
                    x, y = distance * np.cos(angle), distance * np.sin(angle)
                
                # Create station
                station = {
                    'station_id': id_counter,
                    'region': region,
                    'capacity': capacity,
                    'power_level': power_level,
                    'operational_cost': op_cost,
                    'location_x': x,
                    'location_y': y,
                    'allocation_schedule': [[] for _ in range(self.time_window_days * 24)],
                    'revenue': 0.0,
                    'energy_delivered': 0.0,
                    'peak_utilization': 0.0,
                    'renewable_sourced': np.random.uniform(0.1, 0.8) if region != 'highway' else np.random.uniform(0.05, 0.4)
                }
                stations.append(station)
                id_counter += 1
        
        # Add remaining stations to maintain exact count
        remaining = self.num_stations - len(stations)
        if remaining > 0:
            for _ in range(remaining):
                # Default to urban sparse
                capacity = np.random.randint(4, 12)
                power_level = np.random.choice([7, 11, 22, 50])
                op_cost = 0.02 + (power_level * 0.0008)
                
                angle = np.random.uniform(0, 2*np.pi)
                distance = np.random.uniform(5, 15)
                x, y = distance * np.cos(angle), distance * np.sin(angle)
                
                station = {
                    'station_id': id_counter,
                    'region': 'urban_sparse',
                    'capacity': capacity,
                    'power_level': power_level,
                    'operational_cost': op_cost,
                    'location_x': x,
                    'location_y': y,
                    'allocation_schedule': [[] for _ in range(self.time_window_days * 24)],
                    'revenue': 0.0,
                    'energy_delivered': 0.0,
                    'peak_utilization': 0.0,
                    'renewable_sourced': np.random.uniform(0.1, 0.6)
                }
                stations.append(station)
                id_counter += 1
        
        self.stations = stations
        
        # Station capacity statistics
        total_capacity = sum(s['capacity'] for s in stations)
        print(f"Total charging capacity: {total_capacity} ports")
        print(f"Average ports per station: {total_capacity/len(stations):.2f}")
    
    def _generate_users(self):
        """Generate realistic EV user profiles based on behavioral clusters"""
        print("Generating user profiles...")
        
        # Define user archetypes with realistic behavioral patterns
        archetypes = [
            # Regular commuters with fixed patterns
            {
                'name': 'Daily Commuter',
                'probability': 0.35,
                'battery_capacity': (40, 75),  # kWh
                'avg_daily_use': (15, 40),     # kWh
                'charging_pattern': 'routine',
                'time_sensitivity': 'high',
                'price_sensitivity': (0.5, 0.8),
                'range_anxiety': (0.3, 0.7)
            },
            # Home-based charging with occasional public charging
            {
                'name': 'Home Charger',
                'probability': 0.25,
                'battery_capacity': (60, 100),
                'avg_daily_use': (10, 30),
                'charging_pattern': 'home_dominant',
                'time_sensitivity': 'low',
                'price_sensitivity': (0.7, 0.9),
                'range_anxiety': (0.1, 0.4)
            },
            # High mileage users (e.g., rideshare drivers)
            {
                'name': 'High Mileage',
                'probability': 0.15,
                'battery_capacity': (60, 120),
                'avg_daily_use': (50, 100),
                'charging_pattern': 'frequent',
                'time_sensitivity': 'very_high',
                'price_sensitivity': (0.6, 0.8),
                'range_anxiety': (0.6, 0.9)
            },
            # Occasional EV users with irregular patterns
            {
                'name': 'Occasional',
                'probability': 0.20,
                'battery_capacity': (30, 70),
                'avg_daily_use': (5, 25),
                'charging_pattern': 'irregular',
                'time_sensitivity': 'medium',
                'price_sensitivity': (0.3, 0.6),
                'range_anxiety': (0.4, 0.8)
            },
            # Long-distance travelers
            {
                'name': 'Road Tripper',
                'probability': 0.05,
                'battery_capacity': (70, 120),
                'avg_daily_use': (60, 120),
                'charging_pattern': 'destination',
                'time_sensitivity': 'high',
                'price_sensitivity': (0.2, 0.5),
                'range_anxiety': (0.7, 0.95)
            }
        ]
        
        # Determine number of users per archetype
        users = []
        archetype_counts = {}
        for archetype in archetypes:
            count = int(self.num_users * archetype['probability'])
            archetype_counts[archetype['name']] = count
        
        # Adjust to ensure exact user count
        total_assigned = sum(archetype_counts.values())
        diff = self.num_users - total_assigned
        if diff != 0:
            archetype_counts[archetypes[0]['name']] += diff
        
        # Generate users for each archetype
        user_id = 0
        for archetype in archetypes:
            count = archetype_counts[archetype['name']]
            archetype_users = self._generate_archetype_users(archetype, count, user_id)
            users.extend(archetype_users)
            user_id += count
        
        self.users = users
        print(f"Generated {len(users)} user profiles across {len(archetypes)} behavioral archetypes")
    
    def _generate_archetype_users(self, archetype: Dict, count: int, start_id: int) -> List[Dict]:
        """Generate a group of users for a specific archetype"""
        users = []
        
        for i in range(count):
            user_id = start_id + i
            
            # Generate base characteristics
            battery_capacity = np.random.uniform(*archetype['battery_capacity'])
            avg_daily_use = np.random.uniform(*archetype['avg_daily_use'])
            price_sensitivity = np.random.uniform(*archetype['price_sensitivity'])
            range_anxiety = np.random.uniform(*archetype['range_anxiety'])
            
            # Set time sensitivity factors
            time_sensitivity_map = {
                'very_high': (0.8, 0.95),
                'high': (0.6, 0.8),
                'medium': (0.4, 0.6),
                'low': (0.2, 0.4),
                'very_low': (0.05, 0.2)
            }
            time_sensitivity = np.random.uniform(*time_sensitivity_map[archetype['time_sensitivity']])
            
            # Generate initial battery level
            initial_battery = np.random.uniform(0.1, 0.5) * battery_capacity
            
            # Create charging need patterns based on archetype
            if archetype['charging_pattern'] == 'routine':
                # Regular patterns, typically work hours
                preferred_times = []
                for day in range(self.time_window_days):
                    # Morning (pre-work) or evening (post-work) charging
                    if np.random.random() < 0.7:  # 70% evening charging preference
                        hour = np.random.randint(17, 22)  # 5pm - 10pm
                    else:
                        hour = np.random.randint(6, 9)    # 6am - 9am
                    preferred_times.append(day * 24 + hour)
            
            elif archetype['charging_pattern'] == 'home_dominant':
                # Evening/overnight charging
                preferred_times = []
                for day in range(self.time_window_days):
                    if np.random.random() < 0.9:  # 90% evening charging
                        hour = np.random.randint(19, 24)  # 7pm - 12am
                        preferred_times.append(day * 24 + hour)
            
            elif archetype['charging_pattern'] == 'frequent':
                # Multiple charging sessions throughout day
                preferred_times = []
                for day in range(self.time_window_days):
                    # Morning, mid-day and evening sessions
                    if np.random.random() < 0.8:  # 80% chance for morning
                        hour = np.random.randint(6, 10)
                        preferred_times.append(day * 24 + hour)
                    
                    if np.random.random() < 0.7:  # 70% chance for mid-day
                        hour = np.random.randint(11, 15)
                        preferred_times.append(day * 24 + hour)
                    
                    if np.random.random() < 0.9:  # 90% chance for evening
                        hour = np.random.randint(17, 22)
                        preferred_times.append(day * 24 + hour)
            
            elif archetype['charging_pattern'] == 'irregular':
                # Random charging times
                preferred_times = []
                session_count = np.random.randint(2, self.time_window_days * 2)
                for _ in range(session_count):
                    day = np.random.randint(0, self.time_window_days)
                    hour = np.random.randint(6, 23)  # 6am - 11pm
                    preferred_times.append(day * 24 + hour)
            
            else:  # destination charging
                # Concentrated charging on certain days
                preferred_times = []
                active_days = np.random.choice(range(self.time_window_days), 
                                             size=min(3, self.time_window_days),
                                             replace=False)
                for day in active_days:
                    # 2-3 sessions on active days
                    session_count = np.random.randint(2, 4)
                    for _ in range(session_count):
                        hour = np.random.randint(8, 22)
                        preferred_times.append(day * 24 + hour)
            
            # Calculate max willingness to pay
            base_price = self.base_price * (1.0 + np.random.normal(0, 0.15))  # Base with noise
            urgency_factor = 1.0 + (range_anxiety * 0.5)  # Higher anxiety = willing to pay more
            time_factor = 1.0 + (time_sensitivity * 0.3)  # Time sensitive users pay more
            
            # Adjust by archetype-specific factors
            if archetype['name'] == 'High Mileage':
                archetype_factor = 1.15  # Business users, can expense charging
            elif archetype['name'] == 'Road Tripper':
                archetype_factor = 1.25  # Need to charge to continue journey
            elif archetype['name'] == 'Home Charger':
                archetype_factor = 0.9   # Used to cheaper home charging
            else:
                archetype_factor = 1.0
            
            max_price = base_price * urgency_factor * time_factor * archetype_factor
            max_price = max_price * (1.0 - price_sensitivity * 0.3)  # Price sensitivity discount
            
            # Calculate charging needs
            daily_consumption = avg_daily_use * (0.9 + np.random.random() * 0.2)  # ±10% variation
            charging_need = daily_consumption * self.time_window_days * 0.5  # Average need over period
            
            # Preferred charging duration (hours)
            if archetype['name'] == 'Road Tripper' or time_sensitivity > 0.7:
                # Prefer fast charging
                preferred_duration = charging_need / (50 * np.random.uniform(0.8, 1.2))
            else:
                # Regular charging
                preferred_duration = charging_need / (7 * np.random.uniform(0.8, 1.2))
            
            # Minimum charging needed (as percentage of total capacity)
            min_charge_pct = max(0.1, min(0.6, range_anxiety * 0.5 + 0.2))
            min_charge_needed = min_charge_pct * battery_capacity
            
            # Bidding strategy parameters
            price_aggressiveness = (1 - price_sensitivity) * (0.8 + np.random.random() * 0.4)
            time_flexibility = (1 - time_sensitivity) * (0.8 + np.random.random() * 0.4)
            
            # Location preference based on archetype
            if archetype['name'] == 'Daily Commuter':
                location_pref = 'urban_dense' if np.random.random() < 0.6 else 'urban_sparse'
            elif archetype['name'] == 'Home Charger':
                location_pref = 'suburban' if np.random.random() < 0.7 else 'urban_sparse'
            elif archetype['name'] == 'High Mileage':
                location_pref = np.random.choice(['urban_dense', 'urban_sparse', 'highway'], p=[0.4, 0.4, 0.2])
            elif archetype['name'] == 'Occasional':
                location_pref = np.random.choice(['urban_dense', 'urban_sparse', 'suburban'], p=[0.3, 0.4, 0.3])
            else:  # Road Tripper
                location_pref = np.random.choice(['highway', 'urban_sparse'], p=[0.7, 0.3])
            
            # Create user object
            user = {
                'user_id': user_id,
                'archetype': archetype['name'],
                'battery_capacity': battery_capacity,
                'current_charge': initial_battery,
                'min_charge_needed': min_charge_needed,
                'avg_daily_use': avg_daily_use,
                'charging_need': charging_need,
                'preferred_times': preferred_times,
                'preferred_duration': preferred_duration,
                'max_price': max_price,
                'price_sensitivity': price_sensitivity,
                'time_sensitivity': time_sensitivity,
                'range_anxiety': range_anxiety,
                'price_aggressiveness': price_aggressiveness,
                'time_flexibility': time_flexibility,
                'location_preference': location_pref,
                'bids': [],
                'accepted_bids': [],
                'utility_score': 0.0,
                'final_charge': initial_battery,
                'total_cost': 0.0
            }
            
            users.append(user)
        
        return users
    
    def _train_price_model(self):
        """Train a machine learning model to predict optimal pricing based on market conditions"""
        print("Training price prediction model...")
        
        # Extract features from users
        X = []
        y = []
        
        for user in self.users:
            features = [
                user['charging_need'],
                user['time_sensitivity'],
                user['price_sensitivity'],
                user['range_anxiety'],
                user['battery_capacity'],
                user['current_charge'] / user['battery_capacity'],  # Current charge percentage
                len(user['preferred_times']) / (self.time_window_days * 5)  # Normalized charging frequency
            ]
            
            # Target: user's maximum willingness to pay
            X.append(features)
            y.append(user['max_price'])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.price_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.price_model.fit(X_scaled, y)
        
        # Calculate feature importance
        feature_names = [
            'charging_need', 'time_sensitivity', 'price_sensitivity', 
            'range_anxiety', 'battery_capacity', 'current_charge_pct', 
            'charging_frequency'
        ]
        importances = self.price_model.feature_importances_
        
        # Sort and save feature importance
        indices = np.argsort(importances)[::-1]
        feature_importance = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': [importances[i] for i in indices]
        })
        feature_importance.to_csv(f"{self.data_dir}/feature_importance.csv", index=False)
        
        print("Price prediction model trained.")
    
    def _save_datasets(self):
        """Save all generated data to CSV files"""
        # Save user profiles
        user_data = []
        for user in self.users:
            # Create simplified user record for saving
            user_record = {
                'user_id': user['user_id'],
                'archetype': user['archetype'],
                'battery_capacity': user['battery_capacity'],
                'initial_charge': user['current_charge'],
                'charging_need': user['charging_need'],
                'max_price': user['max_price'],
                'price_sensitivity': user['price_sensitivity'],
                'time_sensitivity': user['time_sensitivity'],
                'range_anxiety': user['range_anxiety'],
                'preferred_times_count': len(user['preferred_times']),
                'preferred_duration': user['preferred_duration'],
                'location_preference': user['location_preference']
            }
            user_data.append(user_record)
        
        user_df = pd.DataFrame(user_data)
        user_df.to_csv(f"{self.data_dir}/user_profiles.csv", index=False)
        
        # Save station data
        station_data = []
        for station in self.stations:
            station_record = {
                'station_id': station['station_id'],
                'region': station['region'],
                'capacity': station['capacity'],
                'power_level': station['power_level'],
                'operational_cost': station['operational_cost'],
                'location_x': station['location_x'],
                'location_y': station['location_y'],
                'renewable_sourced': station['renewable_sourced']
            }
            station_data.append(station_record)
        
        station_df = pd.DataFrame(station_data)
        station_df.to_csv(f"{self.data_dir}/charging_stations.csv", index=False)
        
        # Save grid load pattern
        grid_df = pd.DataFrame(self.grid_base_load)
        grid_df.columns = [f"hour_{i}" for i in range(24)]
        grid_df.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        grid_df.to_csv(f"{self.data_dir}/grid_load_pattern.csv")
        
        print(f"Initial datasets saved to {self.data_dir}")
    
    def run_simulation(self):
        """Run the full charging market simulation"""
        print("\n===== Starting EV Charging Market Simulation =====")
        
        # Initialize time tracking
        start_time = time.time()
        
        # Run market rounds
        for round_num in range(self.max_rounds):
            print(f"\nRound {round_num + 1}/{self.max_rounds}")
            
            # Dynamic price adjustment based on grid conditions
            grid_prices = self._calculate_grid_prices()
            
            # Calculate supply-demand balance
            supply, demand = self._calculate_market_balance()
            
            # Update user bids based on market conditions
            self._update_user_bids(grid_prices, supply, demand, round_num)
            
            # Match users to charging stations (allocation)
            transactions = self._allocate_charging_sessions(round_num)
            
            # Update market metrics
            self._update_market_metrics(grid_prices, supply, demand, transactions, round_num)
            
            # Progress report
            self._print_round_summary(round_num, transactions)
        
        # Calculate final metrics and results
        result = self._finalize_simulation()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Execution time
        execution_time = time.time() - start_time
        print(f"\nSimulation completed in {execution_time:.2f} seconds")
        
        return result
    
    def _calculate_grid_prices(self) -> Dict[int, float]:
        """Calculate dynamic grid prices for each timeslot based on load and renewable availability"""
        grid_prices = {}
        
        for hour_idx in range(len(self.time_grid)):
            # Get time components
            dt = self.time_grid[hour_idx]
            day_of_week = dt.weekday()  # 0-6 (Monday-Sunday)
            hour_of_day = dt.hour       # 0-23
            
            # Base grid load for this time
            base_load = self.grid_base_load[day_of_week, hour_of_day]
            
            # Apply seasonal factor (simplified)
            month = dt.month
            if month in [12, 1, 2]:  # Winter
                seasonal_factor = 1.1  # Higher base prices in winter
            elif month in [6, 7, 8]:  # Summer
                seasonal_factor = 1.15  # Higher prices due to AC load
            else:  # Spring/Fall
                seasonal_factor = 0.95  # Lower prices in shoulder seasons
            
            # Calculate renewable availability (simulated daily and hourly pattern)
            # Solar availability peaks mid-day, wind varies
            time_of_day = hour_of_day / 24.0
            solar_factor = max(0, min(1, 1 - (abs(time_of_day - 0.5) * 3)**2))  # Peak at noon
            wind_factor = max(0.2, min(0.8, np.sin(day_of_week + hour_of_day/6) * 0.5 + 0.5))
            
            # Combined renewable factor
            renewable_factor = (solar_factor * 0.6 + wind_factor * 0.4) * self.renewable_integration
            
            # Calculate congestion factor
            congestion = base_load * self.congestion_factor
            
            # Calculate final price for this timeslot
            price = (self.base_price * seasonal_factor * (1 + congestion) * 
                    (1 - renewable_factor * 0.5))  # Renewable reduces price
            
            grid_prices[hour_idx] = price
        
        return grid_prices
    
    def _update_user_bids(self, grid_prices: Dict[int, float], supply: Dict[int, float], 
                         demand: Dict[int, float], round_num: int):
        """Update user bidding strategies based on market conditions and previous outcomes"""
        print("Updating user bids...")
        
        # Competition factor based on supply-demand balance
        competition_factors = {}
        for hour in range(len(self.time_grid)):
            if supply[hour] > 0:
                demand_supply_ratio = demand[hour] / max(supply[hour], 1)
                # Higher ratio = more competition
                competition_factors[hour] = min(2.0, max(0.8, demand_supply_ratio))
            else:
                competition_factors[hour] = 1.0
        
        # Clear previous bids
        for user in self.users:
            user['bids'] = []
        
        # Generate new bids based on updated market conditions
        for user in self.users:
            # Skip if user already has sufficient charge
            if user['current_charge'] >= user['battery_capacity'] * 0.9:
                continue
            
            # Determine remaining charging need
            remaining_need = max(0, user['min_charge_needed'] - user['current_charge'])
            if remaining_need <= 0 and round_num > 0:
                continue
            
            # Adjust bid aggressiveness based on round number
            round_factor = min(1.2, 1.0 + (round_num / self.max_rounds) * 0.4)
            
            # Select bidding times based on preferred times and flexibility
            bidding_times = set(user['preferred_times'])
            
            # Add some flexibility based on user's time flexibility
            if user['time_flexibility'] > 0.3:
                for pt in user['preferred_times']:
                    # Add adjacent hours with decreasing probability
                    for offset in [-2, -1, 1, 2]:
                        adjacent_hour = pt + offset
                        if (0 <= adjacent_hour < len(self.time_grid) and 
                            np.random.random() < user['time_flexibility'] * (1.0 - abs(offset) * 0.3)):
                            bidding_times.add(adjacent_hour)
            
            # Create bids for selected times
            for hour in sorted(bidding_times):
                if hour >= len(self.time_grid):
                    continue
                    
                # Base price is influenced by grid price
                base_price = grid_prices[hour]
                
                # Adjust based on time sensitivity and urgency
                time_preference = 1.0
                if hour in user['preferred_times']:
                    time_preference = 1.2
                
                # Calculate bid price based on user characteristics
                urgency_factor = 1 + (user['range_anxiety'] * 0.3 * 
                                     (1 - user['current_charge']/user['battery_capacity']))
                
                competition_factor = competition_factors.get(hour, 1.0)
                
                # Base willingness to pay
                willingness = user['max_price'] * time_preference * urgency_factor
                
                # Adjust based on game theory - more aggressive in competitive times
                if competition_factor > 1.2:
                    # High competition, bid closer to max
                    bid_factor = user['price_aggressiveness'] * round_factor * competition_factor
                    bid_price = min(willingness, base_price * (1 + bid_factor * 0.5))
                else:
                    # Low competition, bid more conservatively
                    bid_factor = user['price_aggressiveness'] * round_factor
                    bid_price = base_price * (1 + bid_factor * 0.2)
                
                # Ensure bid doesn't exceed user's maximum
                bid_price = min(bid_price, user['max_price'])
                
                # Calculate energy request
                charging_duration = min(1.0, user['preferred_duration'])  # Max 1 hour per slot
                
                # Find applicable stations
                viable_stations = []
                for station in self.stations:
                    # Check if station matches location preference
                    location_match = (station['region'] == user['location_preference'])
                    
                    # Station must have open capacity for this hour
                    has_capacity = (len(station['allocation_schedule'][hour]) < station['capacity'])
                    
                    # Calculate distance (simplified)
                    if location_match and has_capacity:
                        viable_stations.append(station['station_id'])
                
                # Only create bid if viable stations exist
                if viable_stations:
                    # Energy to charge based on power level and duration
                    # We'll use average power level of 22kW (Level 2) for simplification
                    energy_request = 22 * charging_duration  # kWh
                    
                    # Ensure energy doesn't exceed battery capacity
                    energy_request = min(energy_request, 
                                        user['battery_capacity'] - user['current_charge'])
                    
                    if energy_request > 0:
                        bid = {
                            'user_id': user['user_id'],
                            'hour': hour,
                            'price': bid_price,
                            'energy': energy_request,
                            'station_preferences': viable_stations,
                            'utility': time_preference * urgency_factor
                        }
                        user['bids'].append(bid)
        
        # Count total bids
        total_bids = sum(len(user['bids']) for user in self.users)
        print(f"Generated {total_bids} new bids")
    
    def _allocate_charging_sessions(self, round_num: int) -> List[Dict]:
        """Match users to charging stations using an iterative auction mechanism"""
        print("Allocating charging sessions...")
        
        # Collect all bids
        all_bids = []
        for user in self.users:
            all_bids.extend(user['bids'])
        
        # Sort bids by price (descending) and then by utility (descending)
        all_bids.sort(key=lambda x: (x['price'], x['utility']), reverse=True)
        
        # Track allocated stations for each hour
        station_allocations = {}
        for station in self.stations:
            station_id = station['station_id']
            station_allocations[station_id] = {
                'hours': set(),  # Track allocated hours
                'energy': 0.0,   # Total energy allocated
                'revenue': 0.0   # Total revenue generated
            }
        
        # Track transactions
        transactions = []
        
        # Allocate in order of bid price
        for bid in all_bids:
            user_id = bid['user_id']
            hour = bid['hour']
            price = bid['price']
            energy = bid['energy']
            station_prefs = bid['station_preferences']
            
            # Get user
            user = next(u for u in self.users if u['user_id'] == user_id)
            
            # Skip if this user's current charge is already at capacity
            if user['current_charge'] >= user['battery_capacity'] * 0.95:
                continue
            
            # Try to find an available station
            allocated_station = None
            
            for station_id in station_prefs:
                station = next(s for s in self.stations if s['station_id'] == station_id)
                
                # Check if station has available capacity in this hour
                if len(station['allocation_schedule'][hour]) < station['capacity']:
                    # Station is available
                    allocated_station = station
                    break
            
            if allocated_station:
                # Calculate operational cost
                op_cost_per_kwh = allocated_station['operational_cost']
                
                # Calculate profit
                profit = (price - op_cost_per_kwh) * energy
                
                # Only allocate if profitable (unless final rounds where we allocate even at break-even)
                if profit > 0 or round_num > self.max_rounds * 0.7:
                    # Add to station's schedule
                    allocated_station['allocation_schedule'][hour].append({
                        'user_id': user_id,
                        'energy': energy,
                        'price': price
                    })
                    
                    # Update station metrics
                    station_id = allocated_station['station_id']
                    station_allocations[station_id]['hours'].add(hour)
                    station_allocations[station_id]['energy'] += energy
                    station_allocations[station_id]['revenue'] += price * energy
                    
                    # Update station revenue and energy delivered
                    allocated_station['revenue'] += price * energy
                    allocated_station['energy_delivered'] += energy
                    
                    # Update user metrics
                    user['current_charge'] += energy
                    user['accepted_bids'].append(bid)
                    user['total_cost'] += price * energy
                    
                    # Calculate user utility
                    time_utility = 1.0
                    if hour in user['preferred_times']:
                        time_utility = 2.0
                    
                    price_utility = 1.0 - (price / user['max_price'])
                    
                    # Update user utility score
                    charge_gain = energy / user['battery_capacity']
                    utility_gain = charge_gain * time_utility * price_utility
                    user['utility_score'] += utility_gain
                    
                    # Record transaction
                    transactions.append({
                        'round': round_num,
                        'hour': hour,
                        'user_id': user_id,
                        'station_id': allocated_station['station_id'],
                        'energy': energy,
                        'price': price,
                        'utility': utility_gain
                    })
        
        # Update station peak utilization
        for station in self.stations:
            max_hourly_usage = max(len(allocation) for allocation in station['allocation_schedule'])
            utilization = max_hourly_usage / station['capacity']
            station['peak_utilization'] = max(station['peak_utilization'], utilization)
        
        print(f"Completed {len(transactions)} charging transactions")
        return transactions
    
    def _calculate_market_balance(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Calculate supply and demand for each hour in the simulation period"""
        supply = {}
        demand = {}
        
        # Initialize
        for hour in range(len(self.time_grid)):
            supply[hour] = 0.0
            demand[hour] = 0.0
        
        # Calculate available supply (station capacity)
        for station in self.stations:
            capacity = station['capacity']
            power = station['power_level']
            hourly_supply = capacity * power  # kWh available per hour
            
            for hour in range(len(self.time_grid)):
                supply[hour] += hourly_supply
        
        # Calculate demand (from user preferred charging times)
        for user in self.users:
            # Skip if already fully charged
            if user['current_charge'] >= user['battery_capacity'] * 0.9:
                continue
                
            # Calculate energy needed
            energy_needed = user['battery_capacity'] - user['current_charge']
            
            if energy_needed <= 0:
                continue
                
            # Distribute demand across preferred hours
            if user['preferred_times']:
                energy_per_hour = energy_needed / len(user['preferred_times'])
                for hour in user['preferred_times']:
                    if 0 <= hour < len(self.time_grid):
                        demand[hour] += energy_per_hour
        
        return supply, demand
    
    

    def _update_market_metrics(self, grid_prices: Dict[int, float], supply: Dict[int, float], 
                          demand: Dict[int, float], transactions: List[Dict], round_num: int):
    #"""Update market performance metrics"""
    # Calculate average price
        if transactions:
            avg_price = sum(t['price'] for t in transactions) / len(transactions)
        else:
            avg_price = self.base_price
    
    # Calculate transaction volume
        transaction_volume = sum(t['energy'] for t in transactions)
    
    # Calculate total demand and unmet demand
        total_demand = sum(demand.values())
        total_supply = sum(supply.values())
        actual_demand = transaction_volume
        unmet_demand = total_demand - actual_demand
    
    # Calculate grid load
        avg_grid_load = 0
        for hour in range(len(self.time_grid)):
            dt = self.time_grid[hour]
            day_of_week = dt.weekday()
            hour_of_day = dt.hour
            base_load = self.grid_base_load[day_of_week, hour_of_day]
        
            hour_transactions = [t for t in transactions if t['hour'] == hour]
            hour_energy = sum(t['energy'] for t in hour_transactions)
        
            if hour_energy > 0:
                load_pct = base_load + (hour_energy / supply[hour]) * 0.3
                avg_grid_load += load_pct
    
        if len(self.time_grid) > 0:
            avg_grid_load /= len(self.time_grid)
    
    # Calculate revenue
        total_revenue = sum(t['price'] * t['energy'] for t in transactions)
    
    # Calculate renewable contribution
        renewable_energy = 0
        for t in transactions:
            station_id = t['station_id']
            station = next(s for s in self.stations if s['station_id'] == station_id)
            renewable_energy += t['energy'] * station['renewable_sourced']
    
        renewable_contribution = renewable_energy / transaction_volume if transaction_volume > 0 else 0
    
    # Calculate price volatility
        if len(transactions) > 1:
            prices = [t['price'] for t in transactions]
            price_volatility = np.std(prices) / avg_price if avg_price > 0 else 0
        else:
            price_volatility = 0
    
    # Calculate user satisfaction metrics
        total_utility = 0
        satisfied_users = 0
        for user in self.users:
            utility = self._calculate_user_utility(user)
            total_utility += utility
            if utility >= 60:  # Threshold for "satisfied"
                satisfied_users += 1
    
        user_satisfaction = total_utility / len(self.users) if self.users else 0
        satisfaction_rate = satisfied_users / len(self.users) if self.users else 0
    
    # Update market history - ALL arrays will have same length
        self.market_history['price'].append(avg_price)
        self.market_history['demand'].append(total_demand)
        self.market_history['supply'].append(total_supply)
        self.market_history['transactions'].append(transaction_volume)
        self.market_history['unmet_demand'].append(unmet_demand)
        self.market_history['revenue'].append(total_revenue)
        self.market_history['grid_load'].append(avg_grid_load)
        self.market_history['renewable_contribution'].append(renewable_contribution)
        self.market_history['price_volatility'].append(price_volatility)
        self.market_history['user_satisfaction'].append(user_satisfaction)
        self.market_history['satisfaction_rate'].append(satisfaction_rate)
    
    def _print_round_summary(self, round_num: int, transactions: List[Dict]):
        """Print summary of the current round"""
        if not transactions:
            print("No transactions in this round")
            return
        
        # Get metrics from the current round
        idx = len(self.market_history['price']) - 1
        avg_price = self.market_history['price'][idx]
        transaction_volume = self.market_history['transactions'][idx]
        revenue = self.market_history['revenue'][idx]
        renewable = self.market_history['renewable_contribution'][idx] * 100
        grid_load = self.market_history['grid_load'][idx] * 100
        
        print(f"Round {round_num + 1} Summary:")
        print(f"- Transactions: {len(transactions)}")
        print(f"- Energy Traded: {transaction_volume:.2f} kWh")
        print(f"- Avg Price: ${avg_price:.4f}/kWh")
        print(f"- Total Revenue: ${revenue:.2f}")
        print(f"- Renewable %: {renewable:.2f}%")
        print(f"- Grid Load: {grid_load:.2f}%")
    
    def _calculate_user_utility(self, user: Dict) -> float:
    #"""Calculate a comprehensive utility score (0-100) for a user"""
    # Base utility from charge received (0-100)
        if user['battery_capacity'] > 0:
            charge_ratio = user['final_charge'] / user['battery_capacity']
            charge_utility = min(100, charge_ratio * 100)
        else:
            charge_utility = 0
    
    # Price penalty (0-40 points)
        price_penalty = 0
        if user['total_cost'] > 0 and user['max_price'] > 0:
            energy_received = max(1, user['final_charge'] - user['current_charge'])
            avg_price_paid = user['total_cost'] / energy_received
            price_ratio = avg_price_paid / user['max_price']
        
            if price_ratio > 1.0:  # Paid more than max willingness
                price_penalty = min(40, (price_ratio - 1.0) * 100)
            elif price_ratio < 0.8:  # Got a good deal
                price_penalty = -10  # Small bonus
    
    # Time convenience bonus (0-20 points)
        time_bonus = 0
        if user['accepted_bids']:
            preferred_hours = sum(1 for bid in user['accepted_bids'] 
                                if bid['hour'] in user['preferred_times'])
            time_bonus = min(20, preferred_hours * 5)
    
    # Range anxiety adjustment
        anxiety_factor = 1 + (user['range_anxiety'] * 0.5)
    
    # Calculate final score
        utility = max(0, min(100, 
            (charge_utility - price_penalty + time_bonus) * anxiety_factor))
    
        return utility



    def _finalize_simulation(self) -> Dict:
    #"""Calculate final metrics and save results"""
        print("\nFinalizing simulation results...")
    
    # Calculate final charging levels and utilities
        total_utility = 0
        satisfied_users = 0
    
        for user in self.users:
            user['final_charge'] = user['current_charge']
            user['utility_score'] = self._calculate_user_utility(user)
            total_utility += user['utility_score']
        
            if user['utility_score'] >= 60:  # Threshold for "satisfied"
                satisfied_users += 1
    
    # Overall market performance metrics
        avg_price = np.mean(self.market_history['price'])
        total_energy = sum(self.market_history['transactions'])
        total_revenue = sum(self.market_history['revenue'])
        avg_grid_load = np.mean(self.market_history['grid_load'])
        avg_renewable = np.mean(self.market_history['renewable_contribution'])
        avg_satisfaction = total_utility / len(self.users) if self.users else 0
        satisfaction_rate = satisfied_users / len(self.users) if self.users else 0
    
    # Peak metrics
        peak_demand = max(self.market_history['demand'])
        peak_transactions = max(self.market_history['transactions'])
        peak_price = max(self.market_history['price'])
        peak_volatility = max(self.market_history['price_volatility'])
    
    # Station metrics
        active_stations = sum(1 for station in self.stations if station['energy_delivered'] > 0)
        avg_station_utilization = sum(s['peak_utilization'] for s in self.stations) / self.num_stations
        highest_revenue_station = max(self.stations, key=lambda s: s['revenue'])
        highest_utilization_station = max(self.stations, key=lambda s: s['peak_utilization'])
    
    # Results dictionary
        results = {
        'market_summary': {
            'avg_price': avg_price,
            'total_energy_traded': total_energy,
            'total_revenue': total_revenue,
            'avg_grid_load': avg_grid_load,
            'avg_renewable_contribution': avg_renewable,
            'user_satisfaction': avg_satisfaction,
            'satisfaction_rate': satisfaction_rate,
            'peak_demand': peak_demand,
            'peak_transactions': peak_transactions,
            'peak_price': peak_price,
            'price_volatility': peak_volatility
        },
        'station_stats': {
            'active_stations': active_stations,
            'avg_utilization': avg_station_utilization,
            'highest_revenue_station': {
                'station_id': highest_revenue_station['station_id'],
                'region': highest_revenue_station['region'],
                'revenue': highest_revenue_station['revenue']
            },
            'highest_utilization_station': {
                'station_id': highest_utilization_station['station_id'],
                'region': highest_utilization_station['region'],
                'utilization': highest_utilization_station['peak_utilization']
            }
        }
    }
    
    # Save final results
        with open(f"{self.data_dir}/simulation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save market history
        market_history_df = pd.DataFrame(self.market_history)
        market_history_df.to_csv(f"{self.data_dir}/market_history.csv", index=False)
    
    # Save final user states with detailed utility components
        user_final = []
        for user in self.users:
            user_record = {
            'user_id': user['user_id'],
            'archetype': user['archetype'],
            'battery_capacity': user['battery_capacity'],
            'initial_charge': user['current_charge'],
            'final_charge': user['final_charge'],
            'min_charge_needed': user['min_charge_needed'],
            'utility_score': user['utility_score'],
            'satisfied': user['utility_score'] >= 60,
            'total_cost': user['total_cost'],
            'transactions': len(user['accepted_bids']),
            'preferred_time_matches': sum(1 for bid in user['accepted_bids'] 
                                      if bid['hour'] in user['preferred_times']),
            'avg_price_paid': (user['total_cost'] / 
                              (user['final_charge'] - user['current_charge'])) 
                              if (user['final_charge'] > user['current_charge']) else 0,
            'max_price': user['max_price']
        }
        user_final.append(user_record)
    
        pd.DataFrame(user_final).to_csv(f"{self.data_dir}/user_final_states.csv", index=False)
    
    # Save station final states
        station_final = []
        for station in self.stations:
            station_record = {
            'station_id': station['station_id'],
            'region': station['region'],
            'capacity': station['capacity'],
            'energy_delivered': station['energy_delivered'],
            'revenue': station['revenue'],
            'peak_utilization': station['peak_utilization'],
            'renewable_sourced': station['renewable_sourced'],
            'avg_price': (station['revenue'] / station['energy_delivered']) 
                         if station['energy_delivered'] > 0 else 0
        }
        station_final.append(station_record)
    
        pd.DataFrame(station_final).to_csv(f"{self.data_dir}/station_final_states.csv", index=False)
    
        print("Simulation results saved to data directory")
        return results
    
    def _generate_visualizations(self):
        """Generate visualization graphs from simulation data"""
        print("\nGenerating visualizations...")
        
        # Create a nice color palette
        colors = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
        
        # 1. Market price, demand, and transactions over time
        plt.figure(figsize=(14, 8))
        
        # Price subplot
        plt.subplot(2, 1, 1)
        plt.plot(self.market_history['price'], color=colors[0], marker='o', label='Avg Price ($/kWh)')
        plt.title('Market Price Evolution', fontsize=14)
        plt.ylabel('Price ($/kWh)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Demand and transactions subplot
        plt.subplot(2, 1, 2)
        plt.plot(self.market_history['demand'], color=colors[1], marker='s', label='Demand (kWh)')
        plt.plot(self.market_history['transactions'], color=colors[2], marker='^', label='Transactions (kWh)')
        plt.plot(self.market_history['supply'], color=colors[3], marker='*', linestyle='--', label='Supply (kWh)')
        plt.title('Market Volume', fontsize=14)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Energy (kWh)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.graph_dir}/market_evolution.png", dpi=300)
        
        # 2. Grid metrics: load and renewable contribution
        plt.figure(figsize=(12, 6))
        plt.plot([load * 100 for load in self.market_history['grid_load']], 
                 color=colors[4], marker='o', label='Grid Load (%)')
        plt.plot([r * 100 for r in self.market_history['renewable_contribution']], 
                 color=colors[5], marker='s', label='Renewable Contribution (%)')
        plt.title('Grid Performance Metrics', fontsize=14)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.graph_dir}/grid_metrics.png", dpi=300)
        
        # 3. User satisfaction and price volatility
        plt.figure(figsize=(12, 6))
        plt.plot(self.market_history['user_satisfaction'], color=colors[6], marker='o', label='User Satisfaction')
        plt.plot(self.market_history['price_volatility'], color=colors[0], marker='s', label='Price Volatility')
        plt.title('Market Quality Metrics', fontsize=14)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.graph_dir}/market_quality.png", dpi=300)
        
        # 4. Station utilization heatmap
        station_utilization = np.zeros((self.num_stations, 24))
        
        for s_idx, station in enumerate(self.stations):
            for hour in range(24):  # We'll look at first day only for clarity
                utilization = len(station['allocation_schedule'][hour]) / station['capacity']
                station_utilization[s_idx, hour] = utilization
        
        # Sort stations by region for better visualization
        region_order = ['urban_dense', 'urban_sparse', 'suburban', 'rural', 'highway']
        region_ids = {region: i for i, region in enumerate(region_order)}
        
        # Create a list of (station_idx, region_id) tuples for sorting
        station_regions = [(i, region_ids.get(station['region'], 99)) 
                          for i, station in enumerate(self.stations)]
        station_regions.sort(key=lambda x: x[1])
        sorted_indices = [sr[0] for sr in station_regions]
        
        # Select a subset of stations for better visualization
        sample_size = min(50, self.num_stations)
        step = max(1, len(sorted_indices) // sample_size)
        sample_indices = sorted_indices[::step]
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        
        # Create custom colormap from white to blue
        cmap = LinearSegmentedColormap.from_list("BuGn_r", ['#FFFFFF', '#2C3E50'])
        
        plt.imshow(station_utilization[sample_indices, :], aspect='auto', cmap=cmap)
        plt.colorbar(label='Utilization Rate')
        plt.title('Charging Station Utilization (First Day)', fontsize=14)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Station ID', fontsize=12)
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(f"{self.graph_dir}/station_utilization.png", dpi=300)
        
        # 5. User archetype performance
        user_df = pd.DataFrame(self.users)
        archetype_performance = user_df.groupby('archetype').agg({
            'final_charge': 'mean',
            'utility_score': 'mean',
            'total_cost': 'mean'
        }).reset_index()
        
        # Normalize charging metrics for better comparison
        for i, row in archetype_performance.iterrows():
            archetype = row['archetype']
            users_in_archetype = [u for u in self.users if u['archetype'] == archetype]
            if users_in_archetype:
                avg_capacity = sum(u['battery_capacity'] for u in users_in_archetype) / len(users_in_archetype)
                archetype_performance.at[i, 'charge_pct'] = row['final_charge'] / avg_capacity * 100
        
        plt.figure(figsize=(14, 8))
        
        # Charge percentage by archetype
        plt.subplot(1, 2, 1)
        bars = plt.bar(archetype_performance['archetype'], archetype_performance['charge_pct'], color=colors)
        plt.title('Final Charge % by User Archetype', fontsize=14)
        plt.xlabel('User Archetype', fontsize=12)
        plt.ylabel('Average Charge %', fontsize=12)
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Utility score by archetype
        plt.subplot(1, 2, 2)
        bars = plt.bar(archetype_performance['archetype'], archetype_performance['utility_score'], color=colors)
        plt.title('Utility Score by User Archetype', fontsize=14)
        plt.xlabel('User Archetype', fontsize=12)
        plt.ylabel('Average Utility Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.graph_dir}/archetype_performance.png", dpi=300)
        
        # 6. Pricing strategy effectiveness (revenue vs. price)
        # Calculate average price and revenue per kWh for each station
        station_pricing = []
        for station in self.stations:
            if station['energy_delivered'] > 0:
                avg_price = station['revenue'] / station['energy_delivered']
                utilization = station['energy_delivered'] / (station['capacity'] * 24 * self.time_window_days * station['power_level'])
                station_pricing.append({
                    'station_id': station['station_id'],
                    'region': station['region'],
                    'avg_price': avg_price,
                    'revenue': station['revenue'],
                    'energy': station['energy_delivered'],
                    'utilization': utilization
                })
        
        if station_pricing:
            station_df = pd.DataFrame(station_pricing)
            
            plt.figure(figsize=(14, 8))
            
            # Create scatter plot with size based on energy delivered
            scatter = plt.scatter(station_df['avg_price'], 
                                 station_df['revenue'] / station_df['energy'],
                                 s=station_df['energy'] / 10,  # Size based on energy
                                 c=station_df['utilization'] * 100,  # Color based on utilization
                                 cmap='viridis',
                                 alpha=0.7)
            
            plt.colorbar(scatter, label='Utilization (%)')
            plt.title('Pricing Strategy Effectiveness', fontsize=14)
            plt.xlabel('Average Price ($/kWh)', fontsize=12)
            plt.ylabel('Revenue per kWh', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add annotations for station regions
            for i, row in station_df.iterrows():
                if i % 10 == 0:  # Label every 10th point to avoid crowding
                    plt.annotate(row['region'], 
                                (row['avg_price'], 
                                 row['revenue'] / row['energy']),
                                textcoords="offset points",
                                xytext=(0,5),
                                ha='center')
            
            plt.tight_layout()
            plt.savefig(f"{self.graph_dir}/pricing_strategy.png", dpi=300)
            


# How to run the simulation in VS Code
if __name__ == "__main__":
    # Initialize the simulator with default parameters
    simulator = EVChargingMarketSimulator(
        num_users=7500,  
        num_stations=50,
        max_rounds=5,
        time_window_days=7
    )
    
    # Run the simulation
    results = simulator.run_simulation()
    
    # Print summary results
    print("\n=== Simulation Results ===")
    print(f"Average Price: ${results['market_summary']['avg_price']:.4f}/kWh")
    print(f"Total Energy Traded: {results['market_summary']['total_energy_traded']:.2f} kWh")
    print(f"User Satisfaction: {results['market_summary']['user_satisfaction']:.2f}/100")
    print(f"Satisfaction Rate: {results['market_summary']['satisfaction_rate']:.1%}")