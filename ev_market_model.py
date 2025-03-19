import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

# Set the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

class EVMarketDisruptionModel:
    """
    A comprehensive model for simulating how electric vehicles disrupt the traditional auto market,
    including asymmetric motivation of incumbents and changing market dynamics over time.
    """
    
    def __init__(self, simulation_years=15):
        """Initialize the EV market disruption model"""
        self.years = np.arange(2015, 2015 + simulation_years)
        self.simulation_years = simulation_years
        
        # Initialize market parameters
        self.setup_market_segments()
        self.setup_price_trajectories()
        self.setup_performance_trajectories()
        self.setup_economic_parameters()
        
    def setup_market_segments(self):
        """Define the customer segments in the auto market"""
        self.segments = {
            'Luxury': {
                'size': 0.10,  # 10% of market
                'price_sensitivity': 0.3,
                'performance_sensitivity': 0.8,
                'range_anxiety': 0.4,
                'eco_consciousness': 0.6,
                'charging_infrastructure_sensitivity': 0.5
            },
            'Premium': {
                'size': 0.25,
                'price_sensitivity': 0.5,
                'performance_sensitivity': 0.7,
                'range_anxiety': 0.6,
                'eco_consciousness': 0.5,
                'charging_infrastructure_sensitivity': 0.7
            },
            'Mainstream': {
                'size': 0.45,
                'price_sensitivity': 0.7,
                'performance_sensitivity': 0.5,
                'range_anxiety': 0.8,
                'eco_consciousness': 0.4,
                'charging_infrastructure_sensitivity': 0.8
            },
            'Low-End': {
                'size': 0.20,
                'price_sensitivity': 0.9,
                'performance_sensitivity': 0.3,
                'range_anxiety': 0.7,
                'eco_consciousness': 0.3,
                'charging_infrastructure_sensitivity': 0.9
            }
        }
    
    def setup_price_trajectories(self):
        """Set up price trajectories for different vehicle types"""
        # Starting prices (in thousands of dollars)
        self.ice_start_prices = {
            'Luxury': 80,
            'Premium': 45,
            'Mainstream': 28,
            'Low-End': 18
        }
        
        self.ev_start_prices = {
            'Luxury': 100,
            'Premium': 65,
            'Mainstream': 45,
            'Low-End': 35
        }
        
        # Price decline rates
        self.ice_price_decline_rate = 0.01  # 1% per year
        self.ev_price_decline_rate = 0.08   # 8% per year for EVs (faster due to battery cost declines)
    
    def setup_performance_trajectories(self):
        """Set up performance trajectories for different vehicle types"""
        # Initial performance scores (0-100 scale)
        self.ice_start_performance = {
            'Luxury': 85,
            'Premium': 75,
            'Mainstream': 65,
            'Low-End': 50
        }
        
        self.ev_start_performance = {
            'Luxury': 80,
            'Premium': 65,
            'Mainstream': 55,
            'Low-End': 40
        }
        
        # Performance improvement rates
        self.ice_performance_improvement = 0.02  # 2% per year
        self.ev_performance_improvement = 0.06   # 6% per year
        
        # Range and charging parameters (specific to EVs)
        self.ev_start_range = {
            'Luxury': 300,
            'Premium': 250,
            'Mainstream': 200,
            'Low-End': 150
        }
        self.ev_range_improvement = 0.08  # 8% per year
        
        self.charging_infrastructure = 0.3  # Starts at 30% coverage
        self.charging_infrastructure_growth = 0.15  # 15% per year
    
    def setup_economic_parameters(self):
        """Set up economic parameters for market simulation"""
        # Profit margins
        self.ice_margins = {
            'Luxury': 0.20,
            'Premium': 0.15,
            'Mainstream': 0.10,
            'Low-End': 0.05
        }
        
        self.ev_margins = {
            'Luxury': 0.15,
            'Premium': 0.10,
            'Mainstream': 0.05,
            'Low-End': 0.02
        }
        
        # R&D investment parameters
        self.incumbent_rd_budget = 1000  # millions of dollars
        self.disruptor_rd_budget = 500   # millions of dollars
        
        # Investment effectiveness (how well investments convert to performance improvements)
        self.ice_investment_effectiveness = 0.5
        self.ev_investment_effectiveness = 0.8
    
    def calculate_price_trajectory(self):
        """Calculate price trajectories for ICE and EV vehicles over time"""
        # Initialize price dataframes
        ice_prices = pd.DataFrame(index=self.years, columns=self.segments.keys())
        ev_prices = pd.DataFrame(index=self.years, columns=self.segments.keys())
        
        # Calculate prices for each year and segment
        for segment in self.segments:
            # ICE prices
            ice_price = self.ice_start_prices[segment]
            for year_idx, year in enumerate(self.years):
                ice_prices.loc[year, segment] = ice_price
                ice_price *= (1 - self.ice_price_decline_rate)
            
            # EV prices
            ev_price = self.ev_start_prices[segment]
            for year_idx, year in enumerate(self.years):
                ev_prices.loc[year, segment] = ev_price
                ev_price *= (1 - self.ev_price_decline_rate)
        
        return ice_prices, ev_prices
    
    def calculate_performance_trajectory(self):
        """Calculate performance trajectories for ICE and EV vehicles over time"""
        # Initialize performance dataframes
        ice_performance = pd.DataFrame(index=self.years, columns=self.segments.keys())
        ev_performance = pd.DataFrame(index=self.years, columns=self.segments.keys())
        ev_range = pd.DataFrame(index=self.years, columns=self.segments.keys())
        
        # Calculate performance for each year and segment
        for segment in self.segments:
            # ICE performance
            ice_perf = self.ice_start_performance[segment]
            for year_idx, year in enumerate(self.years):
                ice_performance.loc[year, segment] = min(ice_perf, 100)  # Cap at 100
                ice_perf *= (1 + self.ice_performance_improvement)
            
            # EV performance
            ev_perf = self.ev_start_performance[segment]
            for year_idx, year in enumerate(self.years):
                ev_performance.loc[year, segment] = min(ev_perf, 100)  # Cap at 100
                ev_perf *= (1 + self.ev_performance_improvement)
            
            # EV range
            ev_r = self.ev_start_range[segment]
            for year_idx, year in enumerate(self.years):
                ev_range.loc[year, segment] = ev_r
                ev_r *= (1 + self.ev_range_improvement)
        
        # Calculate charging infrastructure growth
        charging_infra = [self.charging_infrastructure]
        for i in range(1, len(self.years)):
            next_value = min(charging_infra[-1] * (1 + self.charging_infrastructure_growth), 1.0)
            charging_infra.append(next_value)
        
        return ice_performance, ev_performance, ev_range, charging_infra
    
    def calculate_customer_utility(self, segment, year_idx, ice_prices, ev_prices, 
                                 ice_performance, ev_performance, ev_range, charging_infra):
        """Calculate utility for a customer segment choosing between ICE and EV"""
        segment_params = self.segments[segment]
        year = self.years[year_idx]
        
        # Extract values for this segment and year
        ice_price = ice_prices.loc[year, segment]
        ev_price = ev_prices.loc[year, segment]
        
        ice_perf = ice_performance.loc[year, segment]
        ev_perf = ev_performance.loc[year, segment]
        
        ev_r = ev_range.loc[year, segment]
        charging = charging_infra[year_idx]
        
        # Calculate price utility (higher price = lower utility)
        max_price = max(self.ice_start_prices['Luxury'], self.ev_start_prices['Luxury']) * 1.2
        ice_price_utility = 1 - (ice_price / max_price) * segment_params['price_sensitivity']
        ev_price_utility = 1 - (ev_price / max_price) * segment_params['price_sensitivity']
        
        # Calculate performance utility
        ice_perf_utility = (ice_perf / 100) * segment_params['performance_sensitivity']
        ev_perf_utility = (ev_perf / 100) * segment_params['performance_sensitivity']
        
        # EV-specific utilities
        range_utility = min(1, ev_r / 400)  # Assumes 400 miles is "enough"
        range_anxiety_factor = 1 - (segment_params['range_anxiety'] * (1 - range_utility))
        
        charging_utility = charging
        charging_impact = 1 - (segment_params['charging_infrastructure_sensitivity'] * (1 - charging_utility))
        
        eco_utility = segment_params['eco_consciousness']
        
        # Combine utilities
        ice_total_utility = 0.5 * ice_price_utility + 0.5 * ice_perf_utility
        
        ev_total_utility = (0.4 * ev_price_utility + 
                           0.3 * ev_perf_utility + 
                           0.1 * range_anxiety_factor + 
                           0.1 * charging_impact + 
                           0.1 * eco_utility)
        
        return ice_total_utility, ev_total_utility
    
    def simulate_market_share(self):
        """Simulate market share evolution over time"""
        # Get price and performance trajectories
        ice_prices, ev_prices = self.calculate_price_trajectory()
        ice_performance, ev_performance, ev_range, charging_infra = self.calculate_performance_trajectory()
        
        # Initialize market share dataframe
        market_share = pd.DataFrame(index=self.years, columns=['ICE', 'EV'])
        market_share.loc[self.years[0], 'ICE'] = 0.99
        market_share.loc[self.years[0], 'EV'] = 0.01
        
        # Detailed segment adoption
        segment_adoption = pd.DataFrame(index=self.years,
                                       columns=[f"{segment}_{vehicle}" 
                                               for segment in self.segments
                                               for vehicle in ['ICE', 'EV']])
        
        # Initial values - start with nearly all ICE
        for segment in self.segments:
            segment_adoption.loc[self.years[0], f"{segment}_ICE"] = self.segments[segment]['size'] * 0.99
            segment_adoption.loc[self.years[0], f"{segment}_EV"] = self.segments[segment]['size'] * 0.01
        
        # Simulate year by year
        for year_idx in range(1, len(self.years)):
            year = self.years[year_idx]
            prev_year = self.years[year_idx-1]
            
            # For each segment, calculate adoption based on utility
            for segment in self.segments:
                ice_utility, ev_utility = self.calculate_customer_utility(
                    segment, year_idx, ice_prices, ev_prices, 
                    ice_performance, ev_performance, ev_range, charging_infra
                )
                
                # Get previous adoption rates
                prev_ice = segment_adoption.loc[prev_year, f"{segment}_ICE"]
                prev_ev = segment_adoption.loc[prev_year, f"{segment}_EV"]
                
                # Calculate transition probabilities
                if ice_utility > ev_utility:
                    # ICE still preferred
                    p_stay_ice = 0.9  # High probability to stay with ICE
                    p_switch_to_ice = 0.2  # Low probability to switch back to ICE
                else:
                    # EV becomes preferred
                    utility_ratio = ev_utility / ice_utility
                    p_stay_ice = max(0.1, 1 - (0.3 * utility_ratio))  # Decreases as EV gets better
                    p_switch_to_ice = max(0.05, 0.2 - (0.15 * utility_ratio))  # Very low as EV gets better
                
                # Calculate new adoption rates with constraints
                new_ice = prev_ice * p_stay_ice + prev_ev * p_switch_to_ice
                new_ev = (prev_ice * (1 - p_stay_ice)) + (prev_ev * (1 - p_switch_to_ice))
                
                # Normalize to ensure segment size remains constant
                total = new_ice + new_ev
                segment_size = self.segments[segment]['size']
                new_ice = (new_ice / total) * segment_size
                new_ev = (new_ev / total) * segment_size
                
                # Update segment adoption
                segment_adoption.loc[year, f"{segment}_ICE"] = new_ice
                segment_adoption.loc[year, f"{segment}_EV"] = new_ev
            
            # Update overall market share
            market_share.loc[year, 'ICE'] = sum(segment_adoption.loc[year, f"{segment}_ICE"] 
                                             for segment in self.segments)
            market_share.loc[year, 'EV'] = sum(segment_adoption.loc[year, f"{segment}_EV"] 
                                             for segment in self.segments)
        
        return market_share, segment_adoption
    
    def calculate_asymmetric_motivation(self):
        """
        Calculate the financial incentives that create asymmetric motivation
        between incumbents and disruptors
        """
        # Get price trajectories and market shares
        ice_prices, ev_prices = self.calculate_price_trajectory()
        market_share, segment_adoption = self.simulate_market_share()
        
        # Initialize dataframes for financial metrics
        incumbent_metrics = pd.DataFrame(index=self.years, 
                                       columns=['Revenue', 'Profit', 'ICE_Revenue', 'EV_Revenue',
                                               'ICE_Profit', 'EV_Profit', 'Growth_Rate'])
        
        disruptor_metrics = pd.DataFrame(index=self.years, 
                                       columns=['Revenue', 'Profit', 'Growth_Rate'])
        
        # Assume a total market size (in units)
        total_market_units = 10000000  # 10 million vehicles annually
        
        # Calculate financial metrics year by year
        for year_idx, year in enumerate(self.years):
            # Incumbent calculations (sells both ICE and some EVs)
            ice_revenues = 0
            ice_profits = 0
            for segment in self.segments:
                ice_price = ice_prices.loc[year, segment]
                ice_margin = self.ice_margins[segment]
                ice_sales = segment_adoption.loc[year, f"{segment}_ICE"] * total_market_units
                
                segment_revenue = ice_price * ice_sales * 1000  # Convert to dollars
                segment_profit = segment_revenue * ice_margin
                
                ice_revenues += segment_revenue
                ice_profits += segment_profit
            
            # Assume incumbent has 20% of the EV market initially
            incumbent_ev_share = 0.2
            if year_idx > 0:
                # Incumbent's EV share gradually declines
                incumbent_ev_share *= 0.9
            
            ev_revenues = 0
            ev_profits = 0
            for segment in self.segments:
                ev_price = ev_prices.loc[year, segment]
                ev_margin = self.ev_margins[segment]
                ev_sales = segment_adoption.loc[year, f"{segment}_EV"] * total_market_units * incumbent_ev_share
                
                segment_revenue = ev_price * ev_sales * 1000  # Convert to dollars
                segment_profit = segment_revenue * ev_margin
                
                ev_revenues += segment_revenue
                ev_profits += segment_profit
            
            incumbent_metrics.loc[year, 'ICE_Revenue'] = ice_revenues / 1e9  # Convert to billions
            incumbent_metrics.loc[year, 'EV_Revenue'] = ev_revenues / 1e9
            incumbent_metrics.loc[year, 'Revenue'] = (ice_revenues + ev_revenues) / 1e9
            
            incumbent_metrics.loc[year, 'ICE_Profit'] = ice_profits / 1e9
            incumbent_metrics.loc[year, 'EV_Profit'] = ev_profits / 1e9
            incumbent_metrics.loc[year, 'Profit'] = (ice_profits + ev_profits) / 1e9
            
            # Disruptor calculations (only sells EVs, with 80% of EV market initially)
            disruptor_ev_share = 0.8
            if year_idx > 0:
                # Disruptor's EV share gradually increases
                disruptor_ev_share = min(0.95, disruptor_ev_share * 1.02)
            
            disruptor_revenues = 0
            disruptor_profits = 0
            for segment in self.segments:
                ev_price = ev_prices.loc[year, segment]
                ev_margin = self.ev_margins[segment]
                ev_sales = segment_adoption.loc[year, f"{segment}_EV"] * total_market_units * disruptor_ev_share
                
                segment_revenue = ev_price * ev_sales * 1000
                segment_profit = segment_revenue * ev_margin
                
                disruptor_revenues += segment_revenue
                disruptor_profits += segment_profit
            
            disruptor_metrics.loc[year, 'Revenue'] = disruptor_revenues / 1e9
            disruptor_metrics.loc[year, 'Profit'] = disruptor_profits / 1e9
        
        # Calculate growth rates
        for year_idx in range(1, len(self.years)):
            year = self.years[year_idx]
            prev_year = self.years[year_idx-1]
            
            incumbent_growth = ((incumbent_metrics.loc[year, 'Revenue'] / 
                               incumbent_metrics.loc[prev_year, 'Revenue']) - 1) * 100
            incumbent_metrics.loc[year, 'Growth_Rate'] = incumbent_growth
            
            disruptor_growth = ((disruptor_metrics.loc[year, 'Revenue'] / 
                               disruptor_metrics.loc[prev_year, 'Revenue']) - 1) * 100
            disruptor_metrics.loc[year, 'Growth_Rate'] = disruptor_growth
        
        # First year growth rate is undefined
        incumbent_metrics.loc[self.years[0], 'Growth_Rate'] = 0
        disruptor_metrics.loc[self.years[0], 'Growth_Rate'] = 0
        
        return incumbent_metrics, disruptor_metrics
    
    def calculate_investment_incentives(self):
        """
        Calculate the financial incentives for investment in ICE vs. EV technology
        for an incumbent automaker, demonstrating asymmetric motivation
        """
        # Get financial metrics
        incumbent_metrics, _ = self.calculate_asymmetric_motivation()
        
        # Calculate ROI for different investment scenarios
        investment_amount = 1000  # $1 billion investment
        
        roi_data = []
        
        for allocation_to_ev in np.arange(0, 1.1, 0.1):  # From 0% to 100% in 10% increments
            allocation_to_ice = 1 - allocation_to_ev
            
            # Create a model with modified parameters reflecting the investment
            model_with_investment = EVMarketDisruptionModel(self.simulation_years)
            
            # Modify ice improvement rate based on investment
            additional_ice_improvement = (allocation_to_ice * investment_amount / 1000) * 0.01
            model_with_investment.ice_performance_improvement = self.ice_performance_improvement + additional_ice_improvement
            
            # Modify ev improvement rate based on investment
            additional_ev_improvement = (allocation_to_ev * investment_amount / 1000) * 0.02
            model_with_investment.ev_performance_improvement = self.ev_performance_improvement + additional_ev_improvement
            
            # Run the model with new parameters
            incumbent_metrics_with_investment, _ = model_with_investment.calculate_asymmetric_motivation()
            
            # Calculate the ROI over 5 years (medium-term)
            baseline_profit_5yr = incumbent_metrics.loc[self.years[:5], 'Profit'].sum()
            new_profit_5yr = incumbent_metrics_with_investment.loc[self.years[:5], 'Profit'].sum()
            roi_5yr = ((new_profit_5yr - baseline_profit_5yr) / investment_amount) * 100
            
            # Calculate the ROI over all years (long-term)
            baseline_profit_all = incumbent_metrics.loc[:, 'Profit'].sum()
            new_profit_all = incumbent_metrics_with_investment.loc[:, 'Profit'].sum()
            roi_all = ((new_profit_all - baseline_profit_all) / investment_amount) * 100
            
            roi_data.append({
                'EV_Allocation': allocation_to_ev * 100,  # Convert to percentage
                'ICE_Allocation': allocation_to_ice * 100,
                'ROI_5yr': roi_5yr,
                'ROI_All': roi_all
            })
        
        return pd.DataFrame(roi_data)
    
    def plot_price_performance_trajectories(self):
        """Plot price and performance trajectories for ICE and EV vehicles"""
        # Get price and performance data
        ice_prices, ev_prices = self.calculate_price_trajectory()
        ice_performance, ev_performance, ev_range, charging_infra = self.calculate_performance_trajectory()
        
        # Create a 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Price Trajectories
        for segment in self.segments:
            axes[0, 0].plot(self.years, ice_prices[segment], 'b-', alpha=0.7)
            axes[0, 0].plot(self.years, ev_prices[segment], 'r-', alpha=0.7)
        
        # Add segment labels for the first and last points
        for segment in self.segments:
            # ICE labels
            axes[0, 0].text(self.years[0] - 1, ice_prices.loc[self.years[0], segment], 
                         f"ICE {segment}", ha='right', va='center', color='blue', fontsize=9)
            
            # EV labels
            axes[0, 0].text(self.years[-1] + 1, ev_prices.loc[self.years[-1], segment], 
                         f"EV {segment}", ha='left', va='center', color='red', fontsize=9)
        
        axes[0, 0].set_title('Price Trajectories Over Time')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Price ($ thousands)')
        axes[0, 0].grid(True)
        
        # Add blue for ICE and red for EV to the legend
        axes[0, 0].plot([], [], 'b-', label='ICE')
        axes[0, 0].plot([], [], 'r-', label='EV')
        axes[0, 0].legend()
        
        # Plot 2: Performance Trajectories
        for segment in self.segments:
            axes[0, 1].plot(self.years, ice_performance[segment], 'b-', alpha=0.7)
            axes[0, 1].plot(self.years, ev_performance[segment], 'r-', alpha=0.7)
        
        # Add segment labels
        for segment in self.segments:
            # ICE labels
            axes[0, 1].text(self.years[0] - 1, ice_performance.loc[self.years[0], segment], 
                         f"ICE {segment}", ha='right', va='center', color='blue', fontsize=9)
            
            # EV labels
            axes[0, 1].text(self.years[-1] + 1, ev_performance.loc[self.years[-1], segment], 
                         f"EV {segment}", ha='left', va='center', color='red', fontsize=9)
        
        axes[0, 1].set_title('Performance Trajectories Over Time')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Performance Score')
        axes[0, 1].grid(True)
        
        # Add blue for ICE and red for EV to the legend
        axes[0, 1].plot([], [], 'b-', label='ICE')
        axes[0, 1].plot([], [], 'r-', label='EV')
        axes[0, 1].legend()
        
        # Plot 3: EV Range Trajectory
        for segment in self.segments:
            axes[1, 0].plot(self.years, ev_range[segment], 'r-', alpha=0.7)
        
        # Add segment labels
        for segment in self.segments:
            axes[1, 0].text(self.years[-1] + 1, ev_range.loc[self.years[-1], segment], 
                         segment, ha='left', va='center', fontsize=9)
        
        axes[1, 0].set_title('EV Range Trajectory')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Range (miles)')
        axes[1, 0].grid(True)
        
        # Plot 4: Charging Infrastructure Development
        axes[1, 1].plot(self.years, charging_infra, 'g-', linewidth=2)
        axes[1, 1].set_title('Charging Infrastructure Development')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Infrastructure Coverage (0-1)')
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_asymmetric_motivation(self):
        """Plot the asymmetric motivation between incumbents and disruptors"""
        incumbent_metrics, disruptor_metrics = self.calculate_asymmetric_motivation()
        
        # Create a 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Revenue Comparison
        axes[0, 0].plot(self.years, incumbent_metrics['Revenue'], 'b-', linewidth=2,
                     label='Incumbent Total')
        axes[0, 0].plot(self.years, incumbent_metrics['ICE_Revenue'], 'b--', linewidth=2,
                     label='Incumbent ICE')
        axes[0, 0].plot(self.years, incumbent_metrics['EV_Revenue'], 'b:', linewidth=2,
                     label='Incumbent EV')
        axes[0, 0].plot(self.years, disruptor_metrics['Revenue'], 'r-', linewidth=2,
                     label='EV Disruptor')
        
        axes[0, 0].set_title('Revenue Comparison')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Revenue ($ billions)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Find crossover point - when disruptor revenue exceeds incumbent EV revenue
        crossover_idx = np.where(disruptor_metrics['Revenue'] > incumbent_metrics['EV_Revenue'])[0]
        if len(crossover_idx) > 0:
            crossover_year = self.years[crossover_idx[0]]
            crossover_value = disruptor_metrics.loc[crossover_year, 'Revenue']
            axes[0, 0].plot(crossover_year, crossover_value, 'ro', markersize=8)
            axes[0, 0].annotate(f'Disruptor exceeds\nincumbent EV revenue\n({crossover_year})',
                             xy=(crossover_year, crossover_value),
                             xytext=(crossover_year+1, crossover_value*1.2),
                             arrowprops=dict(facecolor='red', shrink=0.05))
        
        # Plot 2: Profit Comparison
        axes[0, 1].plot(self.years, incumbent_metrics['Profit'], 'b-', linewidth=2,
                     label='Incumbent Total')
        axes[0, 1].plot(self.years, incumbent_metrics['ICE_Profit'], 'b--', linewidth=2,
                     label='Incumbent ICE')
        axes[0, 1].plot(self.years, incumbent_metrics['EV_Profit'], 'b:', linewidth=2,
                     label='Incumbent EV')
        axes[0, 1].plot(self.years, disruptor_metrics['Profit'], 'r-', linewidth=2,
                     label='EV Disruptor')
        
        axes[0, 1].set_title('Profit Comparison')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Profit ($ billions)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Growth Rate Comparison
        axes[1, 0].plot(self.years[1:], incumbent_metrics.loc[self.years[1:], 'Growth_Rate'], 'b-', linewidth=2,
                     label='Incumbent')
        axes[1, 0].plot(self.years[1:], disruptor_metrics.loc[self.years[1:], 'Growth_Rate'], 'r-', linewidth=2,
                     label='Disruptor')
        
        axes[1, 0].set_title('Year-over-Year Growth Rate')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Growth Rate (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Asymmetric Motivation Visualization
        # This plot demonstrates the key insight of the Innovator's Dilemma:
        # The incumbent's incentive to protect its existing business vs. invest in disruption
        
        # Calculate the absolute revenue growth for both
        incumbent_abs_growth = []
        disruptor_abs_growth = []
        
        for i in range(1, len(self.years)):
            inc_growth = incumbent_metrics.loc[self.years[i], 'Revenue'] - incumbent_metrics.loc[self.years[i-1], 'Revenue']
            dis_growth = disruptor_metrics.loc[self.years[i], 'Revenue'] - disruptor_metrics.loc[self.years[i-1], 'Revenue']
            incumbent_abs_growth.append(inc_growth)
            disruptor_abs_growth.append(dis_growth)
        
        # Create a bar chart showing absolute growth
        x = np.arange(len(self.years) - 1)
        width = 0.35
        
        axes[1, 1].bar(x - width/2, incumbent_abs_growth, width, label='Incumbent Absolute Growth', color='blue')
        axes[1, 1].bar(x + width/2, disruptor_abs_growth, width, label='Disruptor Absolute Growth', color='red')
        
        axes[1, 1].set_title('Asymmetric Motivation: Absolute Revenue Growth')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Absolute Growth ($ billions)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'{y}' for y in self.years[1:]])
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Add annotation explaining the asymmetric motivation
        mid_year_idx = len(self.years) // 2
        axes[1, 1].annotate('Incumbent prioritizes\ndefending larger business\nover small disruption',
                         xy=(mid_year_idx, incumbent_abs_growth[mid_year_idx]),
                         xytext=(mid_year_idx-2, incumbent_abs_growth[mid_year_idx]*1.5),
                         arrowprops=dict(facecolor='blue', shrink=0.05),
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_investment_incentives(self):
        """Plot the ROI for different investment allocations between ICE and EV"""
        roi_data = self.calculate_investment_incentives()
        
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: 5-year ROI
        ax1.plot(roi_data['EV_Allocation'], roi_data['ROI_5yr'], 'b-o')
        
        # Add a vertical line at the maximum ROI
        max_roi_idx = roi_data['ROI_5yr'].idxmax()
        max_ev_allocation = roi_data.loc[max_roi_idx, 'EV_Allocation']
        max_roi = roi_data.loc[max_roi_idx, 'ROI_5yr']
        
        ax1.axvline(x=max_ev_allocation, color='r', linestyle='--', alpha=0.5)
        ax1.plot(max_ev_allocation, max_roi, 'ro', markersize=8)
        ax1.annotate(f'Optimal Allocation: {max_ev_allocation}% to EV\nROI: {max_roi:.1f}%',
                   xy=(max_ev_allocation, max_roi),
                   xytext=(max_ev_allocation+10, max_roi*0.9),
                   arrowprops=dict(facecolor='red', shrink=0.05),
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        ax1.set_title('5-Year ROI for Different Investment Allocations')
        ax1.set_xlabel('Allocation to EV Technology (%)')
        ax1.set_ylabel('5-Year ROI (%)')
        ax1.grid(True)
        
        # Plot 2: Full simulation period ROI
        ax2.plot(roi_data['EV_Allocation'], roi_data['ROI_All'], 'g-o')
        
        # Add a vertical line at the maximum ROI
        max_roi_idx_all = roi_data['ROI_All'].idxmax()
        max_ev_allocation_all = roi_data.loc[max_roi_idx_all, 'EV_Allocation']
        max_roi_all = roi_data.loc[max_roi_idx_all, 'ROI_All']
        
        ax2.axvline(x=max_ev_allocation_all, color='r', linestyle='--', alpha=0.5)
        ax2.plot(max_ev_allocation_all, max_roi_all, 'ro', markersize=8)
        ax2.annotate(f'Optimal Allocation: {max_ev_allocation_all}% to EV\nROI: {max_roi_all:.1f}%',
                   xy=(max_ev_allocation_all, max_roi_all),
                   xytext=(max_ev_allocation_all+10, max_roi_all*0.9),
                   arrowprops=dict(facecolor='red', shrink=0.05),
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        ax2.set_title(f'Long-term ROI ({self.simulation_years}-Year) for Different Investment Allocations')
        ax2.set_xlabel('Allocation to EV Technology (%)')
        ax2.set_ylabel('Long-term ROI (%)')
        ax2.grid(True)
        
        # Add annotation highlighting the difference between short and long-term incentives
        if max_ev_allocation != max_ev_allocation_all:
            fig.text(0.5, 0.01, 
                   f"The Innovator's Dilemma: Short-term incentives favor {max_ev_allocation}% EV investment,\n"
                   f"while long-term value is maximized at {max_ev_allocation_all}% EV investment.",
                   ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.1))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
        
    def run_full_simulation(self):
        """Run a full simulation and visualize all results"""
        print("Running EV Market Disruption Simulation...")
        
        # Generate all plots
        price_perf_fig = self.plot_price_performance_trajectories()
        market_share_fig = self.plot_market_share_evolution()
        asymmetric_fig = self.plot_asymmetric_motivation()
        investment_fig = self.plot_investment_incentives()
        
        # Display a summary of key findings
        market_share, _ = self.simulate_market_share()
        incumbent_metrics, disruptor_metrics = self.calculate_asymmetric_motivation()
        roi_data = self.calculate_investment_incentives()
        
        # Find when EVs reach 50% market share
        ev_50_idx = np.where(market_share['EV'] >= 0.5)[0]
        ev_50_year = self.years[-1]  # Default to end of simulation
        if len(ev_50_idx) > 0:
            ev_50_year = self.years[ev_50_idx[0]]
        
        # Find when disruptor revenue exceeds incumbent revenue
        crossover_idx = np.where(disruptor_metrics['Revenue'] > incumbent_metrics['Revenue'])[0]
        crossover_year = "Beyond simulation period"
        if len(crossover_idx) > 0:
            crossover_year = self.years[crossover_idx[0]]
        
        # Find optimal investment allocations
        short_max_idx = roi_data['ROI_5yr'].idxmax()
        long_max_idx = roi_data['ROI_All'].idxmax()
        short_optimal = roi_data.loc[short_max_idx, 'EV_Allocation']
        long_optimal = roi_data.loc[long_max_idx, 'EV_Allocation']
        
        print("\n=== Simulation Results ===")
        print(f"EVs reach 50% market share by: {ev_50_year}")
        print(f"Disruptor revenue exceeds incumbent revenue by: {crossover_year}")
        print(f"Optimal short-term (5-year) investment allocation to EVs: {short_optimal:.1f}%")
        print(f"Optimal long-term ({self.simulation_years}-year) investment allocation to EVs: {long_optimal:.1f}%")
        
        return {
            'price_performance_plot': price_perf_fig,
            'market_share_plot': market_share_fig,
            'asymmetric_motivation_plot': asymmetric_fig,
            'investment_incentives_plot': investment_fig,
            'ev_50pct_year': ev_50_year,
            'disruptor_exceeds_incumbent': crossover_year,
            'short_term_optimal_ev_investment': short_optimal,
            'long_term_optimal_ev_investment': long_optimal
        }

# Example usage
if __name__ == "__main__":
    # Create model with 15-year simulation
    model = EVMarketDisruptionModel(simulation_years=15)
    
    # Run full simulation
    results = model.run_full_simulation()
    
    # Show plots
    plt.show()return fig
    
    def plot_market_share_evolution(self):
        """Plot the evolution of market share between ICE and EV"""
        market_share, segment_adoption = self.simulate_market_share()
        
        # Create a 2x1 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot 1: Overall market share
        ax1.stackplot(self.years, market_share['ICE'], market_share['EV'], 
                     labels=['Internal Combustion', 'Electric Vehicles'],
                     colors=['#1f77b4', '#ff7f0e'], alpha=0.8)
        
        ax1.set_title('Overall Market Share Evolution')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Market Share')
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Add annotations for key milestones
        # Find when EVs hit 10%, 25%, and 50% market share
        for threshold in [0.1, 0.25, 0.5]:
            crossing_year_idx = np.where(market_share['EV'] >= threshold)[0]
            if len(crossing_year_idx) > 0:
                crossing_year = self.years[crossing_year_idx[0]]
                ax1.axvline(x=crossing_year, color='r', linestyle='--', alpha=0.5)
                ax1.text(crossing_year, 0.5, f'EVs reach {int(threshold*100)}%\n({crossing_year})', 
                       ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot 2: Segment adoption
        segment_ev_cols = [f"{segment}_EV" for segment in self.segments]
        segment_ice_cols = [f"{segment}_ICE" for segment in self.segments]
        
        # Reorder segments from high-end to low-end for stacking
        segment_order = ['Luxury', 'Premium', 'Mainstream', 'Low-End']
        segment_ev_cols = [f"{segment}_EV" for segment in segment_order]
        segment_ice_cols = [f"{segment}_ICE" for segment in segment_order]
        
        # Create a stacked area chart
        ev_data = [segment_adoption[col] for col in segment_ev_cols]
        ice_data = [segment_adoption[col] for col in segment_ice_cols]
        
        # Colors for segments
        segment_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
        
        # Plot stacked areas
        ax2.stackplot(self.years, *ice_data, 
                     labels=[f"{segment} ICE" for segment in segment_order],
                     colors=[color for color in segment_colors], alpha=0.6)
        
        ax2.stackplot(self.years, *ev_data, 
                     labels=[f"{segment} EV" for segment in segment_order],
                     colors=[color for color in segment_colors], alpha=1.0, baseline='wiggle')
        
        ax2.set_title('Segment-Level Adoption')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Market Share')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right', ncol=2)
        ax2.grid(True)
        
        plt.tight_layout()
