# EV Market Disruption Simulation - Streamlit App

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

# Paste the entire EVMarketDisruptionModel class here
# -----------------------------------------------------
# (Copy the complete EVMarketDisruptionModel class from ev_market_model.py)
# -----------------------------------------------------

# Set page configuration
st.set_page_config(
    page_title="EV Market Disruption Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Electric Vehicle Market Disruption Simulator")
st.markdown("""
This app simulates how electric vehicles disrupt the traditional auto market, illustrating 
Clayton Christensen's theory of disruptive innovation alongside microeconomic principles.
""")

# Create sidebar for parameters
st.sidebar.header("Simulation Parameters")

# Set up parameter inputs
simulation_years = st.sidebar.slider("Simulation Years", 5, 25, 15)

# Market parameters
st.sidebar.subheader("Market Parameters")
ev_price_decline = st.sidebar.slider("EV Price Decline Rate (%/year)", 2.0, 15.0, 8.0) / 100
ice_price_decline = st.sidebar.slider("ICE Price Decline Rate (%/year)", 0.0, 5.0, 1.0) / 100
ev_performance_improvement = st.sidebar.slider("EV Performance Improvement (%/year)", 2.0, 12.0, 6.0) / 100
ice_performance_improvement = st.sidebar.slider("ICE Performance Improvement (%/year)", 0.0, 5.0, 2.0) / 100

# Infrastructure parameters
st.sidebar.subheader("Infrastructure Parameters")
initial_charging = st.sidebar.slider("Initial Charging Infrastructure", 10.0, 50.0, 30.0) / 100
charging_growth = st.sidebar.slider("Charging Infrastructure Growth (%/year)", 5.0, 25.0, 15.0) / 100

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Share Evolution", 
    "Price & Performance", 
    "Asymmetric Motivation",
    "Investment Incentives",
    "Detailed Results"
])

# Run simulation button
if st.sidebar.button("Run Simulation", type="primary"):
    # Create progress bar
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    # Update status
    status_text.text("Initializing model...")
    progress_bar.progress(10)
    
    # Create model with custom parameters
    model = EVMarketDisruptionModel(simulation_years=simulation_years)
    
    # Update parameters based on user inputs
    model.ev_price_decline_rate = ev_price_decline
    model.ice_price_decline_rate = ice_price_decline
    model.ev_performance_improvement = ev_performance_improvement
    model.ice_performance_improvement = ice_performance_improvement
    model.charging_infrastructure = initial_charging
    model.charging_infrastructure_growth = charging_growth
    
    # Update status
    status_text.text("Running simulation...")
    progress_bar.progress(30)
    
    # Run simulation and get results
    results = model.run_full_simulation()
    
    # Update status
    status_text.text("Generating visualizations...")
    progress_bar.progress(70)
    
    # Display results in tabs
    with tab1:
        st.subheader("Market Share Evolution")
        st.pyplot(results['market_share_plot'])
        
        # Display key milestone
        st.info(f"**EVs reach 50% market share by: {results['ev_50pct_year']}**")
    
    with tab2:
        st.subheader("Price & Performance Trajectories")
        st.pyplot(results['price_performance_plot'])
        
        # Additional explanation
        st.markdown("""
        This visualization shows:
        - How EV prices decline faster than ICE prices due to battery technology improvements
        - How EV performance improves more rapidly than ICE performance
        - The growth of EV range and charging infrastructure over time
        """)
    
    with tab3:
        st.subheader("Asymmetric Motivation")
        st.pyplot(results['asymmetric_motivation_plot'])
        
        # Explanation of asymmetric motivation
        st.markdown(f"""
        ### The Innovator's Dilemma in Action
        
        This visualization demonstrates why incumbents struggle to respond to disruption:
        
        1. **Revenue Gap**: The incumbent's ICE business remains much larger than EV revenue for many years
        2. **Growth Rate Difference**: The disruptor has higher percentage growth but lower absolute dollar growth
        3. **Absolute Growth Comparison**: Shows why incumbents prioritize defending their core business
        
        Incumbents face an "asymmetric motivation" - investing in the disruptive innovation (EVs) 
        initially produces less absolute profit growth than investing in the sustaining technology (ICE improvements).
        """)
    
    with tab4:
        st.subheader("Investment Incentives")
        st.pyplot(results['investment_incentives_plot'])
        
        # Display optimal investment allocations
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Optimal Short-Term EV Investment", f"{results['short_term_optimal_ev_investment']:.1f}%")
        with col2:
            st.metric("Optimal Long-Term EV Investment", f"{results['long_term_optimal_ev_investment']:.1f}%")
        
        # Explanation of investment incentives
        st.markdown(f"""
        ### Christensen's Key Insight
        
        The difference between short-term and long-term optimal investment allocation illustrates 
        a fundamental insight from Christensen's work:
        
        **Rational short-term financial decision-making often leads incumbents to under-invest 
        in disruptive technologies, even though long-term value would be maximized with 
        greater investment in disruption.**
        """)
    
    with tab5:
        st.subheader("Detailed Simulation Results")
        
        # Get detailed data
        market_share, segment_adoption = model.simulate_market_share()
        incumbent_metrics, disruptor_metrics = model.calculate_asymmetric_motivation()
        
        # Create combined dataframe
        results_df = pd.DataFrame({
            'Year': model.years,
            'EV Market Share (%)': (market_share['EV'] * 100).round(1),
            'ICE Market Share (%)': (market_share['ICE'] * 100).round(1),
            'Incumbent Revenue ($B)': incumbent_metrics['Revenue'].round(1),
            'Disruptor Revenue ($B)': disruptor_metrics['Revenue'].round(1),
            'Incumbent Profit ($B)': incumbent_metrics['Profit'].round(1),
            'Disruptor Profit ($B)': disruptor_metrics['Profit'].round(1)
        })
        
        # Display data table
        st.dataframe(results_df.set_index('Year'), use_container_width=True)
        
        # Download button for results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="ev_market_simulation_results.csv",
            mime="text/csv",
        )
    
    # Update status
    status_text.text("Simulation complete!")
    progress_bar.progress(100)
    
else:
    # Default content when page first loads
    with tab1:
        st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see results.")
        st.image("https://via.placeholder.com/800x400.png?text=Market+Share+Evolution+Chart", use_column_width=True)
    
    with tab2:
        st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see results.")
        st.image("https://via.placeholder.com/800x400.png?text=Price+and+Performance+Trajectories", use_column_width=True)
    
    with tab3:
        st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see results.")
        st.image("https://via.placeholder.com/800x400.png?text=Asymmetric+Motivation+Analysis", use_column_width=True)
    
    with tab4:
        st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see results.")
        st.image("https://via.placeholder.com/800x400.png?text=Investment+Incentives+Chart", use_column_width=True)
    
    with tab5:
        st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to see detailed data.")

# Add instructions in the sidebar
with st.sidebar.expander("How to use this simulator"):
    st.markdown("""
    1. Adjust the simulation parameters using the sliders
    2. Click the "Run Simulation" button
    3. Explore the visualizations in each tab
    4. Try different parameter combinations to see how they affect the pace of disruption
    
    **Learning Objectives:**
    
    - Understand how disruptive innovations follow different performance trajectories
    - See why incumbent firms often struggle to respond to disruption (asymmetric motivation)
    - Observe the segment-by-segment pattern through which disruption typically occurs
    - Examine how economic incentives influence investment decisions during disruption
    """)

# Add explanations of key concepts in sidebar
with st.sidebar.expander("Key Concepts"):
    st.markdown("""
    **Disruptive Innovation**: Christensen's theory that describes how a smaller company with fewer resources can successfully challenge established incumbent businesses.
    
    **Performance Trajectories**: How product performance improves over time, often following predictable patterns.
    
    **Asymmetric Motivation**: Why rational financial incentives lead incumbents to prioritize sustaining innovations over disruptive ones.
    
    **Overshooting**: When established products exceed the needs of many customers, creating an opening for disruptive alternatives.
    """)

# Deployment instructions at the bottom of sidebar
with st.sidebar.expander("Deployment Instructions"):
    st.markdown("""
    To deploy this app:
    
    1. Save this code as `app.py`
    2. Install Streamlit: `pip install streamlit`
    3. Run: `streamlit run app.py`
    4. For web access, deploy to Streamlit Cloud:
       - Create a GitHub repository with this code
       - Connect the repository to Streamlit Cloud
    """)

# Add footer
st.markdown("---")
st.markdown(
    "Created for teaching microeconomics and disruptive innovation using Python. "
    "Based on Clayton Christensen's theories of disruptive innovation."
)
