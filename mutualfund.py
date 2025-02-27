import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from bs4 import BeautifulSoup
import re
import json
from textblob import TextBlob

try:
    import numpy_financial as npf
except ImportError:
    st.error("""
        The numpy_financial package is required but not installed.
        Please run this command in your terminal:
        
        pip install numpy-financial
        
        After installing, please restart the Streamlit app.
    """)
    st.stop()
from scipy import stats
from yahooquery import Ticker
import openpyxl
import concurrent.futures
from scipy import optimize
from mftool import Mftool
mf = Mftool()

# Set page config
st.set_page_config(
    page_title="Mutual Fund Technical Analysis",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Set dark blue background for the entire app */
    .css-18e3th9, .css-1d391kg, .css-1r6slb0 { 
        background-color: #0D1B2A !important;
    }
    body {
        background-color: #0D1B2A;
    }
    
 
    
    /* Hide the default Streamlit hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Make ALL headings white */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Arial', sans-serif;
    }
    
    /* Make sure Streamlit's native elements are also white */
    .css-1vm5v1, .css-10trblm, .css-hi6a2p { 
        color: #FFFFFF !important;
    }
    
    /* Ensure markdown headings are white */
    .markdown-text-container h1,
    .markdown-text-container h2,
    .markdown-text-container h3,
    .markdown-text-container h4,
    .markdown-text-container h5,
    .markdown-text-container h6 {
        color: #FFFFFF !important;
    }

    /* Buttons styling */
    .stButton>button {
        background-color: #457B9D !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.6rem 1rem !important;
        font-weight: 500;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1D3557 !important;
    }
    
    /* Make radio and selectbox labels white */
    .st-radio label, .st-selectbox label {
        color: #FFFFFF !important;
    }
    
    /* Make all text elements white by default */
    .css-12w0qpk, .css-1kyxreq {
        font-family: 'Arial', sans-serif;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main title
st.markdown(
    """
    <h1 style="text-align: center;"><strong>Mutual Fund Analysis Dashboard</strong></h1>
    """,
    unsafe_allow_html=True,
)

st.info(
    """
    This application provides a comprehensive suite of tools to analyze and monitor mutual fund performance, leveraging both historical data and modern financial metrics.
    Use the sidebar to select different functionalities, from historical NAV and volatility assessments to Monte Carlo simulations and news analyses.
    """
)

# -----------------------------------------
# Improved Monte Carlo Simulation function
# -----------------------------------------
def run_monte_carlo_var(nav_data: pd.DataFrame,
                        scheme_name: str,
                        num_simulations: int,
                        num_days: int,
                        confidence_level: float) -> dict:
    """
    Run a Monte Carlo simulation to estimate Value at Risk (VaR) for a mutual fund
    using a log-return based geometric Brownian motion model.

    Parameters
    ----------
    nav_data : pd.DataFrame
        Historical data containing at least one 'nav' column for the mutual fund.
    scheme_name : str
        Name of the mutual fund scheme for display or labeling.
    num_simulations : int
        Number of simulation paths to generate.
    num_days : int
        Number of trading days to project into the future.
    confidence_level : float
        Confidence level at which to compute VaR (e.g., 0.95 for 95% VaR).

    Returns
    -------
    dict
        A dictionary containing:
        - 'figure': A Plotly figure object showing the distribution of simulated final returns
                    with a vertical line indicating the VaR threshold.
        - 'var': The Value at Risk as a percentage (positive number). E.g., 5.0 means a 5% loss.
        - 'expected_shortfall': The average loss in the worst (1 - confidence_level) tail.
    """
    try:
        nav_data['nav'] = pd.to_numeric(nav_data['nav'], errors='coerce')
        nav_data.dropna(subset=['nav'], inplace=True)

        # Use log returns for more accurate simulation
        nav_data['log_return'] = np.log(nav_data['nav'] / nav_data['nav'].shift(1))
        log_returns = nav_data['log_return'].dropna()
        daily_mean = log_returns.mean()
        daily_vol = log_returns.std()
        last_nav = nav_data['nav'].iloc[-1]

        simulation_final_returns = []
        for _ in range(num_simulations):
            # Generate simulated log returns for num_days
            simulated_log_returns = np.random.normal(daily_mean, daily_vol, num_days)
            # The final price is calculated as the exponentiation of the cumulative log return
            final_nav = last_nav * np.exp(simulated_log_returns.sum())
            # Total return as a percentage
            total_return = (final_nav / last_nav) - 1.0
            simulation_final_returns.append(total_return)

        simulation_final_returns = pd.Series(simulation_final_returns)

        alpha = 1.0 - confidence_level  # tail probability
        var_threshold = simulation_final_returns.quantile(alpha)
        var_percent = -var_threshold * 100.0 if var_threshold < 0 else 0.0

        # Expected Shortfall calculation: average loss in the tail
        tail_losses = simulation_final_returns[simulation_final_returns <= var_threshold]
        es_value = -tail_losses.mean() * 100.0 if len(tail_losses) > 0 else 0.0

        # Create histogram figure using Plotly
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=simulation_final_returns * 100.0,
            histnorm='probability',
            nbinsx=50,
            marker_color='#457B9D',
            opacity=0.7,
            name='Simulated Returns'
        ))
        fig.add_vline(
            x=var_threshold * 100.0,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{confidence_level*100:.0f}% VaR<br>{var_percent:.2f}% loss",
            annotation_position="top left"
        )
        fig.add_vline(
            x=tail_losses.mean() * 100.0 if len(tail_losses) else 0.0,
            line_dash="dot",
            line_color="purple",
            annotation_text=f"ES: {es_value:.2f}%",
            annotation_position="bottom left"
        )
        fig.update_layout(
            title=f"Monte Carlo Value at Risk (VaR) - {scheme_name}",
            xaxis_title="Final Return (%)",
            yaxis_title="Frequency (probability)",
            template='plotly_white',
            hovermode='x unified',
            bargap=0.01
        )

        return {
            'figure': fig,
            'var': var_percent,
            'expected_shortfall': es_value,
        }
    except Exception as e:
        return {
            'figure': None,
            'var': None,
            'expected_shortfall': None,
            'error': str(e)
        }

scheme_names = {v: k for k, v in mf.get_scheme_codes().items()}

def analyze_historical_nav(scheme_code: str, mf_instance: Mftool) -> tuple:
    """
    Analyze and prepare historical NAV data with robust date parsing.

    Parameters
    ----------
    scheme_code : str
        The mutual fund scheme code.
    mf_instance : Mftool
        An instance of the Mftool class for fetching data.
    
    Returns
    -------
    tuple
        (DataFrame with processed historical NAV data, boolean indicating success or failure).
    """
    try:
        nav_data = mf_instance.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
        if nav_data is None or nav_data.empty:
            return None, False

        nav_data = nav_data.reset_index()
        
        def try_parse_dates(date_series):
            date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y']
            for fmt in date_formats:
                try:
                    return pd.to_datetime(date_series, format=fmt)
                except:
                    continue
            try:
                return pd.to_datetime(date_series, dayfirst=True)
            except:
                return None
        
        date_col = nav_data.columns[0]
        nav_data['Date'] = try_parse_dates(nav_data[date_col])
        
        if nav_data['Date'].isna().any():
            return None, False
        
        if date_col != 'Date':
            nav_data = nav_data.drop(columns=[date_col])
        
        nav_data['NAV'] = pd.to_numeric(nav_data['nav'], errors='coerce')
        nav_data.drop(columns=['nav'], inplace=True)

        nav_data = nav_data.sort_values('Date').dropna()

        nav_data['Daily_Returns'] = nav_data['NAV'].pct_change()
        nav_data['Cumulative_Returns'] = (1 + nav_data['Daily_Returns']).cumprod() - 1
        
        return nav_data, True

    except Exception as e:
        st.error(f"Error in analyze_historical_nav: {str(e)}")
        return None, False

option = st.sidebar.selectbox(
    "Select an Action",
    [
        "View Available Schemes",
        "Scheme details",
        "Historical NAV",
        "Average AUM",
        "Risk and Return Analysis",
        "Returns Calculator",
        "Moving Averages",
        "Volatility Analysis",
        "Monte Carlo Simulation",
        "AMC News Analysis",
        "Upload Portfolio",
        "Curated Mutual Fund Basket",
    ]
)

# Function: Recommend Mutual Funds


if option == "View Available Schemes":
    st.markdown("""<h2>View Available Schemes</h2>""", unsafe_allow_html=True)
    
    amc = st.sidebar.text_input("Filter by AMC Name", "")
    scheme_type = st.sidebar.selectbox("Filter by Type", ["All", "Equity", "Debt", "Hybrid", "Other"])
    growth_type = st.sidebar.selectbox("Filter by Growth Type", ["All", "Direct Growth", "IDCW", "Regular"])

    schemes = mf.get_available_schemes(amc)

    if schemes:
        schemes_df = pd.DataFrame(schemes.items(), columns=["Scheme Code", "Scheme Name"])
        if scheme_type != "All":
            schemes_df = schemes_df[schemes_df["Scheme Name"].str.contains(scheme_type, case=False)]
        if growth_type != "All":
            schemes_df = schemes_df[schemes_df["Scheme Name"].str.contains(growth_type, case=False)]

        st.dataframe(schemes_df, use_container_width=True)
    else:
        st.warning("No Schemes Found.")
    with st.expander("📘 Learn More: Mutual Fund Schemes"):
        st.markdown("""
    Mutual funds pool money from investors to invest in stocks, bonds, or other securities.  

    **Common Fund Types:**
    - **Equity Funds:** High growth potential, higher risk.
    - **Debt Funds:** Stable returns, lower risk.
    - **Hybrid Funds:** Mix of equity and debt to balance risk and return.
    
    **Common Growth Options:**
    - IDCW (Income Distribution cum Capital Withdrawal): Mutual fund option that provides periodic payouts from profits and capital appreciation.
    - Direct Plan: Mutual fund plan with no intermediary commissions, offering lower expense ratios and potentially higher returns.
    - Regular Plan: Mutual fund plan with distributor involvement, leading to higher expenses but offering advisory services.
    """)


elif option == "Scheme details":
    st.markdown("""<h2 style="color:#FFFFFF;">Scheme Details</h2>""", unsafe_allow_html=True)
    search_term = st.text_input("Enter AMC Name", key="scheme_details_search")
    filtered_schemes = {k: v for k, v in scheme_names.items() if search_term.lower() in k.lower()}
    if not filtered_schemes:
        st.warning("No schemes found matching your search.")
    else:
        selected_scheme = st.selectbox("Select A Scheme", list(filtered_schemes.keys()), key="scheme_details_select")
        scheme_code = scheme_names[selected_scheme]
        try:
            scheme_info = mf.get_scheme_details(scheme_code)
            if not scheme_info:
                st.error("Unable to fetch scheme details. Please try again.")
            else:
                fund_house = scheme_info.get("fund_house", "N/A")
                scheme_type = scheme_info.get("scheme_type", "N/A")
                scheme_category = scheme_info.get("scheme_category", "N/A")
                scheme_code_display = scheme_info.get("scheme_code", "N/A")
                scheme_name_display = scheme_info.get("scheme_name", "N/A")
                start_date_info = scheme_info.get("scheme_start_date", {})
                start_date = start_date_info.get("date", "N/A")
                nav = start_date_info.get("nav", "N/A")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fund House", fund_house)
                    st.metric("Scheme Type", scheme_type)
                with col2:
                    st.metric("Scheme Category", scheme_category)
                    st.metric("Scheme Code", scheme_code_display)
                with col3:
                    st.metric("Scheme Name", scheme_name_display)
                    st.metric("Launch Date", start_date)
                    st.metric("Initial NAV", nav)
        except Exception as e:
            st.error(f"Error fetching scheme details: {str(e)}")


elif option == "Historical NAV":
    st.markdown("""<h2 style="color:#FFFFFF;">Historical NAV Analysis</h2>""", unsafe_allow_html=True)
    selected_scheme = st.selectbox("Select a Scheme", list(scheme_names.keys()), key="historical_nav_scheme")
    scheme_code = scheme_names[selected_scheme]
    nav_data, success = analyze_historical_nav(scheme_code, mf)
    if not success or nav_data is None:
        st.warning("Historical NAV data not available for the selected scheme.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", nav_data['Date'].min(), key="hist_nav_start")
        with col2:
            end_date = st.date_input("End Date", nav_data['Date'].max(), key="hist_nav_end")
        mask = (nav_data['Date'].dt.date >= start_date) & (nav_data['Date'].dt.date <= end_date)
        filtered_data = nav_data.loc[mask]
        col1, col2 = st.columns(2)
        with col1:
            csv_filtered = filtered_data.to_csv(index=False)
            st.download_button(label="📥 Download Filtered Data", data=csv_filtered, file_name=f"{selected_scheme}_filtered_nav_data.csv", mime="text/csv")
        with col2:
            csv_complete = nav_data.to_csv(index=False)
            st.download_button(label="📥 Download Complete Data", data=csv_complete, file_name=f"{selected_scheme}_complete_nav_data.csv", mime="text/csv")
        st.subheader("NAV Trend")
        fig_nav = px.line(filtered_data, x='Date', y='NAV', title=f'NAV Trend: {selected_scheme}', template='plotly_white')
        st.plotly_chart(fig_nav, use_container_width=True)
        st.subheader("Cumulative Returns")
        fig_cum = px.line(filtered_data, x='Date', y='Cumulative_Returns', title=f'Cumulative Returns: {selected_scheme}', template='plotly_white')
        st.plotly_chart(fig_cum, use_container_width=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Minimum NAV", f"₹{filtered_data['NAV'].min():.2f}")
        col2.metric("Maximum NAV", f"₹{filtered_data['NAV'].max():.2f}")
        col3.metric("Average NAV", f"₹{filtered_data['NAV'].mean():.2f}")
        col4.metric("Total Return", f"{filtered_data['Cumulative_Returns'].iloc[-1]:.2%}")
        with st.expander("View Raw Data"):
            st.dataframe(filtered_data, use_container_width=True)

elif option == "Average AUM":
    st.markdown("""<h2 style="color:#FFFFFF;">Average Assets Under Management (AUM)</h2>""", unsafe_allow_html=True)
    with st.expander("📘 Learn More: Assets Under Management (AUM)"):
        st.markdown("""
- **AUM (Assets Under Management):** Total market value of assets managed by the fund.
- **Higher AUM:** Indicates investor trust and fund stability.
- **Lower AUM:** May have liquidity issues but potential for higher returns.
        """)
    quarters = ['July - September 2024', 'April - June 2024', 'January - March 2024', 'October - December 2023']
    col1, col2 = st.columns(2)
    with col1:
        selected_quarter = st.selectbox("Select Quarter", quarters, key="aum_quarter")
    with col2:
        compare_quarters = st.checkbox("Compare with Previous Quarter", key="aum_compare")
    search_query = st.text_input("Search for a Mutual Fund", "", key="aum_search")
    current_aum_data = mf.get_average_aum(selected_quarter, False)
    if current_aum_data:
        current_df = pd.DataFrame(current_aum_data)
        current_df[["AAUM Overseas", "AAUM Domestic"]] = current_df[["AAUM Overseas", "AAUM Domestic"]].astype(float)
        current_df["Total AUM"] = current_df[["AAUM Overseas", "AAUM Domestic"]].sum(axis=1)
        if search_query:
            current_df = current_df[current_df['Fund Name'].str.contains(search_query, case=False)]
            if current_df.empty:
                st.warning("No funds found matching your search criteria")
        aum_df = current_df.copy()
        for _, fund in current_df.iterrows():
            fund_name = fund['Fund Name']
            try:
                scheme_code_temp = scheme_names.get(fund_name)
                if scheme_code_temp:
                    scheme_info = mf.get_scheme_info(scheme_code_temp)
                    if scheme_info and 'sector_allocation' in scheme_info:
                        with st.expander(f"📊 Sector Allocation - {fund_name}"):
                            sector_data = pd.DataFrame(scheme_info['sector_allocation'].items(), columns=['Sector', 'Allocation %'])
                            fig = px.pie(sector_data, values='Allocation %', names='Sector', title=f"Sector Distribution for {fund_name}", color_discrete_sequence=px.colors.qualitative.Set2)
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(sector_data.style.format({'Allocation %': '{:.2f}%'}))
            except Exception:
                st.info(f"Sector allocation data not available for {fund_name}")
        if compare_quarters:
            current_quarter_idx = quarters.index(selected_quarter)
            if current_quarter_idx < len(quarters) - 1:
                previous_quarter = quarters[current_quarter_idx + 1]
                previous_aum_data = mf.get_average_aum(previous_quarter, False)
                if previous_aum_data:
                    previous_df = pd.DataFrame(previous_aum_data)
                    previous_df[["AAUM Overseas", "AAUM Domestic"]] = previous_df[["AAUM Overseas", "AAUM Domestic"]].astype(float)
                    previous_df["Total AUM"] = previous_df[["AAUM Overseas", "AAUM Domestic"]].sum(axis=1)
                    if search_query:
                        previous_df = previous_df[previous_df['Fund Name'].str.contains(search_query, case=False)]
                    merged_df = current_df.merge(previous_df[["Fund Name", "Total AUM"]], on="Fund Name", suffixes=("", "_prev"))
                    merged_df["AUM_Change_Pct"] = ((merged_df["Total AUM"] - merged_df["Total AUM_prev"]) / merged_df["Total AUM_prev"]) * 100
                    st.markdown(f"""<h3 style="color:#FFFFFF;">Quarter-over-Quarter Changes <br>({selected_quarter} vs {previous_quarter})</h3>""", unsafe_allow_html=True)
                    sort_options = ["% Change (Highest to Lowest)", "% Change (Lowest to Highest)", "Total AUM", "Fund Name"]
                    sort_by = st.radio("Sort by", sort_options, horizontal=True, key="qoq_sort")
                    if sort_by == "% Change (Highest to Lowest)":
                        merged_df = merged_df.sort_values("AUM_Change_Pct", ascending=False)
                    elif sort_by == "% Change (Lowest to Highest)":
                        merged_df = merged_df.sort_values("AUM_Change_Pct", ascending=True)
                    elif sort_by == "Total AUM":
                        merged_df = merged_df.sort_values("Total AUM", ascending=False)
                    else:
                        merged_df = merged_df.sort_values("Fund Name")
                    for _, fund in merged_df.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.markdown(f"**{fund['Fund Name']}**")
                            with col2:
                                st.metric("Current AUM", f"₹{fund['Total AUM']:,.2f} Cr", f"{fund['AUM_Change_Pct']:+.2f}%", delta_color="normal")
                            with col3:
                                st.metric("Previous AUM", f"₹{fund['Total AUM_prev']:,.2f} Cr")
                    fig = px.bar(merged_df.head(10), x="Fund Name", y="AUM_Change_Pct", title="Top 10 AUM Changes", labels={"AUM_Change_Pct": "% Change", "Fund Name": "Fund"}, color_discrete_sequence=["#457B9D"])
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data available for {previous_quarter}")
            else:
                st.warning("No previous quarter data available for comparison")
        st.markdown("""<h3 style="color:#FFFFFF;">Overall AUM Analysis</h3>""", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.radio("Sort by", ["Total AUM", "Fund Name"], horizontal=True, key="overall_sort")
        with col2:
            top_n = st.slider("Show top N funds", 5, 50, 10)
        display_df = aum_df.copy()
        if sort_by == "Total AUM":
            display_df = display_df.sort_values("Total AUM", ascending=False)
        else:
            display_df = display_df.sort_values("Fund Name")
        display_df = display_df.head(top_n)
        total_industry_aum = aum_df["Total AUM"].sum()
        avg_fund_aum = aum_df["Total AUM"].mean()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Industry AUM", f"₹{total_industry_aum:,.2f} Cr")
        with col2:
            st.metric("Average Fund AUM", f"₹{avg_fund_aum:,.2f} Cr")
        with col3:
            st.metric("Number of Funds", len(aum_df))
        tab1, tab2 = st.tabs(["AUM Distribution", "Top Funds"])
        with tab1:
            fig = px.histogram(display_df, x="Total AUM", nbins=20, title="AUM Distribution Across Funds", labels={"Total AUM": "Total AUM (Cr)", "count": "Number of Funds"}, color_discrete_sequence=["#457B9D"])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.bar(display_df.head(10), x="Fund Name", y=["AAUM Domestic", "AAUM Overseas"], title="Top 10 Funds by AUM", labels={"value": "AUM (Cr)", "variable": "AUM Type"}, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("View Detailed AUM Data"):
            st.dataframe(display_df[["Fund Name", "AAUM Domestic", "AAUM Overseas", "Total AUM"]], hide_index=True, use_container_width=True)
        with st.expander("AUM Analysis Insights"):
            st.markdown(f"""
            **Key Insights for {selected_quarter}:**
            
            1. **Market Concentration**
               - Top 10 funds control {(display_df.head(10)['Total AUM'].sum() / total_industry_aum * 100):.1f}% of total AUM
               - Average AUM per fund: ₹{avg_fund_aum:,.2f} Cr
            
            2. **Domestic vs Overseas**
               - Domestic AUM: {(display_df['AAUM Domestic'].sum() / total_industry_aum * 100):.1f}% of total
               - Overseas AUM: {(display_df['AAUM Overseas'].sum() / total_industry_aum * 100):.1f}% of total
            
            3. **Distribution**
               - Largest fund: ₹{display_df['Total AUM'].max():,.2f} Cr
               - Smallest fund: ₹{display_df['Total AUM'].min():,.2f} Cr
               - AUM Range: ₹{(display_df['Total AUM'].max() - display_df['Total AUM'].min()):,.2f} Cr
            """, unsafe_allow_html=True)
    else:
        st.error("No AUM data available for the selected quarter.")

elif option == "Performance Heatmap":
    st.markdown("""<h2>Performance Heatmap</h2>""", unsafe_allow_html=True)
    scheme_code = scheme_names[st.sidebar.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["month"] = pd.DatetimeIndex(nav_data['date']).month
        nav_data['nav'] = nav_data['nav'].astype(float)
        if 'dayChange' not in nav_data.columns:
            st.warning("No 'dayChange' column found; heatmap may be unavailable.")
        else:
            heatmap_data = nav_data.groupby("month")["dayChange"].mean().reset_index()
            heatmap_data["month"] = heatmap_data["month"].astype(str)
            fig = px.density_heatmap(
                heatmap_data,
                x="month",
                y="dayChange",
                title="NAV Performance Heatmap",
                color_continuous_scale="magma"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical NAV Data available.")
elif option == "Risk and Return Analysis":
    st.markdown("""<h2>Risk and Return Analysis</h2>""", unsafe_allow_html=True)
    with st.expander("📘 Educational Insights on Metrics"):
            st.markdown("""
            - **Alpha:** Measures excess returns relative to a benchmark. A positive alpha indicates outperformance.
            - **Beta:** Measures the fund's sensitivity to market movements. A beta > 1 means more volatile than the market.
            - **Sharpe Ratio:** Evaluates return per unit of total risk; higher values above 1 are considered good.
            - **Sortino Ratio:** Similar to Sharpe, but considers downside risk only; values above 1 are preferred.
            - **Treynor Ratio:** Measures return earned per unit of systematic risk (beta), useful for comparing funds.
            - **XIRR:** Measures the internal rate of return considering irregular cash flows; values above 10% are desirable.
            - **CAGR:** Compound Annual Growth Rate, showing consistent annual growth; values above 8% indicate strong performance.
            """)
    
    # Automatically select the first scheme by default
    default_scheme_name = list(scheme_names.keys())[0]
    scheme_name = st.selectbox("Select a Scheme", scheme_names.keys(), index=0)
    scheme_code = scheme_names[scheme_name]
    nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

    if not nav_data.empty:
        nav_data = nav_data.reset_index().rename(columns={"index": "date"})
        nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])

        # Calculate returns
        nav_data["returns"] = nav_data["nav"] / nav_data["nav"].shift(-1) - 1
        nav_data = nav_data.dropna(subset=["returns"])

        # Calculate absolute returns
        initial_nav = nav_data["nav"].iloc[-1]
        final_nav = nav_data["nav"].iloc[0]
        absolute_returns = (final_nav - initial_nav) / initial_nav

        # Calculate CAGR
        time_years = (nav_data["date"].iloc[0] - nav_data["date"].iloc[-1]).days / 365.25
        cagr = (final_nav / initial_nav) ** (1 / time_years) - 1

        # Calculate XIRR with validation
        cashflows = []
        dates = []
        
        cashflows.append(-initial_nav)
        dates.append(nav_data["date"].iloc[-1])
        cashflows.append(final_nav)
        dates.append(nav_data["date"].iloc[0])
        
        try:
            if len(cashflows) >= 2 and len(dates) >= 2 and any(cf < 0 for cf in cashflows):
                xirr = npf.xirr(cashflows, dates)
                if -1 < xirr < 10:
                    xirr_display = f"{xirr:.2%}"
                else:
                    xirr_display = "N/A (invalid range)"
            else:
                xirr_display = "N/A (insufficient data)"
        except Exception as e:
            print(f"XIRR calculation error: {str(e)}")
            xirr_display = "N/A (calculation error)"

        # Market analysis
        market_returns = nav_data["returns"] * 0.8
        cov_matrix = np.cov(nav_data["returns"], market_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        # Calculate Alpha (using CAPM)
        risk_free_rate = 0.06
        market_return = market_returns.mean() * 252
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        annualized_returns = (1 + nav_data["returns"].mean())**252 - 1
        alpha = annualized_returns - expected_return

        # Risk metrics
        downside_returns = nav_data["returns"][nav_data["returns"] < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        annualized_volatility = nav_data["returns"].std() * np.sqrt(252)
        sortino_ratio = (annualized_returns - risk_free_rate) / downside_volatility
        sharpe_ratio = (annualized_returns - risk_free_rate) / annualized_volatility
        treynor_ratio = (annualized_returns - risk_free_rate) / beta

        st.write(f"### Metrics for {scheme_name}")
        
        # Create 4 rows with 3 metrics each using columns
        row1_cols = st.columns(3)
        with row1_cols[0]:
            st.metric("Absolute Returns", f"{absolute_returns:.2%}")
        with row1_cols[1]:
            st.metric("CAGR", f"{cagr:.2%}")
        with row1_cols[2]:
            st.metric("XIRR", xirr_display)

        row2_cols = st.columns(3)
        with row2_cols[0]:
            st.metric("Annualized Returns", f"{annualized_returns:.2%}")
        with row2_cols[1]:
            st.metric("Alpha", f"{alpha:.2%}")
        with row2_cols[2]:
            st.metric("Beta", f"{beta:.2f}")

        row3_cols = st.columns(3)
        with row3_cols[0]:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with row3_cols[1]:
            st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
        with row3_cols[2]:
            st.metric("Treynor Ratio", f"{treynor_ratio:.2f}")
        
        st.write(f"### Metrics for {scheme_name}")
        nav_data["year"] = nav_data["date"].dt.year
        
        # 2. Compute annualized return and annualized volatility
        #    - Annual Return is computed by compounding daily returns for that year
        #    - Annual Volatility is daily std * sqrt(252)
        annual_stats = (
            nav_data
            .groupby("year")["returns"]
            .agg(
                annual_return=lambda x: (1 + x).prod() - 1, 
                annual_vol=lambda x: x.std() * np.sqrt(252)
            )
            .reset_index()
        )

        # 3. Plot a scatter of annual_vol (risk) vs. annual_return
        fig = px.scatter(
            annual_stats,
            x="annual_vol",
            y="annual_return",
            text="year",  # Label each point with its corresponding year
            title=f"Risk–Return Scatter for {scheme_name}",
            labels={
                "annual_vol": "Annualized Volatility (Risk)",
                "annual_return": "Annualized Return"
            },
            template="plotly_white"
        )
        fig.update_traces(textposition='top center')
        
        st.plotly_chart(fig, use_container_width=True)
  

def xirr(cashflows, dates):
    """
    Calculate the XIRR given arrays of cashflows and corresponding dates.
    
    Parameters
    ----------
    cashflows : List[float]
        List of cashflows, with negative values for investment outlays and positive for returns.
    dates : List[datetime]
        List of datetime objects corresponding to each cashflow.
    
    Returns
    -------
    float or None
        The XIRR value if convergent, otherwise None.
    """
    try:
        days = [(date - dates[0]).days for date in dates]

        def npv(rate):
            return sum([cf * (1 + rate) ** (-d / 365.0) for cf, d in zip(cashflows, days)])

        def npv_derivative(rate):
            return sum([-cf * d / 365.0 * (1 + rate) ** (-d / 365.0 - 1) for cf, d in zip(cashflows, days)])

        rate = 0.1

        for _ in range(100):
            rate_next = rate - npv(rate) / npv_derivative(rate)
            if abs(rate_next - rate) < 0.0001:
                return rate_next
            rate = rate_next
        return rate
    except:
        return None

def calculate_lumpsum_returns(amount, years, annual_return, inflation_rate):
    """
    Calculate lumpsum investment returns over a specified period.
    
    Parameters
    ----------
    amount : float
        Principal investment amount.
    years : int
        Investment duration in years.
    annual_return : float
        Annual return rate (in percent).
    inflation_rate : float
        Expected annual inflation rate (in percent).
    
    Returns
    -------
    tuple
        absolute_returns, inflation_adjusted_returns, break_even_years, CAGR, XIRR
    """
    absolute_returns = amount * (1 + annual_return / 100) ** years
    inflation_adjusted_returns = amount * (1 + (annual_return - inflation_rate) / 100) ** years

    break_even_years = np.log(1 / (1 - inflation_rate / 100)) / np.log(1 + annual_return / 100)

    cagr = (absolute_returns / amount) ** (1 / years) - 1

    dates = [datetime.now() + relativedelta(years=y) for y in range(years + 1)]
    cashflows = [-amount] + [0] * (years - 1) + [absolute_returns]
    xirr_value = xirr(cashflows, dates)

    return absolute_returns, inflation_adjusted_returns, break_even_years, cagr * 100, (xirr_value * 100 if xirr_value else 0)

def calculate_sip_returns(monthly_investment, years, annual_return, inflation_rate):
    """
    Calculate SIP returns over a specified period, accounting for inflation adjustment and XIRR.
    
    Parameters
    ----------
    monthly_investment : float
        Monthly contribution for the SIP.
    years : int
        Total number of years for the SIP.
    annual_return : float
        Annual return rate (percent).
    inflation_rate : float
        Annual inflation rate (percent).
    
    Returns
    -------
    tuple
        total_invested, sip_absolute_returns, sip_inflation_adjusted, break_even_years, XIRR
    """
    monthly_rate = annual_return / (12 * 100)
    inflation_monthly_rate = (annual_return - inflation_rate) / (12 * 100)
    total_invested = monthly_investment * 12 * years

    sip_absolute_returns = monthly_investment * ((1 + monthly_rate) * ((1 + monthly_rate) ** (12 * years) - 1) / monthly_rate)
    
    sip_inflation_adjusted = monthly_investment * ((1 + inflation_monthly_rate) * ((1 + inflation_monthly_rate) ** (12 * years) - 1) / inflation_monthly_rate)

    break_even_months = np.log(monthly_rate * total_invested / monthly_investment) / np.log(1 + monthly_rate)

    dates = [datetime.now() + relativedelta(months=m) for m in range(years * 12 + 1)]
    cashflows = [-monthly_investment] * (years * 12) + [sip_absolute_returns]
    xirr_value = xirr(cashflows, dates)

    return total_invested, sip_absolute_returns, sip_inflation_adjusted, break_even_months / 12, (xirr_value * 100 if xirr_value else 0)

def calculate_risk_metrics(returns: pd.Series) -> tuple:
    """
    Calculate fundamental risk metrics given a series of returns.
    
    Parameters
    ----------
    returns : Series
        A pandas Series containing daily return values.
    
    Returns
    -------
    tuple
        (Sharpe Ratio, Sortino Ratio, Beta)
    """
    risk_free_rate = 0.06  
    excess_returns = returns - risk_free_rate / 252

    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    downside_returns = returns[returns < 0]
    sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns) if len(downside_returns) > 0 else np.nan

    beta = 1.0

    return sharpe, sortino, beta

def create_chart(x, y_values, labels, title, x_label, y_label):
    fig = go.Figure()
    for y, label in zip(y_values, labels):
        fig.add_trace(go.Scatter(x=x, y=y, name=label, mode='lines+markers'))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode='x unified'
    )
    return fig

def display_metrics(metrics_dict: dict):
    """
    Display a dictionary of metrics in columns.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary with {label: value} pairs to display as metrics.
    """
    cols = st.columns(len(metrics_dict))
    for col, (label, value) in zip(cols, metrics_dict.items()):
        col.metric(label, value)


if option == "Returns Calculator":
    
    st.markdown("""<h2>Investment Returns Calculator</h2>""", unsafe_allow_html=True)
    

    selected_scheme = st.selectbox("Select a Scheme", scheme_names.keys())
    scheme_code = scheme_names[selected_scheme]

    try:
        nav_data, success = analyze_historical_nav(scheme_code, mf)
        if success:
            historical_returns = nav_data['Daily_Returns'].mean() * 252 * 100
            historical_volatility = nav_data['Daily_Returns'].std() * np.sqrt(252) * 100
            sharpe, sortino, beta = calculate_risk_metrics(nav_data['Daily_Returns'])

            st.info(
                f"""
                **Historical Performance Metrics for {selected_scheme}:**
                - Annual Return: {historical_returns:.2f}%
                - Volatility: {historical_volatility:.2f}%
                - Sharpe Ratio: {sharpe:.2f}
                - Sortino Ratio: {sortino:.2f}
                - Beta: {beta:.2f}
                """
            )

            return_type = st.radio(
                "Select Return Calculation Method",
                ["Use Historical Returns", "Custom Returns"],
                horizontal=True
            )

            if return_type == "Use Historical Returns":
                expected_return = historical_returns
                st.success(f"Using historical return rate of {historical_returns:.2f}%")
            else:
                expected_return = st.number_input(
                    "Expected Annual Return (%)",
                    min_value=1.0,
                    max_value=30.0,
                    value=float(f"{historical_returns:.1f}")
                )

        else:
            st.warning("Unable to fetch historical data. Please use custom returns.")
            expected_return = st.number_input(
                "Expected Annual Return (%)",
                min_value=1.0,
                max_value=30.0,
                value=12.0
            )
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        expected_return = st.number_input(
            "Expected Annual Return (%)",
            min_value=1.0,
            max_value=30.0,
            value=12.0
        )

    calc_type = st.radio("Select Investment Type", ["Lumpsum", "SIP"], horizontal=True)

    if calc_type == "Lumpsum":
        col1, col2 = st.columns(2)
        with col1:
            investment_amount = st.number_input("Investment Amount (₹)", min_value=100, value=10000)
            investment_period = st.number_input("Investment Period (Years)", min_value=1, max_value=30, value=5)
        with col2:
            inflation_rate = st.number_input("Expected Inflation Rate (%)", min_value=0.0, max_value=15.0, value=6.0)

        if st.button("Calculate Lumpsum Returns"):
            abs_returns, inf_adj_returns, break_even, cagr, xirr_val = calculate_lumpsum_returns(
                investment_amount, investment_period, expected_return, inflation_rate
            )

            basic_tab, detailed_tab, risk_tab = st.tabs(["Basic View", "Detailed Analysis", "Risk Metrics"])

            with basic_tab:
                display_metrics({
                    "Investment Amount": f"₹{investment_amount:,.2f}",
                    "Expected Value": f"₹{abs_returns:,.2f}",
                    "Inflation Adjusted Value": f"₹{inf_adj_returns:,.2f}",
                    "Break-even (Years)": f"{break_even:.1f}"
                })

                years = list(range(investment_period + 1))
                invested = [investment_amount] * len(years)
                abs_values = [
                    investment_amount * (1 + expected_return / 100) ** year
                    for year in years
                ]
                inf_adj_values = [
                    investment_amount * (1 + (expected_return - inflation_rate) / 100) ** year
                    for year in years
                ]

                st.plotly_chart(
                    create_chart(
                        years,
                        [invested, abs_values, inf_adj_values],
                        ["Amount Invested", "Expected Value", "Inflation Adjusted Value"],
                        "Investment Growth Over Time",
                        "Years",
                        "Value (₹)"
                    ),
                    use_container_width=True
                )

            with detailed_tab:
                total_returns = ((abs_returns - investment_amount) / investment_amount) * 100
                real_returns = ((inf_adj_returns - investment_amount) / investment_amount) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Returns", f"{total_returns:.2f}%")
                    st.metric("CAGR", f"{cagr:.2f}%")
                with col2:
                    st.metric("Real Returns", f"{real_returns:.2f}%")
                    st.metric("XIRR", f"{xirr_val:.2f}%")
                with col3:
                    st.metric("Inflation Impact", f"₹{(abs_returns - inf_adj_returns):,.2f}")
                    st.metric("Break-even Period", f"{break_even:.1f} years")

                cumulative_returns = pd.DataFrame({
                    'Year': years,
                    'Cumulative Returns': [(v - investment_amount) / investment_amount * 100 for v in abs_values],
                    'Real Returns': [(v - investment_amount) / investment_amount * 100 for v in inf_adj_values]
                })
                fig_cumulative = px.line(
                    cumulative_returns,
                    x='Year',
                    y=['Cumulative Returns', 'Real Returns'],
                    title='Cumulative Returns Over Time',
                    template='plotly_white'
                )
                st.plotly_chart(fig_cumulative, use_container_width=True)

            with risk_tab:
                if success:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    with col2:
                        st.metric("Sortino Ratio", f"{sortino:.2f}")
                    with col3:
                        st.metric("Beta", f"{beta:.2f}")

                    rolling_vol = nav_data['Daily_Returns'].rolling(window=252).std() * np.sqrt(252) * 100
                    fig_vol = px.line(
                        rolling_vol,
                        title='Rolling Annual Volatility',
                        labels={'value': 'Volatility (%)', 'index': 'Date'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                    cum_returns = (1 + nav_data['Daily_Returns']).cumprod()
                    rolling_max = cum_returns.rolling(window=252, min_periods=1).max()
                    drawdowns = (cum_returns - rolling_max) / rolling_max * 100
                    fig_dd = px.line(
                        drawdowns,
                        title='Historical Drawdowns',
                        labels={'value': 'Drawdown (%)', 'index': 'Date'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
                else:
                    st.warning("Historical data is required for risk metrics visualization.")

    else:
        col1, col2 = st.columns(2)
        with col1:
            monthly_investment = st.number_input("Monthly SIP Amount (₹)", min_value=100, value=5000)
            investment_period = st.number_input("Investment Period (Years)", min_value=1, max_value=30, value=5)
        with col2:
            inflation_rate = st.number_input("Expected Inflation Rate (%)", min_value=0.0, max_value=15.0, value=6.0)

        if st.button("Calculate SIP Returns"):
            total_inv, sip_abs_ret, sip_inf_adj, break_even, xirr_val = calculate_sip_returns(
                monthly_investment, investment_period, expected_return, inflation_rate
            )

            basic_tab, detailed_tab, risk_tab = st.tabs(["Basic View", "Detailed Analysis", "Risk Metrics"])

            with basic_tab:
                display_metrics({
                    "Total Investment": f"₹{total_inv:,.2f}",
                    "Expected Value": f"₹{sip_abs_ret:,.2f}",
                    "Inflation Adjusted Value": f"₹{sip_inf_adj:,.2f}",
                    "Break-even (Years)": f"{break_even:.1f}"
                })

                months = list(range(investment_period * 12 + 1))
                invested = [monthly_investment * m for m in months]
                abs_values = [
                    monthly_investment * (
                        (1 + expected_return / (12 * 100)) *
                        ((1 + expected_return / (12 * 100)) ** m - 1) /
                        (expected_return / (12 * 100))
                    ) for m in months
                ]
                inf_adj_values = [
                    monthly_investment * (
                        (1 + (expected_return - inflation_rate) / (12 * 100)) *
                        ((1 + (expected_return - inflation_rate) / (12 * 100)) ** m - 1) /
                        ((expected_return - inflation_rate) / (12 * 100))
                    ) for m in months
                ]

                st.plotly_chart(
                    create_chart(
                        [m / 12 for m in months],
                        [invested, abs_values, inf_adj_values],
                        ["Amount Invested", "Expected Value", "Inflation Adjusted Value"],
                        "SIP Growth Over Time",
                        "Years",
                        "Value (₹)"
                    ),
                    use_container_width=True
                )

            with detailed_tab:
                total_returns = ((sip_abs_ret - total_inv) / total_inv) * 100
                real_returns = ((sip_inf_adj - total_inv) / total_inv) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Returns", f"{total_returns:.2f}%")
                    st.metric("XIRR", f"{xirr_val:.2f}%")
                with col2:
                    st.metric("Real Returns", f"{real_returns:.2f}%")
                    st.metric("Break-even Period", f"{break_even:.1f} years")
                with col3:
                    st.metric("Inflation Impact", f"₹{(sip_abs_ret - sip_inf_adj):,.2f}")
                    st.metric("Monthly Investment", f"₹{monthly_investment:,.2f}")

                monthly_data = pd.DataFrame({
                    'Month': [m / 12 for m in months],
                    'Monthly Contribution': invested,
                    'Investment Value': abs_values
                })
                fig_contribution = px.area(
                    monthly_data,
                    x='Month',
                    y=['Monthly Contribution', 'Investment Value'],
                    title='Investment Growth Breakdown',
                    template='plotly_white'
                )
                st.plotly_chart(fig_contribution, use_container_width=True)

            with risk_tab:
                if success:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    with col2:
                        st.metric("Sortino Ratio", f"{sortino:.2f}")
                    with col3:
                        st.metric("Beta", f"{beta:.2f}")

                    rolling_vol = nav_data['Daily_Returns'].rolling(window=252).std() * np.sqrt(252) * 100
                    fig_vol = px.line(
                        rolling_vol,
                        title='Rolling Annual Volatility',
                        labels={'value': 'Volatility (%)', 'index': 'Date'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                    cum_returns = (1 + nav_data['Daily_Returns']).cumprod()
                    rolling_max = cum_returns.rolling(window=252, min_periods=1).max()
                    drawdowns = (cum_returns - rolling_max) / rolling_max * 100
                    fig_dd = px.line(
                        drawdowns,
                        title='Historical Drawdowns',
                        labels={'value': 'Drawdown (%)', 'index': 'Date'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
                else:
                    st.warning("Historical data is required for risk metrics visualization.")

    if 'nav_data' in locals() and success:
        with st.expander("View Historical Performance Analysis"):
            st.subheader("Historical Performance Metrics")

            hist_returns = nav_data['Daily_Returns'].dropna()
            rolling_returns = hist_returns.rolling(window=252).mean() * 252 * 100

            fig_dist = px.histogram(
                hist_returns * 100,
                title="Distribution of Daily Returns",
                labels={'value': 'Daily Return (%)', 'count': 'Frequency'},
                nbins=50,
                template='plotly_white'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            fig_rolling = px.line(
                rolling_returns,
                title="1-Year Rolling Returns",
                labels={'value': 'Annual Return (%)', 'index': 'Date'},
                template='plotly_white'
            )
            st.plotly_chart(fig_rolling, use_container_width=True)

            st.subheader("Risk Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maximum Drawdown", f"{(nav_data['NAV'].pct_change().min() * 100):.2f}%")
                st.metric("Daily Volatility", f"{(hist_returns.std() * 100):.2f}%")
            with col2:
                st.metric("Best Day Return", f"{(hist_returns.max() * 100):.2f}%")
                st.metric("Worst Day Return", f"{(hist_returns.min() * 100):.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                st.metric("Sortino Ratio", f"{sortino:.2f}")

elif option == "Moving Averages":
    st.markdown("""<h2>Moving Average Analysis</h2>""", unsafe_allow_html=True)

    # Add an expander to explain the moving average concepts
    with st.expander("What are Moving Averages?"):
        st.markdown("""
        **Moving Averages (MA)** smooth out short-term fluctuations to identify broader market trends.
        - A **50-day MA** is a shorter-term average, more sensitive to price changes.
        - A **100-day MA** is a longer-term average, less sensitive but can show bigger trend shifts.
        
        **Common Strategy**: 
        - A *Buy signal* can appear when a shorter-term MA crosses above a longer-term MA.
        - A *Sell signal* can appear when a shorter-term MA falls below a longer-term MA.
        """)

    selected_scheme = st.selectbox("Select a Scheme", scheme_names.keys())
    scheme_code = scheme_names[selected_scheme]

    try:
        nav_data, success = analyze_historical_nav(scheme_code, mf)

        if success:
            # Calculate moving averages
            nav_data['50_MA'] = nav_data['NAV'].rolling(window=50).mean()
            nav_data['100_MA'] = nav_data['NAV'].rolling(window=100).mean()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", nav_data['Date'].min())
            with col2:
                end_date = st.date_input("End Date", nav_data['Date'].max())

            mask = (nav_data['Date'].dt.date >= start_date) & (nav_data['Date'].dt.date <= end_date)
            filtered_data = nav_data.loc[mask]

            # Define signals with bold & color
            def get_signal(row):
                if row['50_MA'] > row['100_MA']:
                    return "**<span style='color:green'>BUY</span>**"
                elif row['50_MA'] < row['100_MA']:
                    return "**<span style='color:red'>SELL</span>**"
                else:
                    return "**HOLD**"
            
            filtered_data['Signal'] = filtered_data.apply(get_signal, axis=1)

            # Identify crossover points
            filtered_data['NumericSignal'] = np.where(filtered_data['50_MA'] > filtered_data['100_MA'], 1,
                                        np.where(filtered_data['50_MA'] < filtered_data['100_MA'], -1, 0))
            filtered_data['Crossover'] = filtered_data['NumericSignal'].diff().fillna(0)
            crossovers = filtered_data[filtered_data['Crossover'].abs() == 2].copy()  # +2 = buy crossover, -2 = sell crossover

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['NAV'],
                name='NAV',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['50_MA'],
                name='50-day MA',
                line=dict(color='orange', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['100_MA'],
                name='100-day MA',
                line=dict(color='red', dash='dash')
            ))

            # Plot crossover markers
            if not crossovers.empty:
                def color_marker(sig_value):
                    # NumericSignal difference from +1 -> -1 or -1 -> +1
                    # Here crossovers happen if we jump from -1 to +1 (buy) or +1 to -1 (sell)
                    return 'green' if sig_value == 2 else 'red'

                fig.add_trace(go.Scatter(
                    x=crossovers['Date'],
                    y=crossovers['NAV'],
                    mode='markers',
                    name='Crossover Points',
                    marker=dict(
                        color=[color_marker(val) for val in crossovers['Crossover']],
                        size=10,
                        symbol='diamond'
                    ),
                    text=crossovers['Signal'],
                    hovertemplate='Date: %{x}<br>NAV: %{y}<br>Signal: %{text}<extra></extra>'
                ))

            fig.update_layout(
                title=f'Moving Averages Analysis for {selected_scheme}',
                xaxis_title='Date',
                yaxis_title='NAV',
                template='plotly_white',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display Current Signal in bold/color
            current_signal = filtered_data['Signal'].iloc[-1]
            st.markdown(
                f"### Current Signal: {current_signal}",
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current NAV", f"₹{filtered_data['NAV'].iloc[-1]:.2f}")
            with col2:
                st.metric("50-day MA", f"₹{filtered_data['50_MA'].iloc[-1]:.2f}")
            with col3:
                st.metric("100-day MA", f"₹{filtered_data['100_MA'].iloc[-1]:.2f}")

            with st.expander("View Crossover Points"):
                if not crossovers.empty:
                    # Convert signals to HTML
                    crossovers_display = crossovers[['Date', 'NAV', '50_MA', '100_MA', 'Signal']].copy()
                    st.dataframe(crossovers_display, use_container_width=True)
                else:
                    st.write("No crossover points found in the selected date range.")

            with st.expander("Technical Analysis Insights"):
                current_nav = filtered_data['NAV'].iloc[-1]
                current_50ma = filtered_data['50_MA'].iloc[-1]
                current_100ma = filtered_data['100_MA'].iloc[-1]
                trend_strength = abs((current_50ma - current_100ma) / current_100ma) * 100

                st.markdown(f"""
                - **Trend Strength**: {trend_strength:.2f}%
                - **NAV vs. 50-day MA**: {"Above" if current_nav > current_50ma else "Below"}
                - **NAV vs. 100-day MA**: {"Above" if current_nav > current_100ma else "Below"}
                - **SMA Crossover**: {"Upward trend" if current_50ma > current_100ma else "Downward trend"}
                """)

                volatility = filtered_data['NAV'].pct_change().std() * np.sqrt(252) * 100
                st.markdown(f"**Annualized Volatility**: {volatility:.2f}%")

        else:
            st.error("Unable to fetch historical NAV data for the selected scheme.")

    except Exception as e:
        st.error(f"Error in Moving Averages analysis: {str(e)}")


elif option == "Volatility Analysis":
    st.markdown("""<h2>Volatility Analysis</h2>""", unsafe_allow_html=True)
    st.markdown("### Understanding Volatility")
    with st.expander("Learn More About Volatility"):
      st.markdown("""
    - **Standard Deviation:** Measures the spread of returns.
    - **Skewness:** Indicates asymmetry of return distribution.
    - **Kurtosis:** Measures the peakedness of return distribution.
    - **Maximum Drawdown:** Shows the maximum loss from peak to trough.
    """)


    def get_skewness_interpretation(skew):
        if abs(skew) < 0.5:
            return "Distribution is approximately symmetric"
        elif skew < -0.5:
            return "Distribution is negatively skewed (left-tailed)"
        else:
            return "Distribution is positively skewed (right-tailed)"
            
    scheme_code = scheme_names[st.selectbox("Select a Scheme", scheme_names.keys())]
    nav_data, success = analyze_historical_nav(scheme_code, mf)

    if success and not nav_data.empty:
        nav_data = nav_data.set_index('Date')
        returns = nav_data['Daily_Returns'].dropna()

        std_dev = returns.std() * np.sqrt(252)
        variance = returns.var() * 252
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        market_returns = returns * 0.8
        cov_matrix = np.cov(returns, market_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns.mean()

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Annualized Volatility", f"{std_dev*100:.2f}%")
        with col2:
            st.metric("Annualized Variance", f"{variance*100:.2f}%")
        with col3:
            st.metric("Beta", f"{beta:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
        with col5:
            st.metric("Skewness", f"{skewness:.2f}")
        with col6:
            st.metric("Kurtosis", f"{kurtosis:.2f}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Rolling Volatility",
            "Standard Deviation",
            "Skewness Analysis",
            "Drawdowns",
            "Risk Metrics"
        ])

        window_options = {
            "30 Days": 30,
            "60 Days": 60,
            "90 Days": 90,
            "180 Days": 180,
            "1 Year": 252
        }

        with tab1:
            selected_window = st.selectbox("Select Rolling Window", options=list(window_options.keys()))
            window_size = window_options[selected_window]
            rolling_std = returns.rolling(window=window_size).std() * np.sqrt(252)
            rolling_var = returns.rolling(window=window_size).var() * 252

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=rolling_std.index[window_size:],
                y=rolling_std[window_size:],
                name='Rolling Volatility',
                line=dict(color='red')
            ))
            fig_vol.add_trace(go.Scatter(
                x=rolling_var.index[window_size:],
                y=rolling_var[window_size:],
                name='Rolling Variance',
                line=dict(color='blue')
            ))
            fig_vol.update_layout(
                title=f'Rolling {selected_window} Metrics',
                xaxis_title='Date',
                yaxis_title='Value',
                template='plotly_white'
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        with tab2:
            daily_std = returns.std()
            weekly_std = returns.resample('W').std().dropna()
            monthly_std = returns.resample('M').std().dropna()

            x = np.linspace(-3, 3, 100)
            y = stats.norm.pdf(x, 0, 1)

            fig_std = go.Figure()
            fig_std.add_trace(go.Scatter(
                x=x,
                y=y,
                fill='tozeroy',
                name='Standard Normal Distribution',
                line=dict(color='blue')
            ))

            fig_std.add_vline(
                x=daily_std / daily_std,
                line_dash="dash",
                line_color="red",
                annotation_text="Daily"
            )
            fig_std.add_vline(
                x=weekly_std.mean() / daily_std if daily_std != 0 else 1,
                line_dash="dash",
                line_color="green",
                annotation_text="Weekly"
            )
            fig_std.add_vline(
                x=monthly_std.mean() / daily_std if daily_std != 0 else 1,
                line_dash="dash",
                line_color="orange",
                annotation_text="Monthly"
            )
            fig_std.update_layout(
                title='Relative Standard Deviation Ranges',
                xaxis_title='Standard Deviations',
                yaxis_title='Probability Density',
                template='plotly_white'
            )
            st.plotly_chart(fig_std, use_container_width=True)

        with tab3:
            mean = returns.mean()
            std = returns.std()
            x_norm = np.linspace(returns.min(), returns.max(), 1000)
            y_norm = stats.norm.pdf(x_norm, mean, std)
            y_skewed = stats.skewnorm.pdf(x_norm, skewness, mean, std)

            fig_skew = go.Figure()
            fig_skew.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Actual Returns',
                histnorm='probability density',
                opacity=0.7,
                marker_color='#94a3b8'
            ))
            fig_skew.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                name='Normal Distribution',
                line=dict(color='#2563eb', dash='dash')
            ))
            fig_skew.add_trace(go.Scatter(
                x=x_norm,
                y=y_skewed,
                name='Fitted Skewed Distribution',
                line=dict(color='#dc2626')
            ))
            fig_skew.add_vline(x=mean, line_dash="dash", line_color="#0f766e", annotation_text="Mean")
            fig_skew.add_vline(x=returns.median(), line_dash="dash", line_color="#0369a1", annotation_text="Median")
            fig_skew.update_layout(
                title=f'Return Distribution Analysis (Skewness: {skewness:.2f})',
                xaxis_title='Returns',
                yaxis_title='Density',
                template='plotly_white'
            )
            st.plotly_chart(fig_skew, use_container_width=True)

            st.markdown(f"""
            **Interpreting Skewness:**
            - Mean: {mean:.4f}
            - Median: {returns.median():.4f}
            - Skewness: {skewness:.4f} → {get_skewness_interpretation(skewness)}
            """)

            st.subheader("Rolling Skewness Analysis")
            skewness_window = st.selectbox(
                "Select Rolling Window for Skewness",
                options=list(window_options.keys()),
                key="skewness_window"
            )
            roll_window = window_options[skewness_window]
            rolling_skew = returns.rolling(window=roll_window).skew()

            fig_rolling_skew = go.Figure()
            fig_rolling_skew.add_trace(go.Scatter(
                x=rolling_skew.index,
                y=rolling_skew,
                name='Rolling Skewness',
                line=dict(color='#2563eb')
            ))
            fig_rolling_skew.add_hline(
                y=0,
                line_dash="dash",
                line_color="#475569",
                annotation_text="No Skewness"
            )
            fig_rolling_skew.update_layout(
                title=f'{skewness_window} Rolling Skewness Analysis',
                xaxis_title='Date',
                yaxis_title='Skewness',
                template='plotly_white'
            )
            st.plotly_chart(fig_rolling_skew, use_container_width=True)

        with tab4:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdowns.index,
                y=drawdowns,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red')
            ))
            fig_dd.update_layout(
                title='Historical Drawdowns',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white'
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        with tab5:
            rolling_skew = returns.rolling(window=30).skew()
            rolling_kurt = returns.rolling(window=30).kurt()
            rolling_beta = returns.rolling(window=30).cov(market_returns) / market_returns.rolling(window=30).var()

            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(
                x=rolling_skew.index[30:],
                y=rolling_skew[30:],
                name='Rolling Skewness'
            ))
            fig_risk.add_trace(go.Scatter(
                x=rolling_kurt.index[30:],
                y=rolling_kurt[30:],
                name='Rolling Kurtosis'
            ))
            fig_risk.add_trace(go.Scatter(
                x=rolling_beta.index[30:],
                y=rolling_beta[30:],
                name='Rolling Beta'
            ))
            fig_risk.update_layout(
                title='Rolling Risk Metrics (30-Day Window)',
                xaxis_title='Date',
                yaxis_title='Value',
                template='plotly_white'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.warning("Insufficient data for volatility analysis")


elif option == "Monte Carlo Simulation":
    st.markdown("""<h2>Monte Carlo Simulation</h2>""", unsafe_allow_html=True)
    
    with st.expander("Learn More About VaR and ES"):
        st.markdown("""
        - **Value at Risk (VaR):** Measures the potential loss in value of an investment over a given time period at a specific confidence level.  
        - **Expected Shortfall (ES):** Also known as Conditional VaR, it estimates the average loss in scenarios where the VaR threshold is exceeded, providing a more comprehensive risk measure.
        """)
    
    selected_scheme = st.selectbox("Select a Scheme", scheme_names.keys())
    scheme_code = scheme_names[selected_scheme]

    try:
        nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

        if not nav_data.empty:
            st.subheader("Monte Carlo Simulation Parameters")
            num_simulations = st.slider(
                "Number of Simulations",
                min_value=100,
                max_value=5000,
                value=1000
            )
            num_days = st.slider("Projection Period (days)", 30, 365, 252)
            confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95)

            if st.button("Run Simulation"):
                with st.spinner("Running Monte Carlo simulation..."):
                    results = run_monte_carlo_var(
                        nav_data,
                        selected_scheme,
                        num_simulations,
                        num_days,
                        confidence_level,
                    )

                    if results["figure"] is not None:
                        st.plotly_chart(results["figure"], use_container_width=True)
                        st.metric("VaR", f"{results['var']:.2f}%")
                        st.metric("Expected Shortfall", f"{results['expected_shortfall']:.2f}%")
                    else:
                        st.error("Simulation failed or returned empty results.")

        else:
            st.warning("No historical NAV data for the selected scheme.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

with open('codes.json', 'r') as f:
    fund_tickers = json.load(f)
fund_tickers = {k: v for d in fund_tickers for k, v in d.items()}

if option == "AMC News Analysis":
    st.markdown("""<h2>AMC News Analysis</h2>""", unsafe_allow_html=True)
    
    amc_name = st.text_input("Enter AMC Name (e.g., HDFC Mutual Fund, SBI Mutual Fund):", "")
    if amc_name:
        try:
            NEWS_API_KEY = "5783048c600b4affad0cb9304a507ae0"
            query = f"{amc_name} mutual fund"
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
            
            response = requests.get(url)
            news_data = response.json()

            if news_data["status"] == "ok" and news_data["articles"]:
                st.subheader(f"Latest News for {amc_name}")

                for article in news_data["articles"][:5]:
                    text_content = (article["title"] or "") + " " + (article["description"] or "")
                    sentiment = TextBlob(text_content).sentiment.polarity

                    if sentiment > 0.1:
                        sentiment_category = "🟢 Positive"
                        color = "green"
                    elif sentiment < -0.05:  # Made threshold less strict for negative sentiment
                        sentiment_category = "🔴 Negative" 
                        color = "red"
                    else:
                        # Only truly neutral content gets neutral category
                        if abs(sentiment) < 0.02:
                            sentiment_category = "⚪ Neutral"
                            color = "gray"
                        else:
                            # Slightly negative content is marked as negative
                            sentiment_category = "🔴 Negative"
                            color = "red"

                    with st.expander(f"{article['title']} ({sentiment_category})"):
                        st.write(f"**Published at:** {article['publishedAt'][:10]}")
                        st.write(f"**Source:** {article['source']['name']}")
                        st.write(f"**Sentiment:** :{color}[{sentiment_category}]")
                        st.write(f"**Description:** {article['description']}")
                        if article['urlToImage']:
                            st.image(article['urlToImage'], caption=article['title'])
                        st.markdown(f"[Read full article]({article['url']})")
            else:
                st.info(f"No recent news found for {amc_name}")
                        
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            st.info("Please ensure you have a valid NewsAPI key and check your internet connection.")

def get_nav_on_date(scheme_code, date, mf_client):
    """
    Helper function to get the NAV of a scheme on a specific date.
    scheme_code: str (AMFI scheme code)
    date: datetime object (the purchase date)
    mf_client: an instance of Mftool
    """
    try:
        # Format date as DD-MM-YYYY (as expected by MFTool)
        date_str = date.strftime("%d-%m-%Y")
        # Try to get NAV data as a DataFrame
        historical_nav = mf_client.get_scheme_historical_nav(scheme_code, date_str, date_str, as_Dataframe=True)
        
        # If no data for the exact date, try previous 30 days
        if historical_nav is None or historical_nav.empty:
            for i in range(1, 31):
                prev_date = date - pd.Timedelta(days=i)
                prev_date_str = prev_date.strftime("%d-%m-%Y")
                historical_nav = mf_client.get_scheme_historical_nav(scheme_code, prev_date_str, prev_date_str, as_Dataframe=True)
                if historical_nav is not None and not historical_nav.empty:
                    break
                    
        if historical_nav is not None and not historical_nav.empty:
            # Assuming the DataFrame has a column named 'nav'
            nav_value = float(historical_nav.iloc[0]['nav'])
            return nav_value, True
        else:
            return None, False
    except Exception as e:
        print(f"Error getting NAV for date {date}: {str(e)}")
        return None, False


# ---------------------------
# Upload Portfolio Section
# ---------------------------
if option == "Upload Portfolio":
    st.header("Upload Your Mutual Fund Portfolio")
    st.markdown("### Analyze Your Mutual Fund Portfolio Performance")
    with st.expander("Learn More About Portfolio Analysis"):
        st.markdown("""
        - Evaluate diversification.
        - Compare performance with benchmarks.
        - Assess risk exposure and rebalancing opportunities.
        """)
    
    # File Uploader (CSV or Excel)
    uploaded_file = st.file_uploader("Upload your portfolio (CSV or Excel)", type=["csv", "xlsx"])
    
    # Provide a sample template download
    st.download_button(
        "Download Sample Template",
        data=pd.DataFrame({
            'Scheme Name': ['ICICI Prudential Bluechip Fund - Direct Plan', 'SBI Small Cap Fund - Direct Plan'],
            'Scheme Code': ['120533', '136451'],
            'Units Held': [100, 50],
            'Purchase Date': ['31-05-2019', '01-06-2019']  # DD-MM-YYYY format
        }).to_csv(index=False),
        file_name="sample_portfolio_template.csv",
        mime="text/csv"
    )
    
    # Add explicit instructions about date format
    st.markdown("""
    **Important Instructions:**
    1. Date Format: 
       - Use DD-MM-YYYY format (e.g., 31-05-2019 for 31st May 2019)
       - Use hyphens (-) or forward slashes (/) as separators
       - Always use two digits for day and month (01-05-2019, not 1-5-2019)
    2. Required Columns:
       - Scheme Name
       - Scheme Code
       - Units Held
       - Purchase Date
    3. You can find AMFI scheme codes using our "View Available Schemes" option
    """)
    
    if uploaded_file is not None:
        try:
            # Load the file into a DataFrame
            if uploaded_file.name.endswith('.csv'):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)
            
            # Basic validation
            required_cols = {"Scheme Name", "Scheme Code", "Units Held", "Purchase Date"}
            missing_cols = required_cols - set(user_df.columns)
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            st.subheader("Your Uploaded Portfolio")
            st.dataframe(user_df)
            
            with st.spinner("Processing your portfolio..."):
                # Data type conversion and cleaning
                user_df["Scheme Code"] = user_df["Scheme Code"].astype(str).str.strip()
                user_df["Units Held"] = pd.to_numeric(user_df["Units Held"], errors='coerce')
                
                # Convert dates using a specific format
                def parse_date(date_str):
                    try:
                        # Try parsing with DD-MM-YYYY format
                        return pd.to_datetime(date_str, format='%d-%m-%Y')
                    except:
                        try:
                            # Try parsing with DD/MM/YYYY format
                            return pd.to_datetime(date_str, format='%d/%m/%Y')
                        except:
                            return pd.NaT

                user_df["Purchase Date"] = user_df["Purchase Date"].apply(parse_date)
                
                # Check for invalid dates
                invalid_dates = user_df[user_df["Purchase Date"].isna()]
                if not invalid_dates.empty:
                    st.error("Error parsing the following dates. Please ensure they are in DD-MM-YYYY format:\n" +
                             "\n".join([f"Row {i+1}: {row['Purchase Date']}" for i, row in invalid_dates.iterrows()]))
                    st.stop()
                
                # Remove rows with invalid data
                valid_portfolio = user_df.dropna(subset=["Units Held", "Scheme Code", "Purchase Date"]).copy()
                if valid_portfolio["Purchase Date"].isna().any():
                    st.error("Some dates could not be parsed. Please ensure all dates are in DD-MM-YYYY format.")
                    st.stop()
                
                # Initialize result columns
                valid_portfolio["Current NAV"] = np.nan
                valid_portfolio["Current Value"] = np.nan
                valid_portfolio["Purchase NAV"] = np.nan
                valid_portfolio["Purchase Value"] = np.nan
                valid_portfolio["Gain/Loss %"] = np.nan
                
                # Process each scheme
                for idx, row in valid_portfolio.iterrows():
                    scheme_code = row["Scheme Code"]
                    units = row["Units Held"]
                    purchase_date = row["Purchase Date"]
                    
                    try:
                        # Get current NAV using mf.get_scheme_quote
                        current_nav_data = mf.get_scheme_quote(scheme_code)
                        if current_nav_data and isinstance(current_nav_data, dict) and 'nav' in current_nav_data:
                            current_nav = float(current_nav_data['nav'])
                        else:
                            # Fallback: try latest NAV from historical data
                            latest_nav = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
                            if latest_nav is not None and not latest_nav.empty:
                                current_nav = float(latest_nav.iloc[0]['nav'])
                            else:
                                continue
                        
                        # Get purchase date NAV using historical data
                        historical_nav = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
                        if historical_nav is not None and not historical_nav.empty:
                            # Convert index to datetime if needed
                            if not isinstance(historical_nav.index, pd.DatetimeIndex):
                                historical_nav.index = pd.to_datetime(historical_nav.index)
                            # Find the closest date to the purchase date
                            closest_date = min(historical_nav.index, key=lambda x: abs(x - purchase_date))
                            purchase_nav = float(historical_nav.loc[closest_date, 'nav'])
                            
                            # Calculate values
                            current_value = units * current_nav
                            purchase_value = units * purchase_nav
                            gain_loss = ((current_value - purchase_value) / purchase_value) * 100
                            
                            # Update the DataFrame
                            valid_portfolio.loc[idx, "Current NAV"] = current_nav
                            valid_portfolio.loc[idx, "Current Value"] = current_value
                            valid_portfolio.loc[idx, "Purchase NAV"] = purchase_nav
                            valid_portfolio.loc[idx, "Purchase Value"] = purchase_value
                            valid_portfolio.loc[idx, "Gain/Loss %"] = gain_loss
                        else:
                            st.warning(f"No historical NAV data found for {row['Scheme Name']}")
                            continue
                        
                    except Exception as e:
                        st.warning(f"Error processing {row['Scheme Name']}: {str(e)}")
                        continue
                
                # Remove rows where processing failed
                valid_portfolio = valid_portfolio.dropna(subset=["Current NAV"]).copy()
                
                if len(valid_portfolio) > 0:
                    st.subheader("Portfolio Analysis Results")
                    display_df = valid_portfolio.copy()
                    display_df["Current NAV"] = display_df["Current NAV"].apply(lambda x: f"₹{x:,.2f}")
                    display_df["Current Value"] = display_df["Current Value"].apply(lambda x: f"₹{x:,.2f}")
                    display_df["Purchase NAV"] = display_df["Purchase NAV"].apply(lambda x: f"₹{x:,.2f}")
                    display_df["Purchase Value"] = display_df["Purchase Value"].apply(lambda x: f"₹{x:,.2f}")
                    display_df["Gain/Loss %"] = display_df["Gain/Loss %"].apply(lambda x: f"{x:,.2f}%")
                    st.dataframe(display_df)
                    
                    # Calculate portfolio summary
                    total_investment = valid_portfolio["Purchase Value"].sum()
                    total_current = valid_portfolio["Current Value"].sum()
                    overall_return = ((total_current - total_investment) / total_investment) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Investment", f"₹{total_investment:,.2f}")
                    with col2:
                        st.metric("Current Value", f"₹{total_current:,.2f}")
                    with col3:
                        st.metric("Overall Return", f"{overall_return:.2f}%",
                                  delta=f"{overall_return:.2f}%",
                                  delta_color="normal" if overall_return >= 0 else "inverse")
                else:
                    st.error("Could not process any schemes in your portfolio. Please check the scheme codes.")
        
        except Exception as e:
            st.error(f"Error processing portfolio: {str(e)}")
    
    # ---------- Portfolio Evaluation Parameters and Analysis ----------
    st.subheader("Portfolio Evaluation Parameters")
    analysis_method = st.selectbox("Choose Analysis Method", ["CAPM", "Modern Portfolio Theory"])
    risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 6.0, step=0.1)
    market_return = st.number_input("Expected Market Return (%)", 0.0, 20.0, 10.0, step=0.5)
    target_return = st.number_input("Target Annual Return (%)", 0.0, 50.0, 12.0, step=0.5)
    max_volatility = st.number_input("Maximum Acceptable Volatility (%)", 0.0, 50.0, 15.0, step=0.5)
    
    if st.button("Evaluate Portfolio"):
        scheme_data_list = []
        # Compute normalized allocation (as a fraction, not percentage)
        total_current_value = valid_portfolio["Current Value"].sum()
        valid_portfolio["Normalized Allocation"] = valid_portfolio["Current Value"] / total_current_value
        
        for idx, row in valid_portfolio.iterrows():
            scheme_code = row["Scheme Code"]
            scheme_name = row["Scheme Name"]
            allocation = row["Normalized Allocation"]  # fraction between 0 and 1
            
            scheme_info = mf.get_scheme_details(scheme_code)
            if not scheme_info:
                st.warning(f"Cannot fetch details for {scheme_name}. Skipping.")
                continue
            try:
                beta = float(scheme_info.get("beta", 1.0))
            except:
                beta = 1.0
            
            # Fetch historical NAV data for the scheme for return estimation
            nav_data_scheme, success = analyze_historical_nav(scheme_code, mf)
            if success and nav_data_scheme is not None and not nav_data_scheme.empty:
                daily_rets = nav_data_scheme["Daily_Returns"].dropna()
                annual_return_est = daily_rets.mean() * 252 * 100  # in percentage
                annual_vol_est = daily_rets.std() * np.sqrt(252) * 100  # in percentage
                capm_return = risk_free_rate + beta * (market_return - risk_free_rate)
                
                scheme_data_list.append({
                    "Scheme Name": scheme_name,
                    "Beta": beta,
                    "Allocation": allocation,
                    "Historical Return (%)": annual_return_est,
                    "Volatility (%)": annual_vol_est,
                    "CAPM Return (%)": capm_return
                })
            else:
                st.warning(f"Insufficient NAV data for {scheme_name}. Skipping.")
        
        if not scheme_data_list:
            st.error("No schemes could be evaluated with the given data.")
        else:
            analysis_df = pd.DataFrame(scheme_data_list)
            st.write("Initial Analysis of Each Scheme:")
            st.dataframe(analysis_df)
            
            if analysis_method == "CAPM":
                portfolio_beta = (analysis_df["Beta"] * analysis_df["Allocation"]).sum()
                portfolio_capm_return = risk_free_rate + portfolio_beta * (market_return - risk_free_rate)
                weighted_hist_return = (analysis_df["Historical Return (%)"] * analysis_df["Allocation"]).sum()
                weighted_vol = (analysis_df["Volatility (%)"] * analysis_df["Allocation"]).sum()
                
                st.metric("Portfolio Beta", f"{portfolio_beta:.2f}")
                st.metric("Portfolio CAPM Return (%)", f"{portfolio_capm_return:.2f}%")
                st.metric("Weighted Historical Return (%)", f"{weighted_hist_return:.2f}%")
                st.metric("Weighted Volatility (%)", f"{weighted_vol:.2f}%")
                
                alpha = weighted_hist_return - portfolio_capm_return
                st.metric("Portfolio Alpha (%)", f"{alpha:.2f}%")
            else:
                st.info("Modern Portfolio Theory optimization is not implemented in this snippet. Please implement full mean-variance optimization as needed.")
            
            if weighted_hist_return < target_return or weighted_vol > max_volatility:
                st.warning("Your current portfolio does not meet your target risk/return preferences. Consider adjusting allocations or adding other funds.")
            else:
                st.success("Your portfolio meets your specified risk/return preferences.")

# Initialize mftool instance
mf = Mftool()

@st.cache_data
def recommend_mutual_funds(
    _mf,
    fund_type: str,
    growth_option: str,
    target_annual_return: float,
    max_volatility: float,
    top_n: int = 15
):
    """
    Recommend mutual funds using an optimized approach by pre-filtering
    and limiting the historical data processed per scheme. Returns only
    those schemes that meet the user-defined return and volatility criteria,
    limited to a maximum of 'top_n' funds.
    """
    try:
        available_schemes = _mf.get_available_schemes(amc_name="")
        if not available_schemes:
            return {
                "data": pd.DataFrame(),
                "skipped": 0,
                "error": "No available schemes found."
            }

        # Filter based on user input
        filtered_schemes = {
            scheme_code: scheme_name
            for scheme_code, scheme_name in available_schemes.items()
            if (fund_type == "All" or fund_type.lower() in scheme_name.lower())
            and (growth_option == "All" or growth_option.lower() in scheme_name.lower())
        }
        if not filtered_schemes:
            return {
                "data": pd.DataFrame(),
                "skipped": 0,
                "error": "No schemes match the selected Fund Type and Growth Option."
            }

        recommendations = []
        skipped_schemes = 0

        def process_scheme(scheme_code, scheme_name):
            """
            Fetch the scheme's historical NAV, calculate annual return and volatility,
            and return the fund info if it meets user criteria.
            """
            try:
                nav_data = _mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
                if nav_data is None or nav_data.empty:
                    return None

                nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
                nav_data = nav_data.dropna(subset=["nav"])
                if len(nav_data) < 30:
                    return None

                # Limit to most recent 500 records
                if len(nav_data) > 500:
                    nav_data = nav_data.tail(500)

                # Calculate daily returns
                nav_data["Daily_Returns"] = nav_data["nav"].pct_change().dropna()
                if nav_data["Daily_Returns"].empty:
                    return None

                annual_return = nav_data["Daily_Returns"].mean() * 252 * 100
                annual_volatility = nav_data["Daily_Returns"].std() * np.sqrt(252) * 100

                # Filter by user criteria
                if annual_return >= target_annual_return and annual_volatility <= max_volatility:
                    return {
                        "Scheme Name": scheme_name,
                        "Scheme Code": scheme_code,
                        "CAGR (%)": round(annual_return, 2),
                        "Volatility (%)": round(annual_volatility, 2),
                        "Fund Type": fund_type,
                        "Growth Option": growth_option
                    }
                else:
                    return None
            except Exception as e:
                return None

        # Run in parallel
        max_workers = min(32, len(filtered_schemes))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_scheme, code, name): code
                for code, name in filtered_schemes.items()
            }
            for future in concurrent.futures.as_completed(future_map):
                res = future.result()
                if res:
                    recommendations.append(res)
                else:
                    skipped_schemes += 1

        # Sort by CAGR descending
        recommendations = sorted(
            recommendations,
            key=lambda x: x["CAGR (%)"],
            reverse=True
        )

        # Limit to at most 'top_n' funds
        recommendations = recommendations[:top_n]

        # Convert to DataFrame
        if recommendations:
            df = pd.DataFrame(recommendations)
        else:
            df = pd.DataFrame()

        return {
            "data": df,
            "skipped": skipped_schemes
        }

    except Exception as e:
        return {
            "data": pd.DataFrame(),
            "skipped": 0,
            "error": f"Error in recommendation process: {str(e)}"
        }

def allocate_weights(recommendations_df):
    """
    Allocate weights among recommended funds based on a simple
    risk-adjusted metric: CAGR / (Volatility + eps).
    """
    recommendations_df["CAGR (%)"] = recommendations_df["CAGR (%)"].astype(float)
    recommendations_df["Volatility (%)"] = recommendations_df["Volatility (%)"].astype(float)

    eps = 1e-6
    recommendations_df["Score"] = (
        recommendations_df["CAGR (%)"] / (recommendations_df["Volatility (%)"] + eps)
    )

    total_score = recommendations_df["Score"].sum()
    if total_score == 0:
        n_funds = len(recommendations_df)
        recommendations_df["Allocation (%)"] = 100.0 / n_funds
    else:
        recommendations_df["Allocation (%)"] = (
            recommendations_df["Score"] / total_score
        ) * 100.0

    recommendations_df.drop(columns=["Score"], inplace=True)
    return recommendations_df

def curated_mutual_fund_basket(mf, option):
    """
    Renders the Curated Mutual Fund Basket UI and handles the logic.
    This function is called when option equals "Curated Mutual Fund Basket".
    """
    if option != "Curated Mutual Fund Basket":
        return

    st.markdown("<h2>Curated Mutual Fund Basket</h2>", unsafe_allow_html=True)
    st.info(
        "Select your investment preferences and constraints below to receive a "
        "tailored basket of mutual funds that align with your financial goals "
        "and risk appetite."
    )

    st.subheader("Investment and Risk Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)", 
            min_value=0.0, max_value=10.0, value=6.0, step=0.1
        )
        expected_market_return = st.number_input(
            "Expected Market Return (%)", 
            min_value=0.0, max_value=20.0, value=10.0, step=0.1
        )

    with col2:
        target_annual_return = st.number_input(
            "Target Annual Return (%)", 
            min_value=0.0, max_value=50.0, value=12.0, step=0.1
        )
        max_volatility = st.number_input(
            "Maximum Acceptable Volatility (%)", 
            min_value=0.0, max_value=50.0, value=15.0, step=0.1
        )

    with col3:
        investment_amount = st.number_input(
            "Investment Amount (₹)", 
            min_value=1000, max_value=1_000_000, value=100000, step=1000
        )
        fund_type = st.selectbox("Fund Type", ["All", "Equity", "Debt", "Hybrid", "Other"])

    with col4:
        growth_option = st.selectbox("Growth Option", ["All", "Growth", "Dividend", "Other"])
        # New parameter for number of funds in the basket:
        num_funds = st.number_input(
            "Number of Funds in Basket", min_value=1, max_value=30, value=15, step=1
        )

    if st.button("Get Recommended Basket"):
        with st.spinner("Analyzing and curating your mutual fund basket..."):
            results = recommend_mutual_funds(
                _mf=mf,
                fund_type=fund_type,
                growth_option=growth_option,
                target_annual_return=target_annual_return,
                max_volatility=max_volatility,
                top_n=num_funds
            )

            if "error" in results and results["error"]:
                st.error(results["error"])
                st.stop()

            recommendations_df = results.get("data", pd.DataFrame())
            skipped_schemes = results.get("skipped", 0)

            if recommendations_df.empty:
                st.warning("No mutual funds found matching your criteria. Try adjusting your filters.")
            else:
                st.success(f"Found {len(recommendations_df)} mutual funds matching your criteria.")

                # Allocate based on risk-adjusted metric
                recommendations_df = allocate_weights(recommendations_df)

                # Compute rupee allocation for each scheme
                recommendations_df["Investment Amount (₹)"] = (
                    investment_amount * recommendations_df["Allocation (%)"] / 100
                )

                st.markdown("### Recommended Mutual Fund Basket")
                st.dataframe(
                    recommendations_df[[
                        "Scheme Name", "Scheme Code", "CAGR (%)", "Volatility (%)",
                        "Fund Type", "Growth Option", "Allocation (%)", "Investment Amount (₹)"
                    ]],
                    use_container_width=True
                )

                # Pie chart visualization
                fig_pie = px.pie(
                    recommendations_df,
                    names="Scheme Name",
                    values="Allocation (%)",
                    title="Investment Allocation per Scheme",
                    template='plotly_white'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Download option
                csv_recommendations = recommendations_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommended Basket",
                    data=csv_recommendations,
                    file_name="recommended_mutual_fund_basket.csv",
                    mime="text/csv"
                )

                if skipped_schemes > 0:
                    st.info(f"Skipped {skipped_schemes} schemes due to insufficient data or errors.")

# --------------------
# Main Application
# --------------------


if option == "Curated Mutual Fund Basket":
    curated_mutual_fund_basket(mf, option)


st.markdown(
    """
    ---
    <div style="text-align:center; font-size:0.9em; color:#666;">
        <em>Disclaimer: The information provided in this application is for 
        educational purposes only and should not be construed as financial advice. 
        Always do your own research or consult a professional before investing.</em>
    </div>
    """,
    unsafe_allow_html=True
)