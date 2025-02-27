# Mutual-Fund-Analytical-Dashboard
Below is a detailed README.md that you can include in your repository’s root folder:

---

# Mutual Fund Technical Analysis Dashboard

Welcome to the **Mutual Fund Technical Analysis Dashboard** – an advanced, interactive web application built in Python with Streamlit. This dashboard is designed to provide comprehensive insights into the Indian mutual fund market, offering detailed analysis tools for investors of all levels. Whether you are a novice or an experienced investor, this tool empowers you with historical data analysis, risk assessment, portfolio evaluation, and much more.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Navigating the Dashboard](#navigating-the-dashboard)
  - [Key Functionalities](#key-functionalities)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The Mutual Fund Technical Analysis Dashboard leverages real-time and historical financial data to help you analyze and monitor mutual fund performance. The application integrates multiple analytical modules, including risk metrics, Monte Carlo simulations, returns calculators, and technical analysis tools. It is built using a combination of powerful Python libraries such as Streamlit, Pandas, NumPy, Plotly, and several financial and web-scraping libraries.

## Features

- **Interactive Web Interface:**  
  Enjoy a modern, dark blue-themed user interface with intuitive navigation and responsive design.

- **Scheme Management:**  
  - **View Available Schemes:** Filter mutual fund schemes by AMC name, type (Equity, Debt, Hybrid, etc.), and growth options (Direct Growth, IDCW, Regular).
  - **Scheme Details:** Access in-depth information about each scheme, including fund house, scheme type, category, launch date, and initial NAV.

- **Historical NAV Analysis:**  
  - Visualize historical Net Asset Value (NAV) trends using interactive line charts.
  - Analyze daily and cumulative returns.
  - Download datasets (both filtered and complete) in CSV format for further analysis.

- **Assets Under Management (AUM) Analysis:**  
  - Compute average AUM for selected quarters.
  - Compare quarter-over-quarter changes.
  - Visualize sector allocations and overall AUM distribution with bar charts and histograms.

- **Risk and Return Evaluation:**  
  - Calculate key metrics such as CAGR, XIRR, Sharpe, Sortino, and Treynor ratios.
  - Display risk–return scatter plots for performance evaluation.
  
- **Investment Returns Calculators:**  
  - **Lumpsum Calculator:** Estimate future investment values, adjust for inflation, and calculate break-even points.
  - **SIP Calculator:** Assess systematic investment plans by modeling expected growth trajectories and risk metrics.

- **Technical Analysis Tools:**  
  - **Moving Averages Analysis:** Implement 50-day and 100-day moving averages to generate buy/sell signals and identify trend crossovers.
  - **Volatility Analysis:** Evaluate risk through annualized volatility, skewness, kurtosis, and maximum drawdown. Enjoy detailed rolling analyses via interactive charts.
  - **Monte Carlo Simulation:** Simulate future mutual fund returns using a log-return based geometric Brownian motion model to estimate Value at Risk (VaR) and Expected Shortfall (ES).

- **AMC News Analysis:**  
  Fetch the latest news articles related to asset management companies (AMCs) using external news APIs, providing real-time market insights.

- **Portfolio Analysis and Curated Mutual Fund Basket:**  
  - **Portfolio Analysis:** Upload your portfolio (Excel/CSV) to analyze individual holdings, assess overall risk-return profiles, and verify alignment with investment targets.
  - **Curated Mutual Fund Basket:** Receive personalized mutual fund recommendations based on user-defined constraints such as risk tolerance, expected returns, and investment horizon, powered by multi-threaded evaluation.

## Installation

### Prerequisites

Ensure you have the following installed:
- **Python 3.7+**
- [Git](https://git-scm.com/)

### Required Python Packages

All necessary dependencies are listed in the `requirements.txt` file. A sample requirements file for this project is:

```
streamlit
pandas
numpy
plotly
python-dateutil
requests
beautifulsoup4
textblob
numpy-financial
scipy
yahooquery
openpyxl
mftool
```

### Steps to Install

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/mutual-fund-analytical-dashboard.git
   cd mutual-fund-analytical-dashboard
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Deploy on Streamlit Cloud (Optional):**  
   Ensure the repository contains the `requirements.txt` file so that Streamlit Cloud can automatically install dependencies.

## Usage Guide

### Navigating the Dashboard

When you run the application using Streamlit, you’ll be greeted with a sidebar listing all available functionalities. Use the sidebar to switch between different modules:

- **View Available Schemes**
- **Scheme Details**
- **Historical NAV**
- **Average AUM**
- **Risk and Return Analysis**
- **Returns Calculator**
- **Moving Averages**
- **Volatility Analysis**
- **Monte Carlo Simulation**
- **AMC News Analysis**
- **Upload Portfolio & Curated Mutual Fund Basket**

### Key Functionalities

1. **View Available Schemes:**  
   Filter mutual fund schemes by AMC, type, and growth options. The results are displayed in an interactive table with download options.

2. **Scheme Details:**  
   Select a scheme to view detailed information such as fund house, category, and launch details.

3. **Historical NAV Analysis:**  
   - Choose a scheme and define a date range.
   - Visualize NAV trends and cumulative returns.
   - Download the data for offline analysis.

4. **Average AUM Analysis:**  
   - Select a quarter and optionally compare with previous quarters.
   - View metrics, sector allocation, and visual distributions.
   - Insights are provided to help understand market concentration and fund performance.

5. **Risk and Return Analysis:**  
   - Evaluate historical performance with metrics like CAGR, XIRR, Sharpe, Sortino, and Beta.
   - Explore risk–return scatter plots and detailed rolling analyses.

6. **Returns Calculator:**  
   - **Lumpsum Calculator:** Input your investment amount, period, and expected return to see projected values, inflation-adjusted figures, and risk metrics.
   - **SIP Calculator:** Calculate the growth of your systematic investments along with detailed metrics.

7. **Moving Averages Analysis:**  
   - Visualize 50-day and 100-day moving averages.
   - Identify buy/sell signals from crossover points.
   - Detailed charts help in understanding trend shifts.

8. **Volatility Analysis:**  
   - Explore volatility measures such as standard deviation, skewness, kurtosis, and drawdowns.
   - Interactive charts provide rolling analyses over various time windows.

9. **Monte Carlo Simulation:**  
   - Configure simulation parameters including number of simulations, projection period, and confidence level.
   - Run simulations to estimate VaR and Expected Shortfall with dynamic visual feedback.

10. **AMC News Analysis:**  
    - Enter an AMC name to fetch the latest news articles.
    - Stay updated with real-time market insights directly from reliable news sources.

11. **Portfolio Analysis and Curated Mutual Fund Basket:**  
    - **Portfolio Analysis:** Upload your portfolio file to evaluate your current investment's risk-return profile.
    - **Curated Mutual Fund Basket:** Get personalized mutual fund recommendations based on your investment criteria and risk tolerance.

## Configuration

- **Environment Variables:**  
  If your application requires API keys (for example, for news analysis), set them as environment variables or configure them in a secure manner.
- **Custom Styling:**  
  The application uses custom CSS to maintain a consistent dark blue theme. Modify the CSS in the `st.markdown` section if needed.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Submit a pull request detailing your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions, feedback, or support, please open an issue on GitHub or contact (rudragupta0907@gmail.com).

---
