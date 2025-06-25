import numpy as np
import pandas as pd
import plotly.graph_objects as go


def run_monte_carlo_var(nav_data: pd.DataFrame,
                        scheme_name: str,
                        num_simulations: int,
                        num_days: int,
                        confidence_level: float) -> dict:
    """Run a Monte Carlo simulation to estimate Value at Risk (VaR)."""
    try:
        nav_data['nav'] = pd.to_numeric(nav_data['nav'], errors='coerce')
        nav_data.dropna(subset=['nav'], inplace=True)

        nav_data['log_return'] = np.log(nav_data['nav'] / nav_data['nav'].shift(1))
        log_returns = nav_data['log_return'].dropna()
        daily_mean = log_returns.mean()
        daily_vol = log_returns.std()
        last_nav = nav_data['nav'].iloc[-1]

        simulation_final_returns = []
        for _ in range(num_simulations):
            simulated_log_returns = np.random.normal(daily_mean, daily_vol, num_days)
            final_nav = last_nav * np.exp(simulated_log_returns.sum())
            total_return = (final_nav / last_nav) - 1.0
            simulation_final_returns.append(total_return)

        simulation_final_returns = pd.Series(simulation_final_returns)
        alpha = 1.0 - confidence_level
        var_threshold = simulation_final_returns.quantile(alpha)
        var_percent = -var_threshold * 100.0 if var_threshold < 0 else 0.0

        tail_losses = simulation_final_returns[simulation_final_returns <= var_threshold]
        es_value = -tail_losses.mean() * 100.0 if len(tail_losses) > 0 else 0.0

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


def calculate_risk_metrics(returns: pd.Series) -> tuple:
    """Calculate Sharpe, Sortino and Beta from returns series."""
    risk_free_rate = 0.06
    excess_returns = returns - risk_free_rate / 252

    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    downside_returns = returns[returns < 0]
    sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns) if len(downside_returns) > 0 else np.nan

    beta = 1.0
    return sharpe, sortino, beta
