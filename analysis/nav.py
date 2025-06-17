import pandas as pd
from mftool import Mftool


def analyze_historical_nav(scheme_code: str, mf_instance: Mftool) -> tuple:
    """Fetch and process historical NAV data for a scheme."""
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
                except Exception:
                    continue
            try:
                return pd.to_datetime(date_series, dayfirst=True)
            except Exception:
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
        print(f"Error in analyze_historical_nav: {e}")
        return None, False
