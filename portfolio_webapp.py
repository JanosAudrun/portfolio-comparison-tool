import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import numpy as np

# Set page configuration
st.set_page_config(layout="wide")

# Set up the page layout
st.title("Janko's Quick&Dirty Portfolio Comparison Tool")
st.sidebar.header("Portfolio Settings")

# Adjusted section with font size changes
st.markdown("### Description and Instructions") 

#Description and Instructions Expander
with st.expander("Click to expand"):
# Create two columns for Introduction and Instructions
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Description
        This tool provides analysis of two portfolios, including:

        - **Cumulative Returns**
        - **Drawdown**
        - **Return Contributions**
        - **Ticker Cumulative Returns**
        - **Volatility**
        - **Returns Table**
                    
        #### Changelog:
        <a href="http://janko.work/portfolio-comparison/changelog" target="_blank">View Changelog</a>
        """,
        unsafe_allow_html=True,
        )

    with col2:
        st.markdown("""
        #### Instructions
        1. Enter tickers and weights for both portfolios in the sidebar.
        2. Choose data frequency: daily, weekly or monthly
        3. Set a date range for analysis.
        4. Hit **Send** to generate the results.

        #### Tips:
        - Expand each graph by hovering on it and clicking the expand button in the top right.
        - Download the table in CSV format by hovering and clicking the download icon in the top right.
        """)
       

# Sidebar for Portfolio 1
st.sidebar.subheader("Portfolio 1")
tickers1 = st.sidebar.text_area("Tickers (comma-separated)", value="SPY,AGG", key="tickers1")
weights1 = st.sidebar.text_area("Weights (comma-separated, must sum to 100%)", value="60,40", key="weights1")

# Sidebar for Portfolio 2
st.sidebar.subheader("Portfolio 2")
tickers2 = st.sidebar.text_area("Tickers (comma-separated)", value="SPY,AGG,BTC-USD", key="tickers2")
weights2 = st.sidebar.text_area("Weights (comma-separated, must sum to 100%)", value="58,40,2", key="weights2")

# Sidebar for frequency selection
frequency = st.sidebar.selectbox(
    "Select Data Frequency",
    options=["Daily", "Weekly", "Monthly"],
    index=0  # Default to "Daily"
)

# Sidebar for date range selection
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-11-22"))

# Validate date range
if start_date >= end_date:
    st.sidebar.error("End date must be after start date.")

# Validate weights
def validate_weights(weights, num_tickers):
    try:
        weights = [float(w) for w in weights.split(",")]
        if len(weights) != num_tickers:
            st.sidebar.error(f"Number of weights ({len(weights)}) must match number of tickers ({num_tickers}).")
            return None
        if sum(weights) != 100:
            st.sidebar.error("Weights must sum to 100%.")
            return None
        return [w / 100 for w in weights]  # Normalize to fractions
    except:
        st.sidebar.error("Invalid weights format.")
        return None

# Function to fetch data for multiple tickers and align by common dates
def fetch_data(tickers, start_date, end_date, frequency):
    tickers = [t.strip().upper() for t in tickers.split(",")]
    try:
        # Fetch data for all tickers at once
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

        if isinstance(data, pd.Series):  # Single ticker case
            data = data.to_frame(name=tickers[0])
    
        # Resample data based on frequency
        if frequency == "Weekly":
            data = data.resample('W').last()
        elif frequency == "Monthly":
            data = data.resample('M').last()
        
        # Drop rows where any ticker has NaN values (align by common dates)
        aligned_data = data.dropna()
        
        return aligned_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_portfolio_metrics(data, weights):
    """
    Calculate individual returns, cumulative returns, and contributions for a portfolio.
    :param data: DataFrame of adjusted close prices.
    :param weights: List of portfolio weights.
    :return: DataFrame with calculated metrics.
    """
    metrics = pd.DataFrame(index=data.index)
    daily_returns = data.pct_change().fillna(0)  # Daily returns

    # Calculate individual and portfolio cumulative returns
    metrics["Portfolio Return"] = daily_returns.dot(weights)
    metrics["Portfolio Cumulative Return"] = (1 + metrics["Portfolio Return"]).cumprod() - 1

    for i, ticker in enumerate(data.columns):
        metrics[f"{ticker} Return"] = daily_returns[ticker]
        metrics[f"{ticker} Cumulative Return"] = (1 + daily_returns[ticker]).cumprod() - 1
        metrics[f"{ticker} Contribution"] = metrics[f"{ticker} Return"] * weights[i]
        metrics[f"{ticker} Cumulative Contribution"] = (1 + metrics[f"{ticker} Contribution"]).cumprod() - 1

    return metrics

# Calculate portfolio returns
def calculate_portfolio(data, weights):
    # Ensure weights align with columns in data
    weights = pd.Series(weights, index=data.columns)
    daily_returns = data.pct_change().fillna(0)  # Handle NaNs or missing data
    portfolio_returns = daily_returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    return portfolio_returns, cumulative_returns

# Calculate portfolio drawdowns using metrics DataFrame
def calculate_drawdown(metrics, portfolio_column):
    """
    Calculate drawdown for a portfolio.
    :param metrics: DataFrame containing cumulative portfolio returns.
    :param portfolio_column: Name of the column with portfolio cumulative returns.
    :return: Series with drawdown values.
    """
    cumulative_returns = metrics[portfolio_column]

    peak = cumulative_returns.cummax()  # Identify peaks
    drawdown = cumulative_returns - peak  # Calculate drawdown
    return drawdown

# Calculate volatility with dynamic frequency adjustment
def calculate_volatility(data, weights=None, frequency="Daily"):
    # Determine annualization factor
    if frequency == "Daily":
        annualization_factor = 252
    elif frequency == "Weekly":
        annualization_factor = 52
    elif frequency == "Monthly":
        annualization_factor = 12
    else:
        raise ValueError("Invalid frequency")

    # Daily (or adjusted) returns for each ticker
    returns = data.pct_change().dropna()

    # Ticker volatilities (annualized)
    ticker_volatility = returns.std() * (annualization_factor ** 0.5)

    # Portfolio volatility (if weights are provided, annualized)
    portfolio_volatility = None
    if weights is not None:
        portfolio_volatility = (returns.dot(weights)).std() * (annualization_factor ** 0.5)

    return ticker_volatility, portfolio_volatility

# Refactored function to plot individual ticker cumulative returns using precomputed metrics
def plot_ticker_cumulative_returns(metrics, color_map, title):
    """
    Plot cumulative returns for each ticker using precomputed metrics.

    :param metrics: DataFrame containing precomputed metrics, including cumulative returns.
    :param color_map: Dictionary mapping tickers to their colors.
    :param title: Title for the chart.
    """
    fig, ax = plt.subplots(figsize=(16, 10))  # Larger chart size
    fig.patch.set_facecolor('#212E31')  # Background around chart
    ax.set_facecolor('#212E31')  # Chart background

    # Filter columns for cumulative returns excluding the portfolio
    cumulative_return_cols = [
        col for col in metrics.columns if "Cumulative Return" in col and "Portfolio" not in col
    ]

    for col in cumulative_return_cols:
        ticker = col.split()[0]  # Extract the ticker name
        ax.plot(
            metrics.index,
            metrics[col],
            label=f"{ticker} Cumulative Return",
            linestyle='-',
            color=color_map.get(ticker, "#FFFFFF")  # Use color from unified color_map or default to white
        )

    # Set axis labels, title, and format ticks
    ax.set_xlim(left=metrics.index.min())  # Ensure the first point starts at the Y axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Quarterly ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Date format

    ax.set_xlabel('Date', color='#FFFFFF', fontsize=16)
    ax.set_ylabel('Cumulative Return', color='#FFFFFF', fontsize=16)
    ax.set_title(title, color='#FFFFFF', fontsize=20)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Convert to percentage
    ax.tick_params(axis='both', colors='#FFFFFF', labelsize=14)  # Increase tick font size
    ax.legend(
        facecolor='#212E31',
        edgecolor='#FFFFFF',
        framealpha=0.5,
        labelcolor='#FFFFFF',
        fontsize=14
    )  # Larger legend text
    ax.spines['top'].set_color('#FFFFFF')
    ax.spines['bottom'].set_color('#FFFFFF')
    ax.spines['left'].set_color('#FFFFFF')
    ax.spines['right'].set_color('#FFFFFF')
    ax.grid(True, color='#FFFFFF', alpha=0.3)  # Light grid
    plt.tight_layout()
    return fig

# Refactored function to plot individual ticker contributions using precomputed metrics
def plot_ticker_contributions(metrics, color_map, title):
    """
    Plot cumulative contributions for each ticker using precomputed metrics.

    :param metrics: DataFrame containing precomputed metrics, including cumulative contributions.
    :param color_map: Dictionary mapping tickers to their colors.
    :param title: Title for the chart.
    """
    fig, ax = plt.subplots(figsize=(16, 10))  # Larger chart size
    fig.patch.set_facecolor('#212E31')  # Background around chart
    ax.set_facecolor('#212E31')  # Chart background

    # Filter columns for cumulative contributions
    contribution_cols = [
        col for col in metrics.columns if "Cumulative Contribution" in col
    ]

    for col in contribution_cols:
        ticker = col.split()[0]  # Extract the ticker name
        ax.plot(
            metrics.index,
            metrics[col],
            label=f"{ticker} Contribution",
            linestyle='-',
            color=color_map.get(ticker, "#FFFFFF")  # Use color from unified color_map or default to white
        )

    # Set axis labels, title, and format ticks
    ax.set_xlim(left=metrics.index.min())  # Ensure the first point starts at the Y axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Quarterly ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Date format

    ax.set_xlabel('Date', color='#FFFFFF', fontsize=16)
    ax.set_ylabel('Cumulative Contribution', color='#FFFFFF', fontsize=16)
    ax.set_title(title, color='#FFFFFF', fontsize=20)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Convert to percentage
    ax.tick_params(axis='both', colors='#FFFFFF', labelsize=14)  # Increase tick font size
    ax.legend(
        facecolor='#212E31',
        edgecolor='#FFFFFF',
        framealpha=0.5,
        labelcolor='#FFFFFF',
        fontsize=14
    )  # Larger legend text
    ax.spines['top'].set_color('#FFFFFF')
    ax.spines['bottom'].set_color('#FFFFFF')
    ax.spines['left'].set_color('#FFFFFF')
    ax.spines['right'].set_color('#FFFFFF')
    ax.grid(True, color='#FFFFFF', alpha=0.3)  # Light grid
    plt.tight_layout()
    return fig

# Function to plot portfolio allocation piechart
def plot_allocation_pie(weights, tickers, title, color_map):
    """
    Plots a pie chart for portfolio allocations with callout labels and percentages.
    :param weights: List of portfolio weights.
    :param tickers: List of portfolio tickers.
    :param title: Title for the pie chart.
    :param color_map: Dictionary mapping tickers to colors.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use the color map to set colors
    colors = [color_map[ticker] for ticker in tickers]

    wedges, _ = ax.pie(
        weights,
        labels=None,  # Disable default labels to customize
        startangle=90,
        colors=colors,  # Use the unified color map
    )

    # Add callouts for labels and percentages
    for i, p in enumerate(wedges):
        # Calculate the angle of the slice
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        x = np.cos(np.radians(ang))  # X-coordinate for callout
        y = np.sin(np.radians(ang))  # Y-coordinate for callout

        connectionstyle = f"angle,angleA=0,angleB={ang}"

        # Annotate ticker labels
        ax.annotate(
            f"{tickers[i]}",
            xy=(x * 1.1, y * 1.1),  # Position outside the pie chart
            xytext=(x * 1.4, y * 1.4),  # Offset for callout text
            arrowprops=dict(arrowstyle="-", color="white", connectionstyle=connectionstyle),
            fontsize=12,
            color="white",
            ha="center",
        )

        # Annotate percentages below the labels
        ax.text(
            x * 1.4, y * 1.4 - 0.1,  # Slightly below the label
            f"{weights[i] * 100:.1f}%",  # Format as a percentage
            fontsize=10,
            color="white",
            ha="center",
        )

    # Set title and background colors
    ax.set_title(title, fontsize=14, color="#FFFFFF")
    fig.patch.set_facecolor('#212E31')  # Background around the chart
    ax.set_facecolor('#212E31')  # Chart background
    plt.tight_layout()
    return fig

# Plot portfolio drawdowns using precomputed metrics
def plot_drawdown(metrics, portfolio_column, portfolio_name):
    """
    Plot drawdown for a portfolio using precomputed metrics.
    :param metrics: DataFrame containing cumulative portfolio returns.
    :param portfolio_column: Name of the column with portfolio cumulative returns.
    :param portfolio_name: Name of the portfolio (for chart title/labels).
    """
    drawdown = calculate_drawdown(metrics, portfolio_column)
    
    # Plot the drawdown
    fig, ax = plt.subplots(figsize=(16, 6))  # Adjust size for drawdown graph
    fig.patch.set_facecolor('#212E31')  # Background around chart
    ax.set_facecolor('#212E31')  # Chart background

    ax.plot(drawdown.index, drawdown, label=f'{portfolio_name} Drawdown', linestyle='-', color='#FF6F61')
    ax.fill_between(drawdown.index, drawdown, 0, color='#FF6F61', alpha=0.5)  # Highlight area below drawdown

    ax.set_xlim(left=drawdown.index.min())  # Ensure the first point starts at the Y axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Quarterly ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Date format

    # Set labels, title, and increase font sizes
    ax.set_xlabel('Date', color='#FFFFFF', fontsize=16)
    ax.set_ylabel('Drawdown', color='#FFFFFF', fontsize=16)
    ax.set_title(f'{portfolio_name} Drawdown Over Time', color='#FFFFFF', fontsize=20)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))  # Convert to percentage
    ax.tick_params(axis='both', colors='#FFFFFF', labelsize=14)  # Increase tick font size
    ax.legend(facecolor='#212E31', edgecolor='#FFFFFF', framealpha=0.5, labelcolor='#FFFFFF', fontsize=14)
    ax.spines['top'].set_color('#FFFFFF')
    ax.spines['bottom'].set_color('#FFFFFF')
    ax.spines['left'].set_color('#FFFFFF')
    ax.spines['right'].set_color('#FFFFFF')
    ax.grid(True, color='#FFFFFF', alpha=0.3)  # Light grid
    plt.tight_layout()
    return fig

# Main function
if st.sidebar.button("Send it.", key="calculate_button"):
    # Step 1: Fetch data for the portfolios
    data1 = fetch_data(tickers1, start_date, end_date, frequency)
    data2 = fetch_data(tickers2, start_date, end_date, frequency)

    # Combine all unique tickers from both portfolios
    all_tickers = sorted(set(data1.columns).union(set(data2.columns)))

    # Generate a color map for all tickers
    color_map = {ticker: plt.cm.tab10(i % 10) for i, ticker in enumerate(all_tickers)}

    if data1 is not None and data2 is not None:
        # Step 2: Validate weights for each portfolio
        weights1 = validate_weights(weights1, len(data1.columns))  # Portfolio 1
        weights2 = validate_weights(weights2, len(data2.columns))  # Portfolio 2

        # Align weights with column order in data
        weights1 = [weights1[tickers1.split(",").index(ticker)] for ticker in data1.columns]
        weights2 = [weights2[tickers2.split(",").index(ticker)] for ticker in data2.columns]

        # Step 3: Proceed if both weights are valid
        if weights1 and weights2:
            # Step 4: Calculate metrics for both portfolios
            metrics1 = calculate_portfolio_metrics(data1, weights1)
            metrics2 = calculate_portfolio_metrics(data2, weights2)

            # Calculate volatilities for Portfolio 1 and Portfolio 2
            ticker_volatility1, portfolio_volatility1 = calculate_volatility(data1, weights1, frequency)
            ticker_volatility2, portfolio_volatility2 = calculate_volatility(data2, weights2, frequency)

            # Step 5: Plot the main portfolio comparison graph
            st.write("### Portfolio Comparison")
            fig, ax = plt.subplots(figsize=(16, 8))  # Adjusted size for main chart
            fig.patch.set_facecolor('#212E31')
            ax.set_facecolor('#212E31')

            # Plot cumulative returns for both portfolios
            ax.plot(
                metrics1.index,
                metrics1["Portfolio Cumulative Return"],
                label="Portfolio 1",
                linestyle='-',
                color='#EDEA99'
            )
            ax.plot(
                metrics2.index,
                metrics2["Portfolio Cumulative Return"],
                label="Portfolio 2",
                linestyle='-',
                color='#96CFD8'
            )

            # Customize the chart
            ax.set_xlim(left=metrics1.index.min())  # Ensure the first point starts at the Y axis
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Quarterly ticks
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Date format
            ax.set_xlabel('Date', color='#FFFFFF', fontsize=16)
            ax.set_ylabel('Cumulative Return', color='#FFFFFF', fontsize=16)
            ax.set_title(f'Cumulative Returns {frequency} Data, ({start_date} to {end_date})', color='#FFFFFF', fontsize=20)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Format as percentage
            ax.tick_params(axis='both', colors='#FFFFFF', labelsize=14)
            ax.legend(facecolor='#212E31', edgecolor='#FFFFFF', framealpha=0.5, labelcolor='#FFFFFF', fontsize=14)

            # Style the plot spines and grid
            ax.spines['top'].set_color('#FFFFFF')
            ax.spines['bottom'].set_color('#FFFFFF')
            ax.spines['left'].set_color('#FFFFFF')
            ax.spines['right'].set_color('#FFFFFF')
            ax.grid(True, color='#FFFFFF', alpha=0.3)  # Light grid

            plt.tight_layout()
            st.pyplot(fig)

            # Step 6: Create two columns for detailed portfolio analysis
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Portfolio 1")
                st.pyplot(plot_allocation_pie(weights1, data1.columns, "Portfolio 1 Allocation", color_map))
                st.write("##### Drawdown")
                st.pyplot(plot_drawdown(metrics1, "Portfolio Cumulative Return", "Portfolio 1"))
                st.write("##### Contributions")
                st.pyplot(plot_ticker_contributions(metrics1, color_map, "Portfolio 1: Contributions"))
                st.write("##### Cumulative Returns")
                st.pyplot(plot_ticker_cumulative_returns(metrics1, color_map, "Portfolio 1: Cumulative Returns"))
                st.write("##### Volatility")
                st.write(f"**Portfolio Volatility**: {portfolio_volatility1 * 100:.2f}%")
                vol_table1 = pd.DataFrame({
                    "Ticker": data1.columns,
                    "Annualized Volatility (%)": ticker_volatility1.values * 100
                })
                st.table(vol_table1)

            with col2:
                st.write("#### Portfolio 2")
                st.pyplot(plot_allocation_pie(weights2, data2.columns, "Portfolio 2 Allocation", color_map))
                st.write("##### Drawdown")
                st.pyplot(plot_drawdown(metrics2, "Portfolio Cumulative Return", "Portfolio 2"))
                st.write("##### Contributions")
                st.pyplot(plot_ticker_contributions(metrics2, color_map, "Portfolio 2: Contributions"))
                st.write("##### Cumulative Returns")
                st.pyplot(plot_ticker_cumulative_returns(metrics2, color_map, "Portfolio 2: Cumulative Returns"))
                st.write("##### Volatility")
                st.write(f"**Portfolio Volatility**: {portfolio_volatility2 * 100:.2f}%")
                vol_table2 = pd.DataFrame({
                    "Ticker": data2.columns,
                    "Annualized Volatility (%)": ticker_volatility2.values * 100
                })
                st.table(vol_table2)

          # Step 7: Display the returns table
            st.write("### Returns Table")
            st.write("Below is a table of returns, cumulative returns, and return contributions. You can download it via the button on the top right of the table (hover)")

            # Extract relevant columns for Portfolio 1
            returns_table1 = metrics1[[
                "Portfolio Return",
                "Portfolio Cumulative Return",
            ] + [
                col for col in metrics1.columns if "Return" in col and "Portfolio" not in col
            ] + [
                col for col in metrics1.columns if "Contribution" in col and "Portfolio" not in col
            ]].rename(columns=lambda x: f"{x} (P1)")

            # Extract relevant columns for Portfolio 2
            returns_table2 = metrics2[[
                "Portfolio Return",
                "Portfolio Cumulative Return",
            ] + [
                col for col in metrics2.columns if "Return" in col and "Portfolio" not in col
            ] + [
                col for col in metrics2.columns if "Contribution" in col and "Portfolio" not in col
            ]].rename(columns=lambda x: f"{x} (P2)")

            # Combine both tables into one for display
            return_table = pd.concat([returns_table1, returns_table2], axis=1)
            st.dataframe(return_table)