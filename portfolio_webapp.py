import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.cm as cm

# Set page configuration
st.set_page_config(layout="wide")

# Set up the page layout
st.title("Janko's Quick&Dirty Portfolio Comparison Tool")
st.sidebar.header("Portfolio Settings")

with st.expander("Description and Instructions", expanded=True):
# Create two columns for Introduction and Instructions
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Description
        This tool provides analysis of two portfolios, including:

        - **Cumulative Returns**
        - **Drawdown**
        - **Return Contributions**
        - **Ticker Cumulative Returns**
        - **Volatility**
        - **Returns Table**
        """)

    with col2:
        st.markdown("""
        ### Instructions
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

        # Drop rows where any ticker has NaN values (align by common dates)
        aligned_data = data.dropna()
    
        # Resample data based on frequency
        if frequency == "Weekly":
            data = data.resample('W').last()
        elif frequency == "Monthly":
            data = data.resample('M').last()
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Calculate portfolio returns
def calculate_portfolio(data, weights):
    # Ensure weights align with columns in data
    weights = pd.Series(weights, index=data.columns)
    daily_returns = data.pct_change().fillna(0)  # Handle NaNs or missing data
    portfolio_returns = daily_returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    return portfolio_returns, cumulative_returns

# Calculate portfolio drawdowns
def calculate_drawdown(cumulative_returns):
    # Ensure the input cumulative returns start at 0% and not 1.0
    if cumulative_returns.iloc[0] != 0:
        cumulative_returns -= cumulative_returns.iloc[0]

    peak = cumulative_returns.cummax()  # Identify peaks
    drawdown = (cumulative_returns - peak) # Calculate drawdown
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

# Function to plot individual ticker cumulative returns with distinct colors
def plot_ticker_cumulative_returns(data):
    fig, ax = plt.subplots(figsize=(16, 10))  # Larger chart size
    fig.patch.set_facecolor('#212E31')  # Background around chart
    ax.set_facecolor('#212E31')  # Chart background

    # Generate a color map with distinct colors for each ticker
    color_map = cm.get_cmap('tab10', len(data.columns))

    for idx, ticker in enumerate(data.columns):
        cumulative_returns = (1 + data[ticker].pct_change().fillna(0)).cumprod() - 1
        ax.plot(data.index, cumulative_returns, 
                label=f'{ticker} Cumulative Return', 
                linestyle='-', 
                color=color_map(idx))  # Use distinct color for each ticker

    ax.set_xlim(left=data.index.min())  # Ensure the first point starts at the Y axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Quarterly ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Date format

    # Set labels, title, and increase font sizes
    ax.set_xlabel('Date', color='#FFFFFF', fontsize=16)
    ax.set_ylabel('Cumulative Return', color='#FFFFFF', fontsize=16)
    ax.set_title('Individual Ticker Cumulative Returns', color='#FFFFFF', fontsize=20)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Convert to percentage
    ax.tick_params(axis='both', colors='#FFFFFF', labelsize=14)  # Increase tick font size
    ax.legend(facecolor='#212E31', edgecolor='#FFFFFF', framealpha=0.5, labelcolor='#FFFFFF', fontsize=14)  # Larger legend text
    ax.spines['top'].set_color('#FFFFFF')
    ax.spines['bottom'].set_color('#FFFFFF')
    ax.spines['left'].set_color('#FFFFFF')
    ax.spines['right'].set_color('#FFFFFF')
    ax.grid(True, color='#FFFFFF', alpha=0.3)  # Light grid
    plt.tight_layout()
    return fig

# Function to plot individual ticker contributions to the portfolio
def plot_ticker_contributions(data, weights):
    fig, ax = plt.subplots(figsize=(16, 10))  # Larger chart size
    fig.patch.set_facecolor('#212E31')  # Background around chart
    ax.set_facecolor('#212E31')  # Chart background

    # Generate a color map with distinct colors for each ticker
    color_map = cm.get_cmap('tab10', len(data.columns))

    for idx, ticker in enumerate(data.columns):
        # Calculate weighted daily contributions
        daily_contribution = data[ticker].pct_change().fillna(0) * weights[idx]
        cumulative_contribution = (1 + daily_contribution).cumprod() - 1 # Cumulative contributions

        ax.plot(data.index, cumulative_contribution, 
                label=f'{ticker} Contribution', 
                linestyle='-', 
                color=color_map(idx))  # Use distinct color for each ticker

    ax.set_xlim(left=data.index.min())  # Ensure the first point starts at the Y axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))  # Quarterly ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Date format

    # Set labels, title, and increase font sizes
    ax.set_xlabel('Date', color='#FFFFFF', fontsize=16)
    ax.set_ylabel('Cumulative Contribution', color='#FFFFFF', fontsize=16)
    ax.set_title('Individual Ticker Contributions to Portfolio', color='#FFFFFF', fontsize=20)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Convert to percentage
    ax.tick_params(axis='both', colors='#FFFFFF', labelsize=14)  # Increase tick font size
    ax.legend(facecolor='#212E31', edgecolor='#FFFFFF', framealpha=0.5, labelcolor='#FFFFFF', fontsize=14)
    ax.spines['top'].set_color('#FFFFFF')
    ax.spines['bottom'].set_color('#FFFFFF')
    ax.spines['left'].set_color('#FFFFFF')
    ax.spines['right'].set_color('#FFFFFF')
    ax.grid(True, color='#FFFFFF', alpha=0.3)  # Light grid
    plt.tight_layout()
    return fig

# Function to plot portfolio drawdowns
def plot_drawdown(cumulative_returns, portfolio_name):
    drawdown = calculate_drawdown(cumulative_returns)
    
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

# If Calculate is clicked
if st.sidebar.button("Send it.", key="calculate_button"):
    # Step 1: Fetch data for the portfolios
    data1 = fetch_data(tickers1, start_date, end_date, frequency)
    data2 = fetch_data(tickers2, start_date, end_date, frequency)

    if data1 is not None and data2 is not None:
        # Step 2: Validate weights for each portfolio
        weights1 = validate_weights(weights1, len(data1.columns))  # Portfolio 1
        weights2 = validate_weights(weights2, len(data2.columns))  # Portfolio 2

        # Align weights with column order in data
        weights1 = [weights1[tickers1.split(",").index(ticker)] for ticker in data1.columns]
        weights2 = [weights2[tickers2.split(",").index(ticker)] for ticker in data2.columns]

        # Step 3: Proceed if both weights are valid
        if weights1 and weights2:
            # Step 4: Calculate portfolio returns
            portfolio1_returns, cumulative_returns1 = calculate_portfolio(data1, weights1)
            portfolio2_returns, cumulative_returns2 = calculate_portfolio(data2, weights2)

            # Calculate volatilities for Portfolio 1 and Portfolio 2
            ticker_volatility1, portfolio_volatility1 = calculate_volatility(data1, weights1, frequency)
            ticker_volatility2, portfolio_volatility2 = calculate_volatility(data2, weights2, frequency)

            # Step 5: Plot the main portfolio comparison graph
            st.write("### Portfolio Comparison")
            fig, ax = plt.subplots(figsize=(16, 8))  # Adjusted size for main chart
            fig.patch.set_facecolor('#212E31')
            ax.set_facecolor('#212E31')
            ax.plot(cumulative_returns1.index, cumulative_returns1, label="Portfolio 1", linestyle='-', color='#EDEA99')
            ax.plot(cumulative_returns2.index, cumulative_returns2, label="Portfolio 2", linestyle='-', color='#96CFD8')
            ax.set_xlim(left=cumulative_returns1.index.min())
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.set_xlabel('Date', color='#FFFFFF', fontsize=16)
            ax.set_ylabel('Cumulative Return', color='#FFFFFF', fontsize=16)
            ax.set_title(f'Cumulative Returns {frequency} Data, ({start_date} to {end_date})', color='#FFFFFF', fontsize=20)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax.tick_params(axis='both', colors='#FFFFFF', labelsize=14)
            ax.legend(facecolor='#212E31', edgecolor='#FFFFFF', framealpha=0.5, labelcolor='#FFFFFF', fontsize=14)
            ax.spines['top'].set_color('#FFFFFF')
            ax.spines['bottom'].set_color('#FFFFFF')
            ax.spines['left'].set_color('#FFFFFF')
            ax.spines['right'].set_color('#FFFFFF')
            ax.grid(True, color='#FFFFFF', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # Step 6: Create two columns for detailed portfolio analysis
            col1, col2 = st.columns(2)

            with col1:
                st.write("## Portfolio 1")
                st.write("### Drawdown")
                st.pyplot(plot_drawdown(cumulative_returns1, "Portfolio 1"))
                st.write("### Contributions")
                st.pyplot(plot_ticker_contributions(data1, weights1))
                st.write("### Cumulative Returns")
                st.pyplot(plot_ticker_cumulative_returns(data1))
                st.write("### Volatility")
                st.write(f"**Portfolio Volatility**: {portfolio_volatility1 * 100:.2f}%")
                vol_table1 = pd.DataFrame({
                    "Ticker": data1.columns,
                    "Annualized Volatility (%)": ticker_volatility1.values * 100
                })
                st.table(vol_table1)

            with col2:
                st.write("## Portfolio 2")
                st.write("### Drawdown")
                st.pyplot(plot_drawdown(cumulative_returns2, "Portfolio 2"))
                st.write("### Contributions")
                st.pyplot(plot_ticker_contributions(data2, weights2))
                st.write("### Cumulative Returns")
                st.pyplot(plot_ticker_cumulative_returns(data2))
                st.write("### Volatility")
                st.write(f"**Portfolio Volatility**: {portfolio_volatility2 * 100:.2f}%")
                vol_table2 = pd.DataFrame({
                    "Ticker": data2.columns,
                    "Annualized Volatility (%)": ticker_volatility2.values * 100
                })
                st.table(vol_table2)

            # Step 7: Display the returns table
            return_table = pd.DataFrame({
                "Portfolio 1 Return": portfolio1_returns,
                "Portfolio 2 Return": portfolio2_returns
            })

            for i, ticker in enumerate(data1.columns):
                ticker_returns = data1[ticker].pct_change().fillna(0)
                ticker_daily_contribution = ticker_returns * weights1[i]
                ticker_cumulative_contribution = (1 + ticker_daily_contribution).cumprod() - 1

                return_table[f"{ticker} Return (P1)"] = ticker_returns
                return_table[f"{ticker} Cumulative (P1)"] = (1 + ticker_returns).cumprod() - 1
                return_table[f"{ticker} Contribution (P1)"] = ticker_cumulative_contribution

            for i, ticker in enumerate(data2.columns):
                ticker_returns = data2[ticker].pct_change().fillna(0)
                ticker_daily_contribution = ticker_returns * weights2[i]
                ticker_cumulative_contribution = (1 + ticker_daily_contribution).cumprod() - 1

                return_table[f"{ticker} Return (P2)"] = ticker_returns
                return_table[f"{ticker} Cumulative (P2)"] = (1 + ticker_returns).cumprod() - 1
                return_table[f"{ticker} Contribution (P2)"] = ticker_cumulative_contribution

            st.write("### Returns Table")
            st.write("Below is a table of returns, cumulative returns, and return contributions. You can download it via the button on the top right of the table (hover)")
            st.dataframe(return_table)