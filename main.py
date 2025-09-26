
import matplotlib.pyplot as plt
from frontend import Frontend
from backend import Backend
from cert import TiingoKey

# Initialize backend and frontend
frontend = Frontend()
backend = Backend(api_key=TiingoKey)

# Fetch data for multiple stocks
symbols = ['AAPL', 'TSLA', 'GOOGL']
dataframes = []

for symbol in symbols:
    df = backend.get_daily_price(symbol, '2024-01-01', '2024-12-31')
    dataframes.append(df)

# Create comparison plot
fig = frontend.plot_price_comparison(
    dataframes=dataframes,
    symbols=symbols,
    normalize=True,
    title='Tech Stocks Performance Comparison 2024'
)

plt.show()

