import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import EfficientFrontier
from pypfopt import plotting
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from empyrial import empyrial, Engine
import concurrent.futures


index = pd.read_csv("/Users/punisher/Desktop/x/cs/stonks1/index.csv")
symbols = index['Symbol'].tolist()
symbol = [symbol + '.NS' for symbol in symbols]

def fetch_rtns(symbol):
  #data = yf.download(symbol,interval='1d',start="2006-01-01", end="2023-01-01")['Close']
  data = yf.download(symbol,start="2024-1-01",end="2024-1-30",interval="1d")['Close']
  return data  

data0 = fetch_rtns(symbol)
index = fetch_rtns("^NSEI")

def optmz(choice, value):
    symbol_list = []
    weight_list = []
    if choice == 0:
        rtns = expected_returns.mean_historical_return(data0, frequency=21)
        cov_matrix = risk_models.CovarianceShrinkage(data0, frequency=21).ledoit_wolf()
        frontier = EfficientFrontier(rtns, cov_matrix)
        plotting.plot_efficient_frontier(frontier, show_assets=True, show_fig=True, show_tickers=True)
        print(rtns)
        #print(cov_matrix)
    else:
        rtns = expected_returns.mean_historical_return(data0, frequency=21)
        cov_matrix = risk_models.CovarianceShrinkage(data0, frequency=21).ledoit_wolf()
        frontier = EfficientFrontier(rtns, cov_matrix, solver="SCS")
        raw_weights = frontier.max_sharpe()
        cleaned_weights = frontier.clean_weights()
        frontier.portfolio_performance(verbose=True)
        #plotting.plot_efficient_frontier(frontier, show_assets=True, show_fig=True, show_tickers=False)
        #plotting.plot_covariance(cov_matrix,plot_correlation=True,text_kwargs={'fontsize': 4}, show_tickers=True)
        latest_prices = get_latest_prices(data0)
        da = DiscreteAllocation(raw_weights, latest_prices, total_portfolio_value=value)
        allocation, leftover = da.greedy_portfolio()
        print("Discrete allocation:", allocation)
        print("Funds remaining: ${:.2f}".format(leftover))
        #print(cleaned_weights)
        for k, v in cleaned_weights.items():
            if v > 0:
               symbol_list.append(k)
               weight_list.append(v)
        return symbol_list, weight_list
    

symbols, weights = optmz(1,100000)
dict0 = {'symbol':symbols,'weight':weights}
df = pd.DataFrame(dict0)
print(df.sort_values(by=['weight'],ascending=False))
# backtester(symbols, weights)

def returns(symbol, weight):
    stock = yf.Ticker(symbol)
    stock_history = stock.history(start="2024-01-01",end="2024-01-30", interval="1d")
    stock_return = (((stock_history['Close'][-1])-(stock_history['Close'][0]))/(stock_history['Close'][0]))*100
    return stock_return*weight

def backereturn_calculator():
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = list(executor.map(lambda s,j: returns(s,j),symbols,weights))
    print(sum(result))
    
# optmz(0,100000)
# optmz(1,100000)
# backereturn_calculator()