# GM-Using-Matrices-Portoflio-Optimization-
import yfinance as yf
import pandas as pd 
import numpy as np

stock1 = yf.download('INFY.NS',start= '2024-01-01',end= '2026-01-01',progress= True)['Close'].squeeze()
stock2 = yf.download('TCS.NS',start='2024-01-01',end='2026-01-01',progress= True)['Close'].squeeze()
stock3 = yf.download('RELIANCE.NS',start='2024-01-01',end='2026-01-01',progress= True)['Close'].squeeze()
stock4 = yf.download('BAJAJFINSV.NS',start= '2024-01-01',end= '2026-01-01',progress= True)['Close'].squeeze()
stock5 = yf.download('HCLTECH.NS',start='2024-01-01',end='2026-01-01',progress= True)['Close'].squeeze()
print(type(stock1))
prices = pd.DataFrame({
    'INFY' : stock1,
    "TCS" : stock2,
    "RELIANCE": stock3,
    "BAJAJFINSV" : stock4,
    "HCL": stock5
    })
returns = prices.pct_change()
variances = returns.var()
ann_returns = returns.mean() * 252
ann_variances = variances * 252
ann_std_dev = np.sqrt(ann_variances)
cov_matrix = returns.cov().values
cov_matrix_annual = cov_matrix * 252
inv_cov_matrix = np.linalg.inv(cov_matrix_annual)
ann_expected_returns = ann_returns.values
ones_vector = np.ones(5)
weight_GMV = (inv_cov_matrix @ ones_vector) / (ones_vector.T @ inv_cov_matrix @ ann_expected_returns)
print(weight_GMV)
print(weight_GMV.sum())

# This model is not applicable in a real case scenario for the following reasons: 1.Expected returns are historically calculated. 2.Variance is based on historical movements (optimal to use bottom up variance or Beta). 3.Over-sensitivity towards expectedd returns (Can be adjsuted for using Black-Litterman Model)
