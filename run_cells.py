# Quick script to run remaining notebook cells
import subprocess
import sys

cells_to_run = [
    21, # Moving Average
    22, # MA Metrics
    24, # Exp Smoothing
    25, # ES Metrics
    27, # ARIMA
    28, # ARIMA forecast
    29, # ARIMA metrics
    31, # Comparison
    36, # Future forecast
]

print("Running cells:", cells_to_run)
print("This may take a few minutes...")
print("=" * 50)
