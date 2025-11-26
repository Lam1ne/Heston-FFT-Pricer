"""
Heston Model Pricing Benchmark

Comparing FFT (Carr-Madan) vs Monte Carlo (Numba) for European Call options.
"""

import logging
import time
from typing import Dict, Any

import pandas as pd

from src.models.heston import HestonModel
from src.pricing.fft import price_option_fft
from src.pricing.monte_carlo import price_option_mc

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def run_benchmark():
    """
    Executes the pricing benchmark and displays comparative results.
    """
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    # Market and Model Parameters
    params: Dict[str, float] = {
        "S0": 100.0,    # Spot Price
        "v0": 0.04,     # Initial Variance
        "r": 0.03,      # Risk-Free Rate
        "kappa": 2.0,   # Mean Reversion Speed
        "theta": 0.04,  # Long-Term Variance
        "sigma": 0.3,   # Vol of Vol
        "rho": -0.7,    # Correlation
        "T": 1.0,       # Time to Maturity
        "K": 100.0      # Strike Price
    }

    # Simulation Settings
    mc_config: Dict[str, Any] = {
        "num_paths": 500_000,
        "num_steps": 100,
        "is_call": True
    }

    # ---------------------------------------------------------
    # 2. Initialization
    # ---------------------------------------------------------
    logger.info("Initializing Heston Model...")
    model = HestonModel(
        S0=params["S0"], v0=params["v0"], r=params["r"],
        kappa=params["kappa"], theta=params["theta"], 
        sigma=params["sigma"], rho=params["rho"]
    )
    
    logger.info(f"Pricing Target: European Call | S0={params['S0']} | K={params['K']} | T={params['T']}")

    # ---------------------------------------------------------
    # 3. Pricing: FFT (Benchmark)
    # ---------------------------------------------------------
    logger.info("Executing FFT Pricing (Carr-Madan)...")
    t0 = time.perf_counter()
    price_fft = price_option_fft(model, K=params["K"], T=params["T"])
    dt_fft = time.perf_counter() - t0

    # ---------------------------------------------------------
    # 4. Pricing: Monte Carlo (Numba)
    # ---------------------------------------------------------
    logger.info(f"Executing Monte Carlo Pricing ({mc_config['num_paths']:,} paths)...")
    
    # JIT Compilation Warm-up (to exclude compilation time from benchmark)
    _ = price_option_mc(model, params["K"], params["T"], mc_config["is_call"], num_paths=100, num_steps=10)
    
    t0 = time.perf_counter()
    price_mc = price_option_mc(
        model, 
        params["K"], 
        params["T"], 
        mc_config["is_call"], 
        num_paths=mc_config["num_paths"], 
        num_steps=mc_config["num_steps"]
    )
    dt_mc = time.perf_counter() - t0

    # ---------------------------------------------------------
    # 5. Analysis & Reporting
    # ---------------------------------------------------------
    error_abs = abs(price_mc - price_fft)
    error_rel = error_abs / price_fft if price_fft != 0 else 0.0

    results = [
        {
            "Method": "FFT (Carr-Madan)", 
            "Price": price_fft, 
            "Time (s)": dt_fft, 
            "Abs Error": 0.0,
            "Rel Error (%)": 0.0
        },
        {
            "Method": "Monte Carlo (Numba)", 
            "Price": price_mc, 
            "Time (s)": dt_mc, 
            "Abs Error": error_abs,
            "Rel Error (%)": error_rel * 100
        }
    ]

    df_results = pd.DataFrame(results)
    
    # Display Results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("-" * 80)

    # Validation Check
    if error_rel < 1e-2: # 1% tolerance
        logger.info("Validation: PASSED (Monte Carlo converged within 1% tolerance).")
    else:
        logger.warning(f"Validation: WARNING (Rel Error {error_rel:.2%} > 1%). Check convergence parameters.")

if __name__ == "__main__":
    run_benchmark()
