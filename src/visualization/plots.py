import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

from src.models.heston import HestonModel
from src.pricing.fft import price_option_fft
from src.pricing.monte_carlo import price_option_mc

def plot_mc_convergence(
    model: HestonModel, 
    K: float, 
    T: float, 
    path_steps: List[int] = [1000, 5000, 10000, 50000, 100000, 200000]
):
    """
    Plots the convergence of Monte Carlo price to the FFT benchmark.
    """
    print("Calculating FFT Benchmark...")
    fft_price = price_option_fft(model, K, T)
    
    mc_prices = []
    times = []
    
    print("Running Monte Carlo simulations...")
    # Warmup
    price_option_mc(model, K, T, num_paths=100, num_steps=10)
    
    for n_paths in path_steps:
        t0 = time.perf_counter()
        price = price_option_mc(model, K, T, num_paths=n_paths, num_steps=100)
        dt = time.perf_counter() - t0
        mc_prices.append(price)
        times.append(dt)
        print(f"Paths: {n_paths}, Price: {price:.4f}, Time: {dt:.4f}s")
        
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Price Convergence
    plt.subplot(1, 2, 1)
    plt.plot(path_steps, mc_prices, 'b-o', label='Monte Carlo')
    plt.axhline(y=fft_price, color='r', linestyle='--', label=f'FFT Benchmark ({fft_price:.4f})')
    plt.xscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo Convergence')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Computation Time
    plt.subplot(1, 2, 2)
    plt.plot(path_steps, times, 'g-o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time vs Paths')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def plot_price_surface(model: HestonModel):
    """
    Plots the Call Option Price Surface vs Strike (K) and Maturity (T).
    """
    strikes = np.linspace(model.S0 * 0.5, model.S0 * 1.5, 30)
    maturities = np.linspace(0.1, 2.0, 20)
    
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    prices = np.zeros_like(K_grid)
    
    print("Calculating Price Surface...")
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            prices[i, j] = price_option_fft(model, strikes[j], maturities[i])
            
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        K_grid, T_grid, prices, 
        cmap='viridis', 
        edgecolor='none',
        alpha=0.8
    )
    
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Maturity (T)')
    ax.set_zlabel('Option Price')
    ax.set_title('Heston Call Option Price Surface (FFT)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    model = HestonModel(S0=100, v0=0.04, r=0.03, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    
    print("--- Convergence Plot ---")
    plot_mc_convergence(model, K=100, T=1.0)
    
    print("\n--- Price Surface Plot ---")
    plot_price_surface(model)
