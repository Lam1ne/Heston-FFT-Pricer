import numpy as np
from numba import njit, prange
from src.models.heston import HestonModel

@njit(parallel=True, fastmath=True)
def _generate_paths_and_payoff(
    S0, v0, r, kappa, theta, sigma, rho, T, K, num_paths, num_steps, is_call
):
    """
    Numba-optimized Monte Carlo simulation for Heston Model.
    Uses Euler-Maruyama discretization with Full Truncation for variance.
    Simulates log-price process for numerical stability.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    
    payoff_sum = 0.0
    
    # Pre-compute constants
    rho_compl = np.sqrt(1 - rho**2)
    
    # Parallel loop over paths
    for i in prange(num_paths):
        vt = v0
        xt = np.log(S0)
        
        for j in range(num_steps):
            # Generate two independent standard normals
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()
            
            # Correlate them
            # dW_v = z1
            # dW_s = rho * z1 + rho_compl * z2
            dw_v = z1 * sqrt_dt
            dw_s = (rho * z1 + rho_compl * z2) * sqrt_dt
            
            # Full Truncation for variance: f(v) = max(v, 0)
            vt_plus = max(vt, 0.0)
            sqrt_vt = np.sqrt(vt_plus)
            
            # Update variance
            # dv = kappa * (theta - v+) * dt + sigma * sqrt(v+) * dW_v
            vt += kappa * (theta - vt_plus) * dt + sigma * sqrt_vt * dw_v
            
            # Update log-price
            # dlnS = (r - 0.5 * v+) * dt + sqrt(v+) * dW_s
            xt += (r - 0.5 * vt_plus) * dt + sqrt_vt * dw_s
            
        # Calculate terminal price
        ST = np.exp(xt)
        
        # Calculate payoff
        if is_call:
            payoff = max(ST - K, 0.0)
        else:
            payoff = max(K - ST, 0.0)
            
        payoff_sum += payoff
        
    return payoff_sum / num_paths

def price_option_mc(
    model: HestonModel, 
    K: float, 
    T: float, 
    is_call: bool = True, 
    num_paths: int = 100_000, 
    num_steps: int = 100
) -> float:
    """
    Prices a European option using Monte Carlo simulation with Numba acceleration.
    """
    # Call the JIT-compiled kernel
    average_payoff = _generate_paths_and_payoff(
        model.S0, model.v0, model.r, model.kappa, model.theta, model.sigma, model.rho,
        T, K, num_paths, num_steps, is_call
    )
    
    # Discount back to present value
    price = np.exp(-model.r * T) * average_payoff
    price = np.exp(-model.r * T) * average_payoff
    return price

if __name__ == "__main__":
    # Quick Monte Carlo test
    import time
    model = HestonModel(S0=100, v0=0.04, r=0.03, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    
    print("Starting Monte Carlo...")
    t0 = time.time()
    price = price_option_mc(model, K=100, T=1.0, num_paths=10000, num_steps=100)
    dt = time.time() - t0
    
    print(f"MC Call Price (K=100, T=1.0): {price:.4f}")
    print(f"Computation time: {dt:.4f}s")
