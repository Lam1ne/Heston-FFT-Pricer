import numpy as np
from scipy.interpolate import interp1d
from src.models.heston import HestonModel

def price_option_fft(
    model: HestonModel, 
    K: float, 
    T: float, 
    alpha: float = 1.5, 
    N: int = 4096, 
    B: float = 1000.0
) -> float:
    """
    Prices a European Call option using the Carr-Madan (1999) FFT method.
    We integrate the characteristic function using FFT.
    """
    # 1. Discretization of the integration domain
    # eta is the step size in the frequency domain (v)
    eta = B / N
    
    # v_j = eta * (j - 1) for j = 1 to N
    # We use 0 to N-1 for Python indexing
    v = np.arange(0, N) * eta
    
    # 2. Calculate the characteristic function argument
    # The Carr-Madan formula requires phi(v - (alpha + 1)i)
    # We construct the complex argument u for the characteristic function
    u = v - (alpha + 1) * 1j
    
    # 3. Compute Characteristic Function
    # phi(u) = E[exp(i * u * ln(S_T))]
    phi = model.characteristic_function(u, T)
    
    # 4. Compute the Damped Payoff Transform psi(v)
    # psi(v) = exp(-rT) * phi(v - (alpha+1)i) / (alpha^2 + alpha - v^2 + i(2alpha + 1)v)
    denominator = (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
    psi = np.exp(-model.r * T) * phi / denominator
    
    # 5. Apply Simpson's Rule weights or Trapezoidal for better accuracy
    # We use Trapezoidal rule weights: [0.5, 1, 1, ..., 1, 0.5] * eta
    weights = np.ones(N)
    weights[0] = 0.5
    weights[-1] = 0.5
    
    # 6. FFT Calculation
    # lambda_step is the step size in the log-strike domain (k)
    # lambda * eta = 2 * pi / N
    lambda_step = 2 * np.pi / (N * eta)
    
    # b is the starting point of the log-strikes
    # We center the grid somewhat. 
    # k_u = -b + lambda * u
    # We want the strikes to cover the area around ln(S0).
    # A common choice is b = N * lambda / 2
    b = N * lambda_step / 2
    
    fft_input = np.exp(1j * b * v) * psi * eta * weights
    
    # Apply FFT
    # We use the real part of the FFT output as the option price should be real.
    y = np.fft.fft(fft_input)
    
    # 7. Post-processing
    # The result y[u] corresponds to log-strike k_u
    # C(k_u) = exp(-alpha * k_u) * real(y[u]) / pi
    
    k_grid = -b + lambda_step * np.arange(N)
    
    # We only care about the real part
    call_prices = np.exp(-alpha * k_grid) * np.real(y) / np.pi
    
    # 8. Interpolation
    # We have prices for a grid of log-strikes k_grid.
    # We want the price for log(K).
    target_k = np.log(K)
    
    # Use linear interpolation
    interpolator = interp1d(k_grid, call_prices, kind='cubic')
    
    try:
        price = interpolator(target_k)
    except ValueError:
        # Fallback if K is out of bounds (unlikely with large N and B)
        return 0.0
        
    return float(price)

if __name__ == "__main__":
    # Quick FFT pricing test
    model = HestonModel(S0=100, v0=0.04, r=0.03, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    price = price_option_fft(model, K=100, T=1.0)
    print(f"FFT Call Price (K=100, T=1.0): {price:.4f}")
