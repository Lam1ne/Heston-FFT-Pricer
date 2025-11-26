import numpy as np
from dataclasses import dataclass

@dataclass
class HestonModel:
    """
    Heston Stochastic Volatility Model.
    
    Implements the characteristic function using the stable formulation 
    from Albrecher et al. (2007) to avoid branch cut issues.

    Dynamics:
        dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW^S_t
        dv_t = kappa * (theta - v_t) * dt + sigma * sqrt(v_t) * dW^v_t
        d<W^S, W^v>_t = rho * dt
    """
    S0: float    # Initial stock price
    v0: float    # Initial variance
    r: float     # Risk-free rate
    kappa: float # Mean reversion speed
    theta: float # Long-term variance
    sigma: float # Vol of vol
    rho: float   # Correlation

    def characteristic_function(self, u: np.ndarray, T: float) -> np.ndarray:
        """
        Computes the characteristic function of the log-price ln(S_T).
        phi(u) = E[exp(i * u * ln(S_T))]
        """
        # Parameters unpacking for readability
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        rho = self.rho
        v0 = self.v0
        r = self.r
        S0 = self.S0
        
        # Log-price
        x0 = np.log(S0)

        # Auxiliary variables
        # d_k in literature often refers to the root term
        # D = sqrt((kappa - i*rho*sigma*u)^2 + (u^2 + i*u)*sigma^2)
        # Note: The term (u^2 + i*u) comes from the Heston PDE for log-price.
        # Some formulations use u^2 - i*u depending on if they define phi(u) as E[exp(iuX)] or E[exp(-iuX)].
        # We use standard E[exp(iuX)].
        
        alpha = -0.5 * u * (u + 1j) # This is -0.5 * (u^2 + i*u)
        beta = kappa - 1j * rho * sigma * u
        gamma = 0.5 * sigma**2

        # d = sqrt(beta^2 - 4*alpha*gamma)
        # Simplifying: beta^2 - 4*alpha*gamma 
        # = (kappa - i*rho*sigma*u)^2 - 4 * (-0.5 * (u^2 + i*u)) * (0.5 * sigma^2)
        # = (kappa - i*rho*sigma*u)^2 + (u^2 + i*u) * sigma^2
        # This matches the standard D term.
        
        D = np.sqrt(beta**2 - 4 * alpha * gamma)

        # The "g" variable in Albrecher form
        # g = (beta - D) / (beta + D)
        # Standard Heston often uses g = (kappa - i*rho*sigma*u - D) / (kappa - i*rho*sigma*u + D)
        # which is exactly (beta - D) / (beta + D)
        
        g = (beta - D) / (beta + D)

        # Characteristic function components
        # A(u, T) term
        # The stable form for the log part: -2 * log((1 - g * exp(-D * T)) / (1 - g))
        
        exponent_C = (kappa * theta / sigma**2) * (
            (beta - D) * T - 2 * np.log((1 - g * np.exp(-D * T)) / (1 - g))
        )
        
        # D(u, T) term (coefficient of v0)
        exponent_D = (v0 / sigma**2) * (beta - D) * (1 - np.exp(-D * T)) / (1 - g * np.exp(-D * T))

        # Final Characteristic Function
        # phi(u) = exp(i * u * (x0 + r*T) + exponent_C + exponent_D)
        # Note: The drift r usually appears as i*u*r*T in the log-price process char func.
        
        return np.exp(1j * u * (x0 + r * T) + exponent_C + exponent_D)

if __name__ == "__main__":
    # Quick test to check if class instantiation works
    model = HestonModel(S0=100, v0=0.04, r=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    print("Model created:", model)
    
    # Testing characteristic function
    u = np.array([0.5, 1.0, 1.5])
    phi = model.characteristic_function(u, T=1.0)
    print("Characteristic function values:", phi)
