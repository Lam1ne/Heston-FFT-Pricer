import unittest
import numpy as np
from src.models.heston import HestonModel
from src.pricing.fft import price_option_fft
from src.pricing.monte_carlo import price_option_mc

class TestHestonPricing(unittest.TestCase):
    def setUp(self):
        # Standard Heston parameters
        self.model = HestonModel(
            S0=100.0,
            v0=0.04,
            r=0.03,
            kappa=2.0,
            theta=0.04,
            sigma=0.3,
            rho=-0.7
        )
        self.T = 1.0
        self.K = 100.0

    def test_call_put_parity(self):
        """
        Test Call-Put Parity: C - P = S0 - K * exp(-rT)
        """
        # Calculate Call Price
        call_price_fft = price_option_fft(self.model, self.K, self.T)
        
        # Calculate Put Price using FFT (requires implementing put pricing or using parity to derive it)
        # Since price_option_fft only does calls currently, we can check parity using MC for both or 
        # just check if MC Call and MC Put satisfy parity.
        
        # Let's use MC for parity check as it supports both flags easily
        mc_call = price_option_mc(self.model, self.K, self.T, is_call=True, num_paths=50000, num_steps=50)
        mc_put = price_option_mc(self.model, self.K, self.T, is_call=False, num_paths=50000, num_steps=50)
        
        lhs = mc_call - mc_put
        rhs = self.model.S0 - self.K * np.exp(-self.model.r * self.T)
        
        # Allow some tolerance due to MC noise
        self.assertAlmostEqual(lhs, rhs, delta=0.5)

    def test_deep_in_the_money_call(self):
        """
        Deep ITM Call should be close to S0 - K*exp(-rT) (intrinsic value)
        """
        deep_itm_K = 10.0 # S0=100
        price = price_option_fft(self.model, deep_itm_K, self.T)
        intrinsic = self.model.S0 - deep_itm_K * np.exp(-self.model.r * self.T)
        self.assertAlmostEqual(price, intrinsic, delta=0.1)

    def test_deep_out_of_the_money_call(self):
        """
        Deep OTM Call should be close to 0
        """
        deep_otm_K = 200.0 # S0=100
        price = price_option_fft(self.model, deep_otm_K, self.T)
        self.assertAlmostEqual(price, 0.0, delta=0.1)

    def test_fft_vs_mc_convergence(self):
        """
        Check if MC converges to FFT price within reasonable tolerance
        """
        fft_price = price_option_fft(self.model, self.K, self.T)
        mc_price = price_option_mc(
            self.model, self.K, self.T, 
            is_call=True, 
            num_paths=100_000, 
            num_steps=100
        )
        
        # 1% relative error tolerance
        rel_error = abs(mc_price - fft_price) / fft_price
        self.assertLess(rel_error, 0.015)

if __name__ == '__main__':
    unittest.main()
