"""
Black-Scholes Option Pricing Model
====================================
Implements the analytical Black-Scholes formula for European option pricing.

The Black-Scholes model assumes:
  - European-style exercise (exercise only at expiry)
  - Log-normal distribution of the underlying asset price
  - Constant volatility, risk-free rate, and dividend yield over the option's life
  - No transaction costs or taxes
  - Continuous trading is possible

Key formulas:
  d1 = [ln(S/K) + (r - q + σ²/2) * T] / (σ * √T)
  d2 = d1 - σ * √T

  Call = S * e^(-qT) * N(d1)  -  K * e^(-rT) * N(d2)
  Put  = K * e^(-rT) * N(-d2) -  S * e^(-qT) * N(-d1)

Where:
  S = Spot price of the underlying asset
  K = Strike price
  T = Time to maturity (in years)
  σ = Volatility (annualised)
  r = Risk-free interest rate (annualised)
  q = Continuous dividend yield (annualised)
  N(·) = Cumulative standard normal distribution function
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional


@dataclass
class BlackScholesInputs:
    """Container for all Black-Scholes model inputs."""
    spot: float          # Current price of the underlying (S)
    strike: float        # Strike price (K)
    maturity: float      # Time to expiration in years (T)
    volatility: float    # Annualised volatility as a decimal, e.g. 0.20 for 20% (σ)
    risk_free_rate: float  # Annualised risk-free rate as a decimal (r)
    dividend_yield: float = 0.0   # Continuous dividend yield (q)
    repo_rate: float = 0.0        # Repo / borrowing cost rate (optional adjustment)

    def __post_init__(self):
        """Validate inputs on creation."""
        if self.spot <= 0:
            raise ValueError(f"Spot price must be positive, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"Strike price must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")
        if self.volatility <= 0:
            raise ValueError(f"Volatility must be positive, got {self.volatility}")


class BlackScholesModel:
    """
    Black-Scholes analytical pricing engine for European options.
    
    This class computes:
      - Option prices (Call / Put)
      - The Greeks (Delta, Gamma, Vega, Theta, Rho)
      - Forward price of the underlying
      - Cash-adjusted Greeks for portfolio risk management
    """

    def __init__(self, inputs: BlackScholesInputs):
        self.inputs = inputs
        # Pre-compute d1 and d2 since they are used by every Greek
        self._d1, self._d2 = self._compute_d1_d2()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_d1_d2(self) -> tuple:
        """
        Compute d1 and d2 — the core intermediates of the BS formula.
        """
        S = self.inputs.spot
        K = self.inputs.strike
        T = max(self.inputs.maturity, 1e-6)  # Stability fix
        sigma = self.inputs.volatility
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield
        repo = self.inputs.repo_rate

        # F = S * exp((r - q - repo) * T) -> r_eff = r - q - repo
        r_eff = r - q - repo

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r_eff + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def forward_price(self) -> float:
        """
        Compute the forward price of the underlying.
        
        Forward = S × e^{(r - q - repo) × T}
        """
        S = self.inputs.spot
        T = max(self.inputs.maturity, 1e-6)
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield
        repo = self.inputs.repo_rate
        return S * np.exp((r - q - repo) * T)

    def call_price(self) -> float:
        """European Call price."""
        S = self.inputs.spot
        K = self.inputs.strike
        T = max(self.inputs.maturity, 1e-6)
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield

        price = (S * np.exp(-q * T) * norm.cdf(self._d1)
                 - K * np.exp(-r * T) * norm.cdf(self._d2))
        return price

    def put_price(self) -> float:
        """European Put price."""
        S = self.inputs.spot
        K = self.inputs.strike
        T = max(self.inputs.maturity, 1e-6)
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield

        price = (K * np.exp(-r * T) * norm.cdf(-self._d2)
                 - S * np.exp(-q * T) * norm.cdf(-self._d1))
        return price

    def option_price(self, option_type: str = "Call") -> float:
        """Return the price for the specified option type."""
        if option_type.lower() == "call":
            return self.call_price()
        elif option_type.lower() == "put":
            return self.put_price()
        else:
            raise ValueError(f"Unknown option type: {option_type}. Use 'Call' or 'Put'.")

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def delta(self, option_type: str = "Call") -> float:
        """Delta (Δ)."""
        q = self.inputs.dividend_yield
        T = max(self.inputs.maturity, 1e-6)
        discount = np.exp(-q * T)

        if option_type.lower() == "call":
            return discount * norm.cdf(self._d1)
        else:
            return discount * (norm.cdf(self._d1) - 1)

    def gamma(self) -> float:
        """Gamma (Γ)."""
        S = self.inputs.spot
        T = max(self.inputs.maturity, 1e-6)
        sigma = self.inputs.volatility
        q = self.inputs.dividend_yield

        return (np.exp(-q * T) * norm.pdf(self._d1)
                / (S * sigma * np.sqrt(T)))

    def vega(self) -> float:
        """Vega (ν) per 1% vol move."""
        S = self.inputs.spot
        T = max(self.inputs.maturity, 1e-6)
        q = self.inputs.dividend_yield

        raw_vega = S * np.exp(-q * T) * norm.pdf(self._d1) * np.sqrt(T)
        return raw_vega / 100.0

    def theta(self, option_type: str = "Call") -> float:
        """Theta (Θ) per day."""
        S = self.inputs.spot
        K = self.inputs.strike
        T = max(self.inputs.maturity, 1e-6)
        sigma = self.inputs.volatility
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield

        vol_decay = -(S * np.exp(-q * T) * norm.pdf(self._d1) * sigma) / (2 * np.sqrt(T))

        if option_type.lower() == "call":
            theta_annual = (vol_decay
                            - r * K * np.exp(-r * T) * norm.cdf(self._d2)
                            + q * S * np.exp(-q * T) * norm.cdf(self._d1))
        else:
            theta_annual = (vol_decay
                            + r * K * np.exp(-r * T) * norm.cdf(-self._d2)
                            - q * S * np.exp(-q * T) * norm.cdf(-self._d1))

        return theta_annual / 365.0

    def rho(self, option_type: str = "Call") -> float:
        """Rho (ρ) per 1% move."""
        K = self.inputs.strike
        T = max(self.inputs.maturity, 1e-6)
        r = self.inputs.risk_free_rate

        if option_type.lower() == "call":
            return K * T * np.exp(-r * T) * norm.cdf(self._d2) / 100.0
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-self._d2) / 100.0

    # ------------------------------------------------------------------
    # Convenience: return all Greeks as a dictionary
    # ------------------------------------------------------------------

    def all_greeks(self, option_type: str = "Call") -> dict:
        """Return a dictionary of all Greeks for the given option type."""
        return {
            "Delta": self.delta(option_type),
            "Gamma": self.gamma(),
            "Vega": self.vega(),
            "Theta": self.theta(option_type),
            "Rho": self.rho(option_type),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, option_type: str = "Call") -> dict:
        """Complete pricing summary including price, forward, and Greeks."""
        return {
            "option_type": option_type,
            "spot": self.inputs.spot,
            "strike": self.inputs.strike,
            "maturity": self.inputs.maturity,
            "volatility": self.inputs.volatility,
            "risk_free_rate": self.inputs.risk_free_rate,
            "dividend_yield": self.inputs.dividend_yield,
            "repo_rate": self.inputs.repo_rate,
            "d1": self._d1,
            "d2": self._d2,
            "forward_price": self.forward_price(),
            "option_price": self.option_price(option_type),
            **self.all_greeks(option_type),
        }
