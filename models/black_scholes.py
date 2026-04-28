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

        d1 measures how many standard deviations the log-moneyness
        (adjusted for drift) is from zero.  d2 = d1 - σ√T shifts
        d1 by one volatility unit, representing the probability
        that the option finishes in-the-money under the risk-neutral measure.
        """
        S = self.inputs.spot
        K = self.inputs.strike
        T = self.inputs.maturity
        sigma = self.inputs.volatility
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield

        # The effective cost-of-carry rate includes the repo rate
        # r_eff = r - q - repo (repo increases the cost of holding the position)
        r_eff = r - q - self.inputs.repo_rate

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
        
        The forward price is the expected future price under risk-neutral
        valuation, adjusted for dividends and repo costs.
        """
        S = self.inputs.spot
        T = self.inputs.maturity
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield
        repo = self.inputs.repo_rate
        return S * np.exp((r - q - repo) * T)

    def call_price(self) -> float:
        """
        European Call price using Black-Scholes.
        
        Call = S·e^{-qT}·N(d1) − K·e^{-rT}·N(d2)
        
        Interpretation:
          - First term:  PV of receiving the asset if option is exercised
          - Second term: PV of paying the strike if option is exercised
        """
        S = self.inputs.spot
        K = self.inputs.strike
        T = self.inputs.maturity
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield

        price = (S * np.exp(-q * T) * norm.cdf(self._d1)
                 - K * np.exp(-r * T) * norm.cdf(self._d2))
        return price

    def put_price(self) -> float:
        """
        European Put price using Black-Scholes.
        
        Put = K·e^{-rT}·N(-d2) − S·e^{-qT}·N(-d1)
        
        Derived from put-call parity:  Put = Call - S·e^{-qT} + K·e^{-rT}
        """
        S = self.inputs.spot
        K = self.inputs.strike
        T = self.inputs.maturity
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
        """
        Delta (Δ): Rate of change of the option price w.r.t. the spot price.
        
        Call Delta = e^{-qT} · N(d1)        ∈ [0, 1]
        Put  Delta = e^{-qT} · (N(d1) - 1)  ∈ [-1, 0]
        
        Delta is the hedge ratio — the number of shares needed to
        delta-hedge one option.
        """
        q = self.inputs.dividend_yield
        T = self.inputs.maturity
        discount = np.exp(-q * T)

        if option_type.lower() == "call":
            return discount * norm.cdf(self._d1)
        else:
            return discount * (norm.cdf(self._d1) - 1)

    def gamma(self) -> float:
        """
        Gamma (Γ): Rate of change of Delta w.r.t. the spot price.
        
        Γ = e^{-qT} · φ(d1) / (S · σ · √T)
        
        Where φ(·) is the standard normal PDF.
        
        Gamma is the same for calls and puts.  It measures the
        curvature of the option price — high gamma means the delta
        is changing rapidly, requiring more frequent re-hedging.
        """
        S = self.inputs.spot
        T = self.inputs.maturity
        sigma = self.inputs.volatility
        q = self.inputs.dividend_yield

        return (np.exp(-q * T) * norm.pdf(self._d1)
                / (S * sigma * np.sqrt(T)))

    def vega(self) -> float:
        """
        Vega (ν): Sensitivity of the option price to changes in volatility.
        
        ν = S · e^{-qT} · φ(d1) · √T
        
        Returned in *price units per 1% vol move* (i.e. divided by 100).
        This makes it directly comparable to how traders quote vega.
        Vega is the same for calls and puts.
        """
        S = self.inputs.spot
        T = self.inputs.maturity
        q = self.inputs.dividend_yield

        # Raw vega (per 1 unit of vol)
        raw_vega = S * np.exp(-q * T) * norm.pdf(self._d1) * np.sqrt(T)
        # Convert to per 1% move (divide by 100)
        return raw_vega / 100.0

    def theta(self, option_type: str = "Call") -> float:
        """
        Theta (Θ): Rate of time decay of the option price.
        
        Theta is returned on a *per-day* basis (divided by 365).
        
        Call Θ = -(S·e^{-qT}·φ(d1)·σ)/(2√T) - r·K·e^{-rT}·N(d2) + q·S·e^{-qT}·N(d1)
        Put  Θ = -(S·e^{-qT}·φ(d1)·σ)/(2√T) + r·K·e^{-rT}·N(-d2) - q·S·e^{-qT}·N(-d1)
        
        Theta is almost always negative — the option loses value as
        time passes (time decay / "theta bleed").
        """
        S = self.inputs.spot
        K = self.inputs.strike
        T = self.inputs.maturity
        sigma = self.inputs.volatility
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield

        # Common term: the "volatility decay" component
        vol_decay = -(S * np.exp(-q * T) * norm.pdf(self._d1) * sigma) / (2 * np.sqrt(T))

        if option_type.lower() == "call":
            theta_annual = (vol_decay
                            - r * K * np.exp(-r * T) * norm.cdf(self._d2)
                            + q * S * np.exp(-q * T) * norm.cdf(self._d1))
        else:
            theta_annual = (vol_decay
                            + r * K * np.exp(-r * T) * norm.cdf(-self._d2)
                            - q * S * np.exp(-q * T) * norm.cdf(-self._d1))

        # Convert from annual theta to daily theta
        return theta_annual / 365.0

    def rho(self, option_type: str = "Call") -> float:
        """
        Rho (ρ): Sensitivity of the option price to the risk-free rate.
        
        Call ρ = K · T · e^{-rT} · N(d2)   / 100
        Put  ρ = -K · T · e^{-rT} · N(-d2) / 100
        
        Returned per 1% rate change.
        """
        K = self.inputs.strike
        T = self.inputs.maturity
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
