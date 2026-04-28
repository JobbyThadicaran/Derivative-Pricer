"""
Greeks Utilities
=================
Converts theoretical (unit) Greeks into Cash Greeks that represent
the actual dollar P&L sensitivity of a traded position.

Why Cash Greeks?
  - Unit Greeks tell you per-share, per-option sensitivities.
  - Cash Greeks scale those by the position size (lots × multiplier)
    and, where appropriate, by the spot price, so the result is
    expressed in currency terms (e.g. USD, EUR).

Cash Greek Formulae:
  Cash Delta = Delta × Spot × Lots × Multiplier
      → Dollar P&L for a $1 move in the underlying

  Cash Gamma = Gamma × Spot² × Lots × Multiplier
      → Dollar P&L acceleration (convexity) for a $1 move

  Cash Vega  = Vega × Lots × Multiplier
      → Dollar P&L for a 1% move in implied volatility

  Cash Theta = Theta × Lots × Multiplier
      → Dollar P&L from one day of time decay

P&L Simulation Utilities:
  - Linear P&L   = Delta × ΔS
  - Gamma P&L    = 0.5 × Gamma × (ΔS)²
  - Total P&L    ≈ Delta × ΔS + 0.5 × Gamma × (ΔS)²  (Taylor expansion)
  - Vega P&L     = Vega × Δσ
"""

import numpy as np
import pandas as pd
from models.black_scholes import BlackScholesModel, BlackScholesInputs


class CashGreeks:
    """
    Converts unit Greeks into cash (dollar) Greeks for a given position.

    Parameters
    ----------
    model : BlackScholesModel
        A fully initialised BS pricing model.
    option_type : str
        'Call' or 'Put'.
    lots : int
        Number of lots (contracts) in the position.
    multiplier : float
        Contract multiplier (e.g. 100 for standard equity options).
    """

    def __init__(self, model: BlackScholesModel, option_type: str = "Call",
                 position_type: str = "Long", lots: int = 1, multiplier: float = 1.0):
        self.model = model
        self.option_type = option_type
        self.position_type = position_type  # "Long" or "Short"
        self.lots = lots
        self.multiplier = multiplier
        self._pos_sign = 1.0 if position_type.lower() == "long" else -1.0

    @property
    def notional(self) -> float:
        """Total notional = Spot × Lots × Multiplier."""
        return self.model.inputs.spot * self.lots * self.multiplier

    # ------------------------------------------------------------------
    # Cash Greeks (Adjusted for Position Type)
    # ------------------------------------------------------------------

    def cash_delta(self) -> float:
        """Cash Delta = Delta × Spot × Lots × Multiplier × Sign."""
        return (self.model.delta(self.option_type)
                * self.model.inputs.spot * self.lots * self.multiplier * self._pos_sign)

    def cash_gamma(self) -> float:
        """Cash Gamma = Gamma × Spot² × Lots × Multiplier × Sign."""
        S = self.model.inputs.spot
        return self.model.gamma() * S * S * self.lots * self.multiplier * self._pos_sign

    def cash_vega(self) -> float:
        """Cash Vega = Vega × Lots × Multiplier × Sign (per 1% vol move)."""
        return self.model.vega() * self.lots * self.multiplier * self._pos_sign

    def cash_theta(self) -> float:
        """Cash Theta = Theta × Lots × Multiplier × Sign (per day)."""
        return self.model.theta(self.option_type) * self.lots * self.multiplier * self._pos_sign

    def cash_rho(self) -> float:
        """Cash Rho = Rho × Lots × Multiplier (per 1% rate move)."""
        return self.model.rho(self.option_type) * self.lots * self.multiplier

    def all_cash_greeks(self) -> dict:
        """Return a dictionary of all cash Greeks."""
        return {
            "Cash Delta": self.cash_delta(),
            "Cash Gamma": self.cash_gamma(),
            "Cash Vega": self.cash_vega(),
            "Cash Theta": self.cash_theta(),
            "Cash Rho": self.cash_rho(),
        }

    def summary_table(self) -> pd.DataFrame:
        """
        Return a combined DataFrame of unit Greeks and cash Greeks.
        Both are signed according to the position type.
        """
        unit = self.model.all_greeks(self.option_type)
        cash = self.all_cash_greeks()

        rows = []
        greek_names = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
        for name in greek_names:
            rows.append({
                "Greek": name,
                "Unit Value (Signed)": unit[name] * self._pos_sign,
                "Cash Value": cash[f"Cash {name}"],
            })
        return pd.DataFrame(rows)


class PnLSimulator:
    """
    Simulates P&L across spot and volatility shocks using Taylor-expansion
    approximations and full re-pricing.

    This is useful for stress-testing and scenario analysis.
    """

    def __init__(self, model: BlackScholesModel, option_type: str = "Call",
                 position_type: str = "Long", lots: int = 1, multiplier: float = 1.0):
        self.model = model
        self.option_type = option_type
        self.position_type = position_type
        self.lots = lots
        self.multiplier = multiplier
        self._pos_sign = 1.0 if position_type.lower() == "long" else -1.0
        self._base_price = model.option_price(option_type)

    def compute_pnl_decomposition(self, ds: float, dv: float) -> dict:
        """
        Compute P&L breakdown for a given spot change (ds) and vol change (dv).
        
        dv is in percentage points (e.g. 1.0 for a +1% vol move).
        """
        delta = self.model.delta(self.option_type)
        gamma = self.model.gamma()
        vega = self.model.vega()  # per 1% move
        scale = self.lots * self.multiplier * self._pos_sign

        delta_pnl = delta * ds * scale
        gamma_pnl = 0.5 * gamma * (ds**2) * scale
        vega_pnl = vega * dv * scale

        return {
            "Delta P&L": delta_pnl,
            "Gamma P&L": gamma_pnl,
            "Vega P&L": vega_pnl,
            "Total Approx P&L": delta_pnl + gamma_pnl + vega_pnl
        }

    def spot_pnl(self, spot_shocks: np.ndarray) -> pd.DataFrame:
        """
        Compute P&L for an array of spot price shocks.

        For each shocked spot S' = S + ΔS, computes:
          - Full re-pricing P&L  (exact)
          - Delta P&L            (linear approximation)
          - Gamma P&L            (quadratic correction)
          - Total approx P&L     (Delta + Gamma)

        Parameters
        ----------
        spot_shocks : np.ndarray
            Array of absolute spot shocks (ΔS values).

        Returns
        -------
        pd.DataFrame with columns: Spot Shock, New Spot, Full Reprice P&L,
                                    Delta P&L, Gamma P&L, Approx P&L
        """
        inp = self.model.inputs
        delta = self.model.delta(self.option_type)
        gamma = self.model.gamma()
        scale = self.lots * self.multiplier

        results = []
        for ds in spot_shocks:
            new_spot = inp.spot + ds
            if new_spot <= 0:
                continue

            # Full re-pricing with the shocked spot
            shocked_inputs = BlackScholesInputs(
                spot=new_spot, strike=inp.strike, maturity=inp.maturity,
                volatility=inp.volatility, risk_free_rate=inp.risk_free_rate,
                dividend_yield=inp.dividend_yield, repo_rate=inp.repo_rate,
            )
            shocked_model = BlackScholesModel(shocked_inputs)
            new_price = shocked_model.option_price(self.option_type)

            full_pnl = (new_price - self._base_price) * scale
            delta_pnl = delta * ds * scale
            gamma_pnl = 0.5 * gamma * ds ** 2 * scale
            approx_pnl = delta_pnl + gamma_pnl

            results.append({
                "Spot Shock": ds,
                "New Spot": new_spot,
                "Full Reprice P&L": full_pnl,
                "Delta P&L": delta_pnl,
                "Gamma P&L": gamma_pnl,
                "Approx P&L": approx_pnl,
            })
        return pd.DataFrame(results)

    def vol_pnl(self, vol_shocks_pct: np.ndarray) -> pd.DataFrame:
        """
        Compute P&L for an array of volatility shocks (in percentage points).

        Parameters
        ----------
        vol_shocks_pct : np.ndarray
            Array of vol shocks in percentage points (e.g., +5 means +5% vol).

        Returns
        -------
        pd.DataFrame with columns: Vol Shock (%), New Vol (%), Full Reprice P&L, Vega P&L
        """
        inp = self.model.inputs
        vega = self.model.vega()  # already per 1% move
        scale = self.lots * self.multiplier

        results = []
        for dv in vol_shocks_pct:
            new_vol = inp.volatility + dv / 100.0
            if new_vol <= 0:
                continue

            shocked_inputs = BlackScholesInputs(
                spot=inp.spot, strike=inp.strike, maturity=inp.maturity,
                volatility=new_vol, risk_free_rate=inp.risk_free_rate,
                dividend_yield=inp.dividend_yield, repo_rate=inp.repo_rate,
            )
            shocked_model = BlackScholesModel(shocked_inputs)
            new_price = shocked_model.option_price(self.option_type)

            full_pnl = (new_price - self._base_price) * scale
            vega_pnl = vega * dv * scale  # vega is per 1% so dv is in %

            results.append({
                "Vol Shock (%)": dv,
                "New Vol (%)": new_vol * 100,
                "Full Reprice P&L": full_pnl,
                "Vega P&L": vega_pnl,
            })
        return pd.DataFrame(results)

    def combined_pnl_matrix(self, spot_shocks: np.ndarray,
                            vol_shocks_pct: np.ndarray) -> pd.DataFrame:
        """
        Compute a P&L matrix across spot AND vol shocks simultaneously.
        Returns a DataFrame with spot shocks as index, vol shocks as columns.
        """
        inp = self.model.inputs
        scale = self.lots * self.multiplier
        matrix = {}

        for dv in vol_shocks_pct:
            new_vol = inp.volatility + dv / 100.0
            if new_vol <= 0:
                continue
            col = []
            for ds in spot_shocks:
                new_spot = inp.spot + ds
                if new_spot <= 0:
                    col.append(np.nan)
                    continue
                shocked_inputs = BlackScholesInputs(
                    spot=new_spot, strike=inp.strike, maturity=inp.maturity,
                    volatility=new_vol, risk_free_rate=inp.risk_free_rate,
                    dividend_yield=inp.dividend_yield, repo_rate=inp.repo_rate,
                )
                shocked_model = BlackScholesModel(shocked_inputs)
                new_price = shocked_model.option_price(self.option_type)
                pnl = (new_price - self._base_price) * scale
                col.append(pnl)
            matrix[f"Vol {dv:+.0f}%"] = col

        return pd.DataFrame(matrix, index=[f"S {ds:+.1f}" for ds in spot_shocks])
