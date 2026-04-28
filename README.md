# 📈 Derivative Pricer — Black-Scholes Options Pricing Engine

A production-ready **European options pricing dashboard** built with Python and Streamlit. This application implements the **Black-Scholes model** to price vanilla European Call and Put options, compute Greeks in both unit and cash terms, and simulate P&L scenarios across spot and volatility shocks.

---

## 🚀 Live Demo

🔗 **[Launch the App](https://derivative-pricer-production.up.railway.app/)**

---

## 📋 Table of Contents

- [Features](#-features)
- [Finance Background](#-finance-background)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **European Option Pricing** | Analytical Black-Scholes pricing for Calls and Puts |
| **Full Greeks Suite** | Delta, Gamma, Vega, Theta, Rho — both unit and cash values |
| **Cash Greeks** | Dollar-denominated risk metrics scaled by position size |
| **Interactive Charts** | Option Price vs Spot, Delta vs Spot, Gamma vs Spot |
| **P&L Simulation** | Spot shock, vol shock, and combined heatmap analysis |
| **Batch Scaling** | Compare Greeks across 1, 10, 100, 1000 lots side-by-side |
| **Forward Pricing** | Computes the theoretical forward price of the underlying |
| **Dividend & Repo** | Supports continuous dividend yield and repo rate adjustments |
| **Error Handling** | Input validation with informative error messages |

---

## 📚 Finance Background

### The Black-Scholes Model

The **Black-Scholes model** (1973) provides a closed-form solution for pricing European options. It assumes the underlying asset follows a geometric Brownian motion:

```
dS = (r - q) × S × dt + σ × S × dW
```

Where:
- **S** = Spot price of the underlying asset
- **r** = Risk-free interest rate
- **q** = Continuous dividend yield
- **σ** = Volatility (annualised standard deviation of log-returns)
- **dW** = Increment of a Wiener process

### Pricing Formulae

```
d₁ = [ln(S/K) + (r - q + σ²/2) × T] / (σ × √T)
d₂ = d₁ - σ × √T

Call = S × e^(-qT) × N(d₁) − K × e^(-rT) × N(d₂)
Put  = K × e^(-rT) × N(-d₂) − S × e^(-qT) × N(-d₁)
```

### The Greeks

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| **Delta (Δ)** | ∂V/∂S | Price sensitivity to a $1 spot move |
| **Gamma (Γ)** | ∂²V/∂S² | Rate of change of Delta (convexity) |
| **Vega (ν)** | ∂V/∂σ | Sensitivity to 1% volatility change |
| **Theta (Θ)** | ∂V/∂t | Daily time decay |
| **Rho (ρ)** | ∂V/∂r | Sensitivity to 1% rate change |

### Cash Greeks

Cash Greeks translate unit sensitivities into dollar terms for a given position size:

```
Cash Delta = Delta × Spot × Lots × Multiplier
Cash Gamma = Gamma × Spot² × Lots × Multiplier
Cash Vega  = Vega × Lots × Multiplier
Cash Theta = Theta × Lots × Multiplier
```

### P&L Approximation (Taylor Expansion)

```
Total P&L ≈ Delta × ΔS + ½ × Gamma × (ΔS)² + Vega × Δσ + Theta × Δt
```

This second-order Taylor expansion captures:
- **Linear risk** (Delta × ΔS)
- **Convexity** (½ × Gamma × ΔS²) — the "Gamma P&L"
- **Volatility risk** (Vega × Δσ)
- **Time decay** (Theta × Δt)

---

## 🏗 Architecture

```
Derivative-Pricer/
│
├── app.py                     # Streamlit dashboard (UI layer)
├── models/
│   ├── __init__.py
│   └── black_scholes.py       # BS pricing engine (quant layer)
├── utils/
│   ├── __init__.py
│   └── greeks.py              # Cash Greeks & P&L simulation
│
├── .streamlit/
│   └── config.toml            # Streamlit theme & server config
├── requirements.txt           # Python dependencies
├── Procfile                   # Railway deployment command
├── .gitignore
└── README.md
```

### Design Principles

- **Modular**: Quant logic (`models/`) is cleanly separated from UI (`app.py`) and utilities (`utils/`)
- **OOP**: `BlackScholesModel` and `CashGreeks` classes encapsulate state and behaviour
- **Validated**: `BlackScholesInputs` dataclass validates all parameters on construction
- **Documented**: Every function includes docstrings explaining the financial intuition

---

## 💻 Installation

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/JobbyThadicaran/Derivative-Pricer.git
cd Derivative-Pricer

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🎮 Usage

1. **Set Parameters** — Use the sidebar to input spot price, strike, maturity, volatility, risk-free rate, dividend yield, and position sizing
2. **View Pricing** — The "Pricing" tab shows the option price, total premium, forward price, and model internals (d₁, d₂)
3. **Analyse Greeks** — The "Greeks" tab displays all unit and cash Greeks in a table and as metric cards
4. **Explore Charts** — Interactive Plotly charts show how price, delta, and gamma vary with spot
5. **Simulate P&L** — Use sliders to stress-test the position across spot and vol shocks; view the P&L heatmap
6. **Batch Scale** — Compare positions across multiple lot sizes

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Spot (S) | 100 | Current asset price |
| Strike (K) | 100 | At-the-money |
| Maturity (T) | 1 year | One year to expiry |
| Volatility (σ) | 20% | Moderate implied vol |
| Risk-Free Rate (r) | 5% | Annualised rate |
| Dividend Yield (q) | 0% | No dividends |

---

## 🚢 Deployment

### Railway (Recommended)

1. Push your code to GitHub
2. Go to [railway.app](https://railway.app) and create a new project
3. Connect your GitHub repository
4. Railway will auto-detect the `Procfile` and deploy
5. Set the `PORT` environment variable if needed (Railway usually handles this)

The `Procfile` runs:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### Streamlit Cloud (Alternative)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch, and `app.py` as the main file
4. Deploy

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit dashboard with 5 tabs: Pricing, Greeks, Charts, P&L, Batch |
| `models/black_scholes.py` | Black-Scholes pricing engine with all Greeks |
| `utils/greeks.py` | Cash Greeks calculator and P&L simulator |
| `.streamlit/config.toml` | Dark theme and server configuration |
| `requirements.txt` | Pinned Python dependencies |
| `Procfile` | Railway deployment command |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Jobby Thadicaran**

- GitHub: [@JobbyThadicaran](https://github.com/JobbyThadicaran)

---

<p align="center">
  Built with ❤️ using Python, Streamlit, and the Black-Scholes Model
</p>
