"""
Derivative Pricer — Streamlit Application
==========================================
A production-ready European options pricing dashboard built on the
Black-Scholes model.  Features include:
  • Real-time pricing with configurable parameters
  • Unit and Cash Greeks display
  • Interactive charts (Price vs Spot, Delta vs Spot)
  • P&L simulation with spot & vol shocks
  • Batch scaling across lot sizes
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from models.black_scholes import BlackScholesModel, BlackScholesInputs
from utils.greeks import CashGreeks, PnLSimulator


# ======================================================================
# Page Configuration
# ======================================================================
st.set_page_config(
    page_title="Derivative Pricer — Black-Scholes",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Custom CSS for a premium look
# ======================================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
        font-weight: 500;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-weight: 700;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        color: #e0e7ff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .main-header p {
        color: #a5b4fc;
        font-size: 1rem;
        margin: 4px 0 0 0;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        font-weight: 500;
        padding: 8px 20px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border-color: #6366f1 !important;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider */
    hr {
        border-color: rgba(99, 102, 241, 0.2);
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #c7d2fe;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# Session State Initialization
# ======================================================================
if 'shock_ds_pct' not in st.session_state:
    st.session_state.shock_ds_pct = 0.0
if 'shock_dv' not in st.session_state:
    st.session_state.shock_dv = 0.0

def reset_shocks():
    st.session_state.shock_ds_pct = 0.0
    st.session_state.shock_dv = 0.0

# ======================================================================
# Helper: Plotly theme
# ======================================================================
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15, 15, 35, 0.8)",
    plot_bgcolor="rgba(15, 15, 35, 0.8)",
    font=dict(family="Inter, sans-serif", color="#c7d2fe"),
    title_font=dict(size=16, color="#e0e7ff"),
    xaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)", zerolinecolor="rgba(99, 102, 241, 0.2)"),
    yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)", zerolinecolor="rgba(99, 102, 241, 0.2)"),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="#1e1b4b", font_color="#e0e7ff"),
)


def format_number(value: float, decimals: int = 4) -> str:
    """Format a number with commas and fixed decimals."""
    return f"{value:,.{decimals}f}"


# ======================================================================
# SIDEBAR — Input Parameters
# ======================================================================
with st.sidebar:
    st.markdown("## 📊 Option Parameters")
    st.markdown("---")

    # Option type
    col_type1, col_type2 = st.columns(2)
    with col_type1:
        option_type = st.selectbox("Option Type", ["Call", "Put"], index=0)
    with col_type2:
        position_type = st.selectbox("Position", ["Long", "Short"], index=0,
                                     help="Are you buying (Long) or selling (Short) this option?")

    st.markdown("### 📌 Market Data")
    spot = st.number_input("Spot Price (S)", min_value=0.01, value=100.0,
                           step=1.0, format="%.2f",
                           help="Current market price of the underlying asset")

    strike = st.number_input("Strike Price (K)", min_value=0.01, value=100.0,
                             step=1.0, format="%.2f",
                             help="Exercise price of the option")

    maturity = st.number_input("Maturity (T, years)", min_value=0.001,
                               value=1.0, step=0.25, format="%.3f",
                               help="Time to expiration in years")

    st.markdown("### 📉 Volatility & Rates")
    vol_pct = st.number_input("Volatility (σ, %)", min_value=0.1,
                              value=20.0, step=1.0, format="%.1f",
                              help="Annualised implied volatility in percent")

    rate_pct = st.number_input("Risk-Free Rate (r, %)", min_value=0.0,
                               value=5.0, step=0.25, format="%.2f",
                               help="Annualised risk-free interest rate in percent")

    div_pct = st.number_input("Dividend Yield (q, %)", min_value=0.0,
                              value=0.0, step=0.25, format="%.2f",
                              help="Annualised continuous dividend yield in percent")

    repo_pct = st.number_input("Repo Rate (%)", min_value=0.0,
                               value=0.0, step=0.25, format="%.2f",
                               help="Repo / borrowing cost rate (optional)")

    st.markdown("### 📦 Position Sizing")
    lots = st.number_input("Number of Lots", min_value=1, value=1, step=1,
                           help="Number of option contracts")

    multiplier = st.number_input("Contract Multiplier", min_value=0.01,
                                 value=1.0, step=1.0, format="%.2f",
                                 help="Contract size multiplier (e.g. 100 for equity options)")

    st.markdown("---")
    st.markdown("### ⚡ Batch Scaling")
    batch_lots = st.text_input("Compare Lot Sizes (comma-separated)",
                               value="1, 10, 100, 1000",
                               help="Enter lot sizes to compare side-by-side")


# ======================================================================
# Build the model
# ======================================================================
try:
    inputs = BlackScholesInputs(
        spot=spot,
        strike=strike,
        maturity=maturity,
        volatility=vol_pct / 100.0,
        risk_free_rate=rate_pct / 100.0,
        dividend_yield=div_pct / 100.0,
        repo_rate=repo_pct / 100.0,
    )
    model = BlackScholesModel(inputs)
    cash = CashGreeks(model, option_type, position_type, lots, multiplier)
    pnl_sim = PnLSimulator(model, option_type, position_type, lots, multiplier)

    pricing_ok = True
except ValueError as e:
    st.error(f"⚠️ Invalid input: {e}")
    pricing_ok = False


# ======================================================================
# MAIN PANEL
# ======================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Derivative Pricer</h1>
    <p>Black-Scholes European Options Pricing Engine</p>
</div>
""", unsafe_allow_html=True)

if pricing_ok:
    # ==================================================================
    # Tab layout
    # ==================================================================
    tab_pricing, tab_greeks, tab_charts, tab_pnl, tab_intelligence, tab_batch = st.tabs([
        "💰 Pricing", "📐 Greeks", "📊 Charts", "📈 P&L Simulation", "🧠 Intelligence Layer", "📦 Batch Scaling"
    ])

    # ==================================================================
    # TAB 1 — Pricing
    # ==================================================================
    with tab_pricing:
        col1, col2, col3, col4 = st.columns(4)

        option_price = model.option_price(option_type)
        pos_sign = 1.0 if position_type == "Long" else -1.0
        signed_total_price = pos_sign * option_price * lots * multiplier
        fwd = model.forward_price()

        premium_label = "Premium Paid (Outflow)" if position_type == "Long" else "Premium Received (Inflow)"

        with col1:
            st.metric("Option Price (per unit)", format_number(option_price))
        with col2:
            st.metric(premium_label, format_number(signed_total_price, 2))
        with col3:
            st.metric("Forward Price", format_number(fwd, 4))
        with col4:
            moneyness = spot / strike
            st.metric("Moneyness (S/K)", format_number(moneyness, 4))

        st.markdown("---")

        # Detailed breakdown
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🔢 Model Internals")
            internals = pd.DataFrame({
                "Parameter": ["d₁", "d₂", "N(d₁)", "N(d₂)", "Forward Price"],
                "Value": [
                    format_number(model._d1, 6),
                    format_number(model._d2, 6),
                    format_number(float(np.exp(-inputs.dividend_yield * inputs.maturity) *
                                       norm.cdf(model._d1)), 6),
                    format_number(float(norm.cdf(model._d2)), 6),
                    format_number(fwd, 4),
                ]
            })
            st.dataframe(internals, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("#### 📋 Input Summary")
            input_summary = pd.DataFrame({
                "Parameter": ["Spot (S)", "Strike (K)", "Maturity (T)", "Volatility (σ)",
                               "Risk-Free Rate (r)", "Dividend Yield (q)", "Repo Rate",
                               "Lots", "Multiplier"],
                "Value": [
                    f"{spot:.2f}", f"{strike:.2f}", f"{maturity:.3f}",
                    f"{vol_pct:.1f}%", f"{rate_pct:.2f}%", f"{div_pct:.2f}%",
                    f"{repo_pct:.2f}%", str(lots), f"{multiplier:.2f}",
                ]
            })
            st.dataframe(input_summary, use_container_width=True, hide_index=True)

        # Intuitive explanation
        st.markdown(f"""
        <div class="info-box">
            <strong>💡 Interpretation:</strong> A European <b>{option_type}</b> option on an asset
            trading at <b>{spot:.2f}</b> with strike <b>{strike:.2f}</b>, expiring in
            <b>{maturity:.2f}</b> years, is worth <b>{option_price:.4f}</b> per unit.
            With <b>{lots}</b> lot(s) × <b>{multiplier:.0f}</b> multiplier,
            the total premium is <b>{total_price:,.2f}</b>.
        </div>
        """, unsafe_allow_html=True)

    # ==================================================================
    # TAB 2 — Greeks
    # ==================================================================
    with tab_greeks:
        st.markdown(f"#### {position_type} Position: Unit & Cash Greeks")

        greeks_df = cash.summary_table()
        greeks_df["Unit Value (Signed)"] = greeks_df["Unit Value (Signed)"].apply(lambda x: format_number(x, 6))
        greeks_df["Cash Value"] = greeks_df["Cash Value"].apply(lambda x: format_number(x, 2))
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Individual Greek cards
        cols = st.columns(5)
        greek_names = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
        greek_icons = ["📐", "📏", "🌊", "⏳", "💹"]
        unit_greeks = model.all_greeks(option_type)
        cash_greeks = cash.all_cash_greeks()
        pos_sign = 1.0 if position_type == "Long" else -1.0

        for i, (name, icon) in enumerate(zip(greek_names, greek_icons)):
            with cols[i]:
                st.metric(
                    f"{icon} {name}",
                    format_number(unit_greeks[name] * pos_sign, 6),
                    delta=f"Cash: {format_number(cash_greeks[f'Cash {name}'], 2)}",
                )

        st.markdown("""
        <div class="info-box">
            <strong>📘 Cash Greeks Explained:</strong><br>
            • <b>Cash Delta</b>: Dollar P&L for a $1 move in spot<br>
            • <b>Cash Gamma</b>: Change in Cash Delta for a $1 move (convexity)<br>
            • <b>Cash Vega</b>: Dollar P&L for a 1% move in implied volatility<br>
            • <b>Cash Theta</b>: Daily time decay cost in dollars<br>
            • <b>Cash Rho</b>: Dollar P&L for a 1% move in interest rates
        </div>
        """, unsafe_allow_html=True)

    # ==================================================================
    # TAB 3 — Charts
    # ==================================================================
    with tab_charts:
        # Spot range for plotting
        spot_range = np.linspace(max(0.5, spot * 0.5), spot * 1.5, 200)

        # Compute prices and delta across the spot range
        prices = []
        deltas = []
        gammas = []
        for s in spot_range:
            try:
                temp_inputs = BlackScholesInputs(
                    spot=s, strike=strike, maturity=maturity,
                    volatility=vol_pct / 100.0, risk_free_rate=rate_pct / 100.0,
                    dividend_yield=div_pct / 100.0, repo_rate=repo_pct / 100.0,
                )
                temp_model = BlackScholesModel(temp_inputs)
                prices.append(temp_model.option_price(option_type))
                deltas.append(temp_model.delta(option_type))
                gammas.append(temp_model.gamma())
            except ValueError:
                prices.append(np.nan)
                deltas.append(np.nan)
                gammas.append(np.nan)

        # Chart 1: Option Price vs Spot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=spot_range, y=prices,
            mode='lines',
            name=f'{option_type} Price',
            line=dict(color='#818cf8', width=3),
            fill='tozeroy',
            fillcolor='rgba(129, 140, 248, 0.1)',
        ))
        # Mark current spot
        fig1.add_vline(x=spot, line_dash="dash", line_color="#f472b6",
                       annotation_text=f"Spot = {spot:.2f}")
        fig1.add_vline(x=strike, line_dash="dot", line_color="#34d399",
                       annotation_text=f"Strike = {strike:.2f}")
        fig1.update_layout(
            title=f"{option_type} Price vs Spot",
            xaxis_title="Spot Price",
            yaxis_title="Option Price",
            **PLOTLY_LAYOUT,
        )

        # Chart 2: Delta vs Spot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=spot_range, y=deltas,
            mode='lines',
            name='Delta',
            line=dict(color='#34d399', width=3),
            fill='tozeroy',
            fillcolor='rgba(52, 211, 153, 0.1)',
        ))
        fig2.add_vline(x=spot, line_dash="dash", line_color="#f472b6",
                       annotation_text=f"Spot = {spot:.2f}")
        fig2.add_vline(x=strike, line_dash="dot", line_color="#fbbf24",
                       annotation_text=f"Strike = {strike:.2f}")
        fig2.update_layout(
            title=f"{option_type} Delta vs Spot",
            xaxis_title="Spot Price",
            yaxis_title="Delta",
            **PLOTLY_LAYOUT,
        )

        # Chart 3: Gamma vs Spot
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=spot_range, y=gammas,
            mode='lines',
            name='Gamma',
            line=dict(color='#fbbf24', width=3),
            fill='tozeroy',
            fillcolor='rgba(251, 191, 36, 0.1)',
        ))
        fig3.add_vline(x=spot, line_dash="dash", line_color="#f472b6",
                       annotation_text=f"Spot = {spot:.2f}")
        fig3.update_layout(
            title="Gamma vs Spot",
            xaxis_title="Spot Price",
            yaxis_title="Gamma",
            **PLOTLY_LAYOUT,
        )

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(fig1, use_container_width=True)
        with col_chart2:
            st.plotly_chart(fig2, use_container_width=True)

        st.plotly_chart(fig3, use_container_width=True)

    # ==================================================================
    # TAB 4 — P&L Simulation
    # ==================================================================
    with tab_pnl:
        st.markdown("#### 📈 Spot Shock P&L")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            spot_shock_min_pct = st.slider("Min Spot Shock (%)", -50.0, 0.0, -20.0, 1.0)
        with col_s2:
            spot_shock_max_pct = st.slider("Max Spot Shock (%)", 0.0, 50.0, 20.0, 1.0)

        # UI in %, internal in $
        spot_shocks_pct = np.linspace(spot_shock_min_pct, spot_shock_max_pct, 50)
        spot_shocks_abs = spot * spot_shocks_pct / 100.0

        spot_pnl_df = pnl_sim.spot_pnl(spot_shocks_abs)
        # Add the % back for display
        spot_pnl_df["Spot Shock (%)"] = spot_shocks_pct

        if not spot_pnl_df.empty:
            fig_spot_pnl = go.Figure()
            fig_spot_pnl.add_trace(go.Scatter(
                x=spot_pnl_df["Spot Shock (%)"], y=spot_pnl_df["Full Reprice P&L"],
                name="Full Reprice", line=dict(color="#818cf8", width=3),
            ))
            fig_spot_pnl.add_trace(go.Scatter(
                x=spot_pnl_df["Spot Shock (%)"], y=spot_pnl_df["Delta P&L"],
                name="Delta P&L", line=dict(color="#34d399", width=2, dash="dash"),
            ))
            fig_spot_pnl.add_trace(go.Scatter(
                x=spot_pnl_df["Spot Shock (%)"], y=spot_pnl_df["Approx P&L"],
                name="Delta+Gamma Approx", line=dict(color="#fbbf24", width=2, dash="dot"),
            ))
            fig_spot_pnl.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
            fig_spot_pnl.update_layout(
                title="P&L vs Spot Shock (%)",
                xaxis_title="Spot Shock (%)",
                yaxis_title="P&L ($)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_spot_pnl, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🌊 Volatility Shock P&L")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            vol_shock_min = st.slider("Min Vol Shock (percentage points)", -15.0, 0.0, -10.0, 0.5)
        with col_v2:
            vol_shock_max = st.slider("Max Vol Shock (percentage points)", 0.0, 15.0, 10.0, 0.5)

        vol_shocks = np.linspace(vol_shock_min, vol_shock_max, 40)
        vol_pnl_df = pnl_sim.vol_pnl(vol_shocks)

        if not vol_pnl_df.empty:
            fig_vol_pnl = go.Figure()
            fig_vol_pnl.add_trace(go.Scatter(
                x=vol_pnl_df["Vol Shock (%)"], y=vol_pnl_df["Full Reprice P&L"],
                name="Full Reprice", line=dict(color="#818cf8", width=3),
            ))
            fig_vol_pnl.add_trace(go.Scatter(
                x=vol_pnl_df["Vol Shock (%)"], y=vol_pnl_df["Vega P&L"],
                name="Vega P&L", line=dict(color="#f472b6", width=2, dash="dash"),
            ))
            fig_vol_pnl.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
            fig_vol_pnl.update_layout(
                title="P&L vs Volatility Shock (pts)",
                xaxis_title="Vol Shock (percentage points)",
                yaxis_title="P&L ($)",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_vol_pnl, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔥 Combined P&L Heatmap (Spot % × Vol pts)")

        combined_spot_pct = np.linspace(spot_shock_min_pct, spot_shock_max_pct, 15)
        combined_spot_abs = spot * combined_spot_pct / 100.0
        combined_vol = np.linspace(vol_shock_min, vol_shock_max, 11)

        pnl_matrix = pnl_sim.combined_pnl_matrix(combined_spot_abs, combined_vol)
        pnl_matrix.index = [f"S {pct:+.1f}%" for pct in combined_spot_pct]
        pnl_matrix.columns = [f"Vol {pts:+.1f}pts" for pts in combined_vol]

        if not pnl_matrix.empty:
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=pnl_matrix.values,
                x=pnl_matrix.columns.tolist(),
                y=pnl_matrix.index.tolist(),
                colorscale=[
                    [0.0, '#ef4444'],
                    [0.25, '#f87171'],
                    [0.5, '#1e1b4b'],
                    [0.75, '#34d399'],
                    [1.0, '#10b981'],
                ],
                colorbar=dict(title="P&L ($)"),
                hovertemplate="Spot: %{y}<br>Vol: %{x}<br>P&L: $%{z:,.2f}<extra></extra>",
            ))
            fig_heatmap.update_layout(
                title="P&L Heatmap: Spot Shock × Vol Shock",
                xaxis_title="Volatility Shock",
                yaxis_title="Spot Shock",
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

    # ==================================================================
    # TAB 5 — Trading Intelligence Layer
    # ==================================================================
    with tab_intelligence:
        st.markdown("### 🧠 Trading Intelligence Layer")
        
        # 1. Position Exposure & Interpretation
        st.markdown("#### 🛡️ Position Risk Interpretation")
        col_risk1, col_risk2 = st.columns([1, 2])
        
        with col_risk1:
            st.markdown(f"**Current Mode:** {position_type} {option_type}")
            unit_delta = model.delta(option_type)
            unit_gamma = model.gamma()
            unit_vega = model.vega()
            
            # Exposure cards
            st.write(f"Net Delta: `{cash.cash_delta():+,.2f}`")
            st.write(f"Net Gamma: `{cash.cash_gamma():+,.2f}`")
            st.write(f"Net Vega: `{cash.cash_vega():+,.2f}`")
            
        with col_risk2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            if position_type == "Long":
                if option_type == "Call":
                    st.write("🟢 **Bullish View**: You benefit from an increase in the spot price.")
                else:
                    st.write("🔴 **Bearish View**: You benefit from a decrease in the spot price.")
                st.write("🚀 **Long Gamma**: You benefit from high volatility (convexity profit).")
                st.write("🌊 **Long Vega**: You profit if implied volatility rises.")
                st.write("⏳ **Theta Decay**: You are paying 'rent' for this position daily.")
            else:
                if option_type == "Call":
                    st.write("🔴 **Bearish/Neutral View**: You profit if the spot stays below the strike.")
                else:
                    st.write("🟢 **Bullish/Neutral View**: You profit if the spot stays above the strike.")
                st.write("📉 **Short Gamma**: Large spot moves hurt you; you prefer a stable market.")
                st.write("🔥 **Short Vega**: A vol spike (fear) will increase the option value you sold, hurting you.")
                st.write("💰 **Theta Income**: You are 'collecting rent' from time decay.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # 2. Scenario Engine
        st.markdown("#### ⚡ Scenario Engine (Shock Buttons)")
        st.markdown("Select shocks to see the combined estimated P&L decomposition.")
        
        col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn6 = st.columns(6)
        
        if col_btn1.button("Spot +1%"): st.session_state.shock_ds_pct += 1.0
        if col_btn2.button("Spot +3%"): st.session_state.shock_ds_pct += 3.0
        if col_btn3.button("Spot +5%"): st.session_state.shock_ds_pct += 5.0
        if col_btn4.button("Vol +1 pt"): st.session_state.shock_dv += 1.0
        if col_btn5.button("Vol +2 pts"): st.session_state.shock_dv += 2.0
        if col_btn6.button("Reset Shocks", type="primary"): reset_shocks()
        
        # Display active shocks
        st.write(f"Active Shocks: **Spot {st.session_state.shock_ds_pct:+.1f}%** | **Vol {st.session_state.shock_dv:+.1f} pts**")
        
        if st.session_state.shock_ds_pct != 0 or st.session_state.shock_dv != 0:
            abs_ds = spot * st.session_state.shock_ds_pct / 100.0
            decomp = pnl_sim.compute_pnl_decomposition(abs_ds, st.session_state.shock_dv)
            
            st.markdown("**Estimated P&L Breakdown ($):**")
            col_dec1, col_dec2, col_dec3, col_dec4 = st.columns(4)
            col_dec1.metric("Delta P&L", f"${decomp['Delta P&L']:+,.2f}")
            col_dec2.metric("Gamma P&L", f"${decomp['Gamma P&L']:+,.2f}")
            col_dec3.metric("Vega P&L", f"${decomp['Vega P&L']:+,.2f}")
            col_dec4.metric("Total Approx", f"${decomp['Total Approx P&L']:+,.2f}")
            
            st.markdown(f"""
            <div style="font-size: 0.85rem; color: #a5b4fc; background: rgba(99, 102, 241, 0.1); padding: 10px; border-radius: 8px; border: 1px solid rgba(99, 102, 241, 0.2);">
                <strong>Quant Formulas used:</strong><br>
                • Δ P&L = Delta × ΔS × Lots × Multiplier × Sign<br>
                • Γ P&L ≈ 0.5 × Gamma × (ΔS)² × Lots × Multiplier × Sign<br>
                • Vega P&L = Vega × ΔVol(pts) × Lots × Multiplier × Sign
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 3. Interview Mode (Cases)
        st.markdown("#### 🎓 Interview Mode (Preloaded Cases)")
        case = st.selectbox("Select a Practice Case:", [
            "Select a case...",
            "Spot moves +5% → Estimate Gamma Impact",
            "Vol +2 pts → Compute Vega Impact",
            "Time passes 1 week → Calculate Theta Bleed"
        ])
        
        if case != "Select a case...":
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            pos_sign = 1.0 if position_type == "Long" else -1.0
            if "Spot moves +5%" in case:
                ds_val = spot * 0.05
                d_pnl = model.delta(option_type) * ds_val * lots * multiplier * pos_sign
                g_pnl = 0.5 * model.gamma() * (ds_val**2) * lots * multiplier * pos_sign
                st.write(f"**Intuition**: A 5% move (${ds_val:,.2f}) creates linear risk and convexity profit/loss.")
                st.write(f"• Linear Delta Impact: **${d_pnl:,.2f}**")
                st.write(f"• Convexity (Gamma) Impact: **${g_pnl:,.2f}**")
                st.write(f"**Formula**: $\\Gamma\\ PnL \\approx 0.5 \\times \\Gamma \\times (\\Delta S)^2$")
            elif "Vol +2 pts" in case:
                v_impact = model.vega() * 2.0 * lots * multiplier * pos_sign
                st.write(f"**Intuition**: Vega measures the price change per 1 percentage point move in volatility.")
                st.write(f"• Total Vega Impact: **${v_impact:,.2f}**")
                st.write("**Rule of Thumb**: Long vega is long fear. If vol spikes 2 pts, your P&L moves by $2 \times \text{Cash Vega}$.")
            elif "1 week" in case:
                t_impact = model.theta(option_type) * 7 * lots * multiplier * pos_sign
                st.write(f"**Intuition**: Theta is the daily cost of holding the option (time decay).")
                st.write(f"• 7-Day Theta Impact: **${t_impact:,.2f}**")
                st.write("**Rule of Thumb**: Theta is the price you pay for the 'right' to benefit from Gamma.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # 4. Expiry Behavior & Pin Risk
        st.markdown("#### 📅 Expiry Behavior (Gamma Risk)")
        st.markdown("See how Delta jumps as spot crosses the Strike near expiry.")
        
        days_to_expiry = st.slider("Days to Expiry", 0.1, 10.0, 1.0, 0.1)
        exp_T = days_to_expiry / 365.0
        exp_spots = np.linspace(strike * 0.95, strike * 1.05, 100)
        
        exp_deltas = []
        for s in exp_spots:
            try:
                m = BlackScholesModel(BlackScholesInputs(s, strike, exp_T, vol_pct/100, rate_pct/100, div_pct/100))
                exp_deltas.append(m.delta(option_type))
            except: exp_deltas.append(np.nan)
            
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(x=exp_spots, y=exp_deltas, name="Delta", line=dict(color="#f472b6", width=3)))
        fig_exp.add_vline(x=strike, line_dash="dash", line_color="white", annotation_text="STRIKE")
        fig_exp.update_layout(title=f"Delta Profile at {days_to_expiry:.1f} Days to Expiry", 
                            xaxis_title="Spot Price", yaxis_title="Delta", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_exp, use_container_width=True)
        st.write("💡 **Pin Risk**: Notice how the Delta curve becomes a step function as time to expiry $\\to 0$. This makes hedging extremely difficult near the strike.")

        st.markdown("---")

        # 5. Forward Intuition Panel
        st.markdown("#### 🔭 Forward Intuition Panel")
        fwd_price = model.forward_price()
        diff = fwd_price - spot
        
        col_fwd1, col_fwd2 = st.columns(2)
        with col_fwd1:
            st.metric("Forward Price", f"{fwd_price:,.4f}")
            st.write(f"Forward Premium: `{diff:+,.4f}` points")
        
        with col_fwd2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write("**Impact of Dividends:**")
            st.write("Rising dividends **reduce** the Forward Price.")
            st.write("• Calls become **cheaper** (you lose dividend income while holding the call).")
            st.write("• Puts become **more expensive**.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        div_shock = st.slider("Dividend Shock (%)", -5.0, 5.0, 0.0, 0.5)
        if div_shock != 0:
            new_q = max(0, (div_pct + div_shock) / 100.0)
            new_model = BlackScholesModel(BlackScholesInputs(
                spot, strike, max(maturity, 1e-6), vol_pct/100, rate_pct/100, new_q, repo_pct/100
            ))
            new_price = new_model.option_price(option_type)
            st.write(f"Price after Dividend Shock: **{new_price:,.4f}** (Change: `{new_price - model.option_price(option_type):+,.4f}`)")

    # ==================================================================
    # TAB 6 — Batch Scaling
    # ==================================================================
    with tab_batch:
        st.markdown("#### 📦 Batch Lot Comparison")
        st.markdown("""
        <div class="info-box">
            Compare how option premium and cash Greeks scale across different position sizes.
            This is essential for portfolio managers sizing their hedges.
        </div>
        """, unsafe_allow_html=True)

        try:
            lot_sizes = [int(x.strip()) for x in batch_lots.split(",") if x.strip()]
        except ValueError:
            lot_sizes = [1, 10, 100, 1000]
            st.warning("Could not parse lot sizes. Using defaults: 1, 10, 100, 1000")

        batch_rows = []
        for lot_size in lot_sizes:
            batch_cash = CashGreeks(model, option_type, position_type, lot_size, multiplier)
            batch_rows.append({
                "Lots": lot_size,
                "Total Premium": signed_total_price / lots * lot_size, # scaled correctly
                "Cash Delta": batch_cash.cash_delta(),
                "Cash Gamma": batch_cash.cash_gamma(),
                "Cash Vega": batch_cash.cash_vega(),
                "Cash Theta": batch_cash.cash_theta(),
                "Cash Rho": batch_cash.cash_rho(),
            })

        batch_df = pd.DataFrame(batch_rows)

        # Format numerically
        for col in batch_df.columns[1:]:
            batch_df[col] = batch_df[col].apply(lambda x: format_number(x, 2))

        st.dataframe(batch_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        batch_numeric = pd.DataFrame(batch_rows)
        fig_batch = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Total Premium by Lot Size",
                                                  "Cash Delta by Lot Size"])

        fig_batch.add_trace(go.Bar(
            x=[str(l) for l in lot_sizes],
            y=batch_numeric["Total Premium"],
            name="Premium",
            marker_color="#818cf8",
        ), row=1, col=1)

        fig_batch.add_trace(go.Bar(
            x=[str(l) for l in lot_sizes],
            y=batch_numeric["Cash Delta"],
            name="Cash Delta",
            marker_color="#34d399",
        ), row=1, col=2)

        fig_batch.update_layout(
            **PLOTLY_LAYOUT,
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_batch, use_container_width=True)


# ======================================================================
# Methodology & Help
# ======================================================================
with st.expander("📚 Methodology & Financial Logic"):
    st.markdown("""
    ### The Black-Scholes Model
    This application uses the analytical Black-Scholes-Merton model for European options.
    
    **Core Assumptions:**
    *   **Geometric Brownian Motion**: Underlying price follows $dS = (r-q)Sdt + \sigma SdW$.
    *   **Log-normal Distribution**: Asset prices at expiry are log-normally distributed.
    *   **Continuous Trading**: No transaction costs or liquidity constraints.
    *   **European Style**: Options can only be exercised at the exact moment of expiration.

    ### The Greeks & Risk Management
    *   **Delta ($\Delta$)**: The hedge ratio. Represents the equivalent number of shares of the underlying.
    *   **Gamma ($\Gamma$)**: The 'convexity' of the option. High gamma indicates that delta changes rapidly as the spot moves.
    *   **Vega ($\nu$)**: Sensitivity to a 1% change in Implied Volatility.
    *   **Theta ($\Theta$)**: The cost of 'renting' the optionality (time decay).

    ### Cash Greeks Calculation
    Traders manage risk in dollar terms. We convert unit Greeks to **Cash Greeks**:
    *   **Cash Delta** = $\Delta \times \text{Spot} \times \text{Lots} \times \text{Multiplier}$
    *   **Cash Gamma** = $\Gamma \times \text{Spot}^2 \times \text{Lots} \times \text{Multiplier}$

    ### P&L Approximation (Taylor Expansion)
    The simulated P&L uses the second-order Taylor expansion:
    $$\Delta PnL \approx \Delta \cdot \delta S + \frac{1}{2} \Gamma \cdot (\delta S)^2 + \text{Vega} \cdot \delta \sigma + \Theta \cdot \delta t$$
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6366f1; font-size: 0.85rem; padding: 16px 0;">
    <strong>Derivative Pricer</strong> — Built with Black-Scholes Model &amp; Streamlit<br>
    <span style="color: #a5b4fc;">European Options • Cash Greeks • P&L Simulation</span>
</div>
""", unsafe_allow_html=True)
