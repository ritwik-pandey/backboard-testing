"""
Monte Carlo Simulation Engine
==============================
This module contains all Monte Carlo simulation logic, data loading, 
distribution building, and visualization functions.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import truncnorm, lognorm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# =====================================================
# CONFIGURATION
# =====================================================

CSV_PATH = "./monte_carlo_final_data 2.csv"
NUM_SIMULATIONS = 10000
np.random.seed(42)

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def safe_float(value):
    """Safely convert value to float, handling N/A, empty strings, and None."""
    if pd.isna(value) or value in ("N/A", "", None):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "")
        return float(value)
    except (ValueError, TypeError):
        return None


def truncated_normal(mean, std, min_val, max_val, size=1):
    """Generate samples from a truncated normal distribution."""
    if std <= 0:
        return np.full(size, mean)
    a, b = (min_val - mean) / std, (max_val - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

# =====================================================
# DATA LOADING
# =====================================================

def load_financials(csv_path):
    """Load financial data from CSV file."""
    df = pd.read_csv(csv_path)

    # Remove reporting artefacts
    if "period" in df.columns:
        df = df[~df["period"].str.contains("Unknown", na=False)]

    cols = [
        "revenue_total",
        "gross_profit",
        "operating_expenses_total",
        "net_income",
        "cash_from_operations",
        "capital_expenditure",
        "cash_end_period",
        "employee_count"
    ]

    for c in cols:
        df[c] = df[c].apply(safe_float)

    return df.dropna(subset=["revenue_total"])

# =====================================================
# HISTORICAL METRICS DERIVATION
# =====================================================

def derive_historical_metrics(df):
    """Derive historical metrics from financial data."""
    # Latest valid cash row
    cash_valid = df[df["cash_end_period"] > 0]
    if len(cash_valid) == 0:
        raise ValueError("No valid non-zero cash rows found")

    latest = cash_valid.iloc[-1]

    # SAFE employee handling
    raw_emp = latest["employee_count"]
    employee_count = int(raw_emp) if raw_emp and raw_emp > 0 else 1

    revenues = df["revenue_total"].values

    # Quarterly → annualized revenue growth
    revenue_growths = []
    for i in range(1, len(revenues)):
        if revenues[i-1] > 0:
            q_growth = (revenues[i] - revenues[i-1]) / revenues[i-1]
            revenue_growths.append(q_growth * 4)

    gross_margins = df["gross_profit"] / df["revenue_total"]
    opex_ratios = df["operating_expenses_total"] / df["revenue_total"]

    cash_conversion = df["cash_from_operations"] / df["net_income"].replace(0, np.nan)
    cash_conversion = cash_conversion.dropna()

    capex_ratios = (df["capital_expenditure"].abs() / df["revenue_total"]).dropna()

    # 🔑 DATA-DRIVEN CapEx cap (THIS IS THE KEY FIX)
    capex_cap = float(np.percentile(capex_ratios, 95))

    base = {
        "revenue": latest["revenue_total"],
        "cash": latest["cash_end_period"],
        "employee_count": employee_count,

        "revenue_growth_mean": float(np.mean(revenue_growths)),
        "revenue_growth_std": float(np.std(revenue_growths)) or 0.08,
        "revenue_growth_min": float(np.percentile(revenue_growths, 5)),
        "revenue_growth_max": float(np.percentile(revenue_growths, 95)),

        "gross_margin_mean": float(gross_margins.mean()),
        "gross_margin_std": float(gross_margins.std()) or 0.04,

        "opex_ratio_mean": float(opex_ratios.mean()),
        "opex_ratio_std": float(opex_ratios.std()) or 0.05,

        "cash_conversion_mean": float(cash_conversion.mean()),
        "cash_conversion_std": float(cash_conversion.std()) or 0.15,

        "capex_ratio_mean": float(capex_ratios.mean()),
        "capex_ratio_std": float(capex_ratios.std()) or 0.03,

        # 🔥 learned from CSV, not assumed
        "capex_cap": capex_cap
    }

    # Salary & liquidity rules (still policy, not math)
    base["average_annual_salary_cost"] = (
        latest["operating_expenses_total"] * 0.65
    ) / employee_count

    base["min_cash_buffer_for_hiring"] = latest["operating_expenses_total"] * 0.5

    return base

# =====================================================
# DISTRIBUTION BUILDING
# =====================================================

def build_distributions(base):
    """Build probability distributions from historical metrics."""
    return {
        "revenue_growth": {
            "mean": base["revenue_growth_mean"],
            "std": base["revenue_growth_std"],
            "min_val": base["revenue_growth_min"],
            "max_val": base["revenue_growth_max"]
        },
        "gross_margin": {
            "mean": base["gross_margin_mean"],
            "std": base["gross_margin_std"],
            "min_val": max(0.05, base["gross_margin_mean"] - 3 * base["gross_margin_std"]),
            "max_val": min(0.95, base["gross_margin_mean"] + 3 * base["gross_margin_std"])
        },
        "opex_ratio": {
            "mean": base["opex_ratio_mean"],
            "std": base["opex_ratio_std"],
            "min_val": max(0.05, base["opex_ratio_mean"] - 3 * base["opex_ratio_std"]),
            "max_val": min(0.90, base["opex_ratio_mean"] + 3 * base["opex_ratio_std"])
        },
        "cash_conversion": {
            "mean": base["cash_conversion_mean"],
            "std": base["cash_conversion_std"],
            "min_val": max(0.20, base["cash_conversion_mean"] - 3 * base["cash_conversion_std"]),
            "max_val": min(2.00, base["cash_conversion_mean"] + 3 * base["cash_conversion_std"])
        },
        "capex_ratio": {
            "mean": base["capex_ratio_mean"],
            "std": base["capex_ratio_std"]
        }
    }

# =====================================================
# SINGLE SIMULATION
# =====================================================

def run_simulation(base, dists):
    """Run a single Monte Carlo simulation."""
    revenue = base["revenue"]
    cash = base["cash"]
    employees = base["employee_count"]
    hired = False

    # --- Revenue ---
    g = truncated_normal(**dists["revenue_growth"])[0]
    revenue *= (1 + g)

    # --- Operations ---
    margin = truncated_normal(**dists["gross_margin"])[0]
    opex_ratio = truncated_normal(**dists["opex_ratio"])[0]
    cash_conv = truncated_normal(**dists["cash_conversion"])[0]

    gross_profit = revenue * margin
    opex = revenue * opex_ratio
    operating_income = gross_profit - opex
    cfo = operating_income * cash_conv

    # --- CapEx (DATA-DRIVEN CAP) ---
    capex_ratio = lognorm(
        s=dists["capex_ratio"]["std"] / max(dists["capex_ratio"]["mean"], 0.001),
        scale=dists["capex_ratio"]["mean"]
    ).rvs()

    capex_ratio = min(capex_ratio, base["capex_cap"])
    capex = revenue * capex_ratio

    # ALL FOUR CONDITIONS MUST BE TRUE FOR HIRING TO OCCUR
    if g > 0.05 and (cash > base["min_cash_buffer_for_hiring"] or cash > 0.1 * revenue):
        rev_per_emp = revenue / employees
        if rev_per_emp > base["average_annual_salary_cost"] * 2:
            hired = True
            cash -= base["average_annual_salary_cost"] * 0.5  # 6 months upfront cost
            employees += 1

    # --- Cash update ---
    cash += cfo - capex
    
    # Calculate additional metrics for comprehensive analysis
    gross_margin_pct = (gross_profit / revenue * 100) if revenue > 0 else 0
    operating_margin_pct = (operating_income / revenue * 100) if revenue > 0 else 0
    ebitda = operating_income  # Simplified (no depreciation/amortization in model)
    cash_flow = cfo - capex

    return {
        "cash": cash,
        "revenue": revenue,
        "hired": hired,
        "gross_margin": gross_margin_pct,
        "operating_margin": operating_margin_pct,
        "ebitda": ebitda,
        "cash_flow": cash_flow,
        "revenue_growth": g,
        "opex": opex,
        "capex": capex,
        "cfo": cfo
    }

# =====================================================
# MONTE CARLO ENGINE
# =====================================================

def run_monte_carlo_simulations(base: dict, dists: dict, num_sims: int = None) -> Dict:
    """Run comprehensive Monte Carlo simulations and return all metrics."""
    if num_sims is None:
        num_sims = NUM_SIMULATIONS
    
    # Optimized: Use list comprehension and direct array conversion
    simulations = [run_simulation(base, dists) for _ in range(num_sims)]
    
    # Convert to arrays for analysis (faster with list comprehension)
    cash = np.array([s["cash"] for s in simulations])
    revenue = np.array([s["revenue"] for s in simulations])
    gross_margin = np.array([s["gross_margin"] for s in simulations])
    operating_margin = np.array([s["operating_margin"] for s in simulations])
    ebitda = np.array([s["ebitda"] for s in simulations])
    cash_flow = np.array([s["cash_flow"] for s in simulations])
    revenue_growth = np.array([s["revenue_growth"] for s in simulations])
    opex = np.array([s["opex"] for s in simulations])
    capex = np.array([s["capex"] for s in simulations])
    cfo = np.array([s["cfo"] for s in simulations])
    hired = np.array([s["hired"] for s in simulations])
    
    return {
        "base": base,
        "simulations": simulations,
        "cash": cash,
        "revenue": revenue,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "ebitda": ebitda,
        "cash_flow": cash_flow,
        "revenue_growth": revenue_growth,
        "opex": opex,
        "capex": capex,
        "cfo": cfo,
        "hired": hired
    }


def run_engine(return_cash_array=False):
    """Run the complete Monte Carlo engine."""
    df = load_financials(CSV_PATH)
    base = derive_historical_metrics(df)
    dists = build_distributions(base)

    sim_data = run_monte_carlo_simulations(base, dists)
    cash = sim_data["cash"]
    revenue = sim_data["revenue"]
    hire_results = sim_data["hired"]

    results = {
        "starting_cash": base["cash"],
        "starting_revenue": base["revenue"],
        "median_ending_cash": float(np.median(cash)),
        "p5_ending_cash": float(np.percentile(cash, 5)),
        "p10_ending_cash": float(np.percentile(cash, 10)),
        "p90_ending_cash": float(np.percentile(cash, 90)),
        "p95_ending_cash": float(np.percentile(cash, 95)),
        "probability_cash_negative": float(np.mean(cash < 0)),
        "expected_revenue_growth": float((np.mean(revenue) - base["revenue"]) / base["revenue"]),
        "probability_should_hire": float(np.mean(hire_results)),
        "capex_cap_used": base["capex_cap"],
        "hiring_parameters": {
            "revenue_growth_threshold": "> 5% (0.05)",
            "cash_buffer_requirement": f"min_cash_buffer ({base['min_cash_buffer_for_hiring']:,.0f}) OR 10% of revenue",
            "revenue_per_employee_threshold": f"> 2x salary cost ({base['average_annual_salary_cost'] * 2:,.0f})",
            "min_cash_buffer": base["min_cash_buffer_for_hiring"],
            "average_salary_cost": base["average_annual_salary_cost"]
        }
    }
    
    if return_cash_array:
        return results, cash
    return results

# =====================================================
# VISUALIZATION
# =====================================================

def plot_monte_carlo_bell_curve(results: dict, cash_outcomes: np.ndarray):
    """
    Plot the Monte Carlo simulation results as a bell curve (normal distribution)
    showing the distribution of ending cash positions across all simulations.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create histogram (empirical distribution)
    n_bins = 60
    counts, bins, patches = ax.hist(cash_outcomes, bins=n_bins, density=True, 
                                    alpha=0.7, color='steelblue', edgecolor='black', 
                                    linewidth=0.5, label='Monte Carlo Simulation Results')
    
    # Overlay theoretical normal distribution curve
    mean_cash = np.mean(cash_outcomes)
    std_cash = np.std(cash_outcomes)
    x_curve = np.linspace(cash_outcomes.min(), cash_outcomes.max(), 1000)
    y_curve = (1 / (std_cash * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_curve - mean_cash) / std_cash) ** 2)
    ax.plot(x_curve, y_curve, 'r-', linewidth=2.5, label='Normal Distribution Fit', alpha=0.8)
    
    # Mark key percentiles
    p5 = results["p5_ending_cash"]
    p10 = results["p10_ending_cash"]
    median = results["median_ending_cash"]
    p90 = results["p90_ending_cash"]
    p95 = results["p95_ending_cash"]
    starting = results["starting_cash"]
    
    # Vertical lines for percentiles
    ax.axvline(p5, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'5th Percentile: {p5:,.0f}')
    ax.axvline(p10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'10th Percentile: {p10:,.0f}')
    ax.axvline(median, color='green', linestyle='-', linewidth=3, alpha=0.9, label=f'Median (50th): {median:,.0f}')
    ax.axvline(p90, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'90th Percentile: {p90:,.0f}')
    ax.axvline(p95, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'95th Percentile: {p95:,.0f}')
    ax.axvline(starting, color='purple', linestyle=':', linewidth=2.5, alpha=0.8, label=f'Starting Cash: {starting:,.0f}')
    
    # Mark zero line if negative values exist
    if cash_outcomes.min() < 0:
        ax.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.6, label='Zero Cash (Risk Threshold)')
        # Fill negative cash region
        ax.axvspan(cash_outcomes.min(), 0, alpha=0.15, color='red', label='Negative Cash Zone')
    
    # Fill area under normal curve
    ax.fill_between(x_curve, 0, y_curve, alpha=0.2, color='steelblue')
    
    # Labels and title
    ax.set_xlabel('Ending Cash Position', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=13, fontweight='bold')
    ax.set_title('Monte Carlo Simulation: Ending Cash Distribution (Bell Curve)\n' + 
                 f'Based on {NUM_SIMULATIONS:,} Simulations', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=10, framealpha=0.95, ncol=2)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Mean: {mean_cash:,.0f}
Std Dev: {std_cash:,.0f}
Min: {cash_outcomes.min():,.0f}
Max: {cash_outcomes.max():,.0f}
Prob < 0: {results['probability_cash_negative']*100:.2f}%
Prob Should Hire: {results.get('probability_should_hire', 0)*100:.2f}%"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black'))
    
    # Add hiring parameters explanation
    hiring_params = results.get("hiring_parameters", {})
    if hiring_params:
        hiring_text = f"""Hiring Decision Parameters:
1. Revenue Growth > 5%
2. Cash > {hiring_params.get('min_cash_buffer', 0):,.0f} OR 10% of revenue
3. Revenue/Employee > {hiring_params.get('average_salary_cost', 0)*2:,.0f}
4. Valid employee count"""
        
        ax.text(0.98, 0.98, hiring_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='blue'))
    
    plt.tight_layout()
    plt.savefig('monte_carlo_bell_curve.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graph saved as 'monte_carlo_bell_curve.png'")
    plt.show()


def plot_revenue_distribution(sim_data: Dict):
    """Plot revenue distribution for revenue-related questions"""
    revenue = sim_data["revenue"]
    base = sim_data["base"]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_bins = 50
    ax.hist(revenue, bins=n_bins, density=True, alpha=0.7, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='Revenue Distribution')
    
    # Overlay normal curve
    mean_rev = np.mean(revenue)
    std_rev = np.std(revenue)
    x_curve = np.linspace(revenue.min(), revenue.max(), 1000)
    y_curve = (1 / (std_rev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_curve - mean_rev) / std_rev) ** 2)
    ax.plot(x_curve, y_curve, 'r-', linewidth=2, label='Normal Fit', alpha=0.8)
    
    # Mark key percentiles
    p10 = np.percentile(revenue, 10)
    p50 = np.percentile(revenue, 50)
    p90 = np.percentile(revenue, 90)
    
    ax.axvline(p10, color='orange', linestyle='--', linewidth=2, label=f'P10: {p10:,.0f}')
    ax.axvline(p50, color='green', linestyle='-', linewidth=2, label=f'Median: {p50:,.0f}')
    ax.axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'P90: {p90:,.0f}')
    ax.axvline(base["revenue"], color='purple', linestyle=':', linewidth=2, 
               label=f'Current: {base["revenue"]:,.0f}')
    
    ax.set_xlabel('Revenue', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Revenue Distribution - Monte Carlo Simulation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('revenue_distribution.png', dpi=300, bbox_inches='tight')
    print("\n📊 Revenue distribution graph saved as 'revenue_distribution.png'")
    plt.show()


def plot_cash_distribution(sim_data: Dict):
    """Plot cash distribution for cash/liquidity-related questions"""
    cash = sim_data["cash"]
    base = sim_data["base"]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    n_bins = 50
    ax.hist(cash, bins=n_bins, density=True, alpha=0.7, color='steelblue', 
            edgecolor='black', linewidth=0.5, label='Cash Distribution')
    
    # Overlay normal curve
    mean_cash = np.mean(cash)
    std_cash = np.std(cash)
    x_curve = np.linspace(cash.min(), cash.max(), 1000)
    y_curve = (1 / (std_cash * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_curve - mean_cash) / std_cash) ** 2)
    ax.plot(x_curve, y_curve, 'r-', linewidth=2, label='Normal Fit', alpha=0.8)
    
    # Mark key percentiles
    p5 = np.percentile(cash, 5)
    p50 = np.percentile(cash, 50)
    p95 = np.percentile(cash, 95)
    
    ax.axvline(p5, color='red', linestyle='--', linewidth=2, label=f'P5: {p5:,.0f}')
    ax.axvline(p50, color='green', linestyle='-', linewidth=2, label=f'Median: {p50:,.0f}')
    ax.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'P95: {p95:,.0f}')
    ax.axvline(base["cash"], color='purple', linestyle=':', linewidth=2, 
               label=f'Current: {base["cash"]:,.0f}')
    
    if cash.min() < 0:
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Zero Cash')
        ax.axvspan(cash.min(), 0, alpha=0.15, color='red', label='Negative Zone')
    
    ax.set_xlabel('Cash Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax.set_title('Cash Distribution - Monte Carlo Simulation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('cash_distribution.png', dpi=300, bbox_inches='tight')
    print("\n📊 Cash distribution graph saved as 'cash_distribution.png'")
    plt.show()
