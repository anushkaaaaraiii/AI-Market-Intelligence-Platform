"""
Fundamental feature engineering — Literature Review Section 9

Features:
  - FCF yield     → Primary fundamental filter (confirmed on NSE 2005-2020)
  - P/E z-score   → Relative to sector peers (NOT absolute P/E)
  - D/E ratio     → Hard exclusion filter (D/E > 2.0 in Bull, > 1.0 in Bear)
  - Revenue QoQ   → Growth momentum

Literature Review verdict:
  - Fundamentals are SCREENING FILTERS, not timing signals
  - FCF yield outperformed NSE averages by 4-7% annually
  - P/E value factor has 12-18 month realization lag
  - D/E is critical post-IL&FS (2018)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from backend.core.logging_config import get_logger

logger = get_logger(__name__)


def compute_pe_zscore(
    pe_ratio: float | None,
    sector_pe_values: list[float],
) -> float | None:
    """
    P/E z-score vs sector peers.
    PRD Section 5.2: pe_zscore — Z-score vs sector peers (not absolute P/E).

    Literature Review Section 9.1:
    - Fama & French (1992): Low P/E outperforms over 5-10 year horizons
    - Mohanty (2002): Value factor premium exists on NSE with 12-18 month lag
    - NOT a useful short-term signal (< 30 days)
    """
    if pe_ratio is None or not sector_pe_values:
        return None

    # Remove outliers (P/E < 0 or > 200)
    valid_pes = [p for p in sector_pe_values if 0 < p < 200]
    if len(valid_pes) < 3:
        return None

    mean_pe = np.mean(valid_pes)
    std_pe = np.std(valid_pes)

    if std_pe == 0:
        return 0.0

    return (pe_ratio - mean_pe) / std_pe


def compute_fcf_yield(fcf: float | None, market_cap: float | None) -> float | None:
    """
    FCF Yield = Free Cash Flow / Market Cap

    Literature Review Section 9.2:
    - FCF yield is one of the strongest fundamental predictors
    - High FCF yield stocks outperformed NSE averages by 4-7% annually (2005-2020)
    - Valid screening criterion for fundamental classifier
    """
    if fcf is None or market_cap is None or market_cap <= 0:
        return None
    return fcf / market_cap


def compute_de_exclusion(
    de_ratio: float | None,
    regime_is_bull: bool,
) -> bool:
    """
    Debt-to-Equity exclusion filter.

    Literature Review Section 9.3:
    - D/E > 2.0 = hard exclusion in Bull markets
    - D/E > 1.0 = hard exclusion in Bear/Volatile regimes
    - Post-IL&FS (2018): high leverage mid-caps suffered severe deratings
    """
    if de_ratio is None:
        return False  # Don't exclude if data unavailable

    if regime_is_bull:
        return de_ratio > 2.0
    else:
        return de_ratio > 1.0


def compute_fundamental_features(
    symbol: str,
    fundamentals: dict,
    sector_fundamentals: list[dict],
    regime_is_bull: bool,
) -> dict:
    """
    Compute all fundamental features for a stock.

    Args:
        symbol: Stock symbol
        fundamentals: Dict with pe_ratio, fcf, market_cap, de_ratio, revenue_qoq_growth
        sector_fundamentals: List of fundamentals for all stocks in the same sector
        regime_is_bull: Current market regime

    Returns:
        Dict with fcf_yield, pe_zscore, de_ratio, excluded (bool)
    """
    # FCF Yield
    fcf_yield = compute_fcf_yield(
        fundamentals.get("fcf"),
        fundamentals.get("market_cap"),
    )

    # P/E Z-score vs sector
    sector_pes = [
        s.get("pe_ratio")
        for s in sector_fundamentals
        if s.get("pe_ratio") is not None
    ]
    pe_zscore = compute_pe_zscore(fundamentals.get("pe_ratio"), sector_pes)

    # D/E ratio and exclusion check
    de_ratio = fundamentals.get("debt_to_equity")
    if de_ratio is not None:
        de_ratio = de_ratio / 100.0 if de_ratio > 10 else de_ratio  # Normalize if in %

    excluded = compute_de_exclusion(de_ratio, regime_is_bull)

    return {
        "symbol": symbol,
        "fcf_yield": fcf_yield,
        "pe_zscore": pe_zscore,
        "de_ratio": de_ratio,
        "excluded_by_de": excluded,
        "revenue_qoq_growth": fundamentals.get("revenue_qoq_growth"),
    }


def get_sector_fundamentals(db: Session, sector: str) -> list[dict]:
    """Fetch fundamentals for all stocks in a sector (for P/E z-score computation)."""
    result = db.execute(
        text("""
            SELECT symbol, pe_zscore, fcf_yield, de_ratio
            FROM features_daily
            WHERE symbol IN (SELECT symbol FROM stocks WHERE sector = :sector AND is_active = TRUE)
            AND time = (SELECT MAX(time) FROM features_daily)
        """),
        {"sector": sector},
    )
    return [dict(row._mapping) for row in result]
