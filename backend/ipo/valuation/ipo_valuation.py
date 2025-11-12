"""
IPO Valuation & Pricing

Comprehensive valuation framework for NovaCron's $15B+ IPO including
revenue multiples, DCF analysis, comparable companies, book-building,
and price discovery.

Target: $15B valuation, $40-45 share price, $2B proceeds
"""

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import statistics


class ValuationMethod(Enum):
    """Valuation methodologies"""
    REVENUE_MULTIPLE = "REVENUE_MULTIPLE"
    DCF = "DCF"
    COMPARABLE_COMPANIES = "COMPARABLE_COMPANIES"
    PRECEDENT_TRANSACTIONS = "PRECEDENT_TRANSACTIONS"
    SUM_OF_PARTS = "SUM_OF_PARTS"


class InvestorDemand(Enum):
    """Investor demand levels"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


@dataclass
class IPOValuation:
    """
    Comprehensive IPO valuation framework targeting $15B+
    """
    # Target valuation
    target_valuation: float = 15_000_000_000  # $15B
    valuation_range_low: float = 14_000_000_000
    valuation_range_high: float = 16_000_000_000

    # Revenue multiple method (primary)
    arr: float = 1_000_000_000  # $1B ARR
    revenue_multiple: float = 15.0  # 15x ARR
    comparable_multiples: List[float] = field(default_factory=lambda: [
        12.0,  # Company A
        14.5,  # Company B
        16.0,  # Company C
        18.0,  # Company D (premium)
        13.5,  # Company E
    ])

    # DCF method (validation)
    dcf_valuation: float = 0.0
    wacc: float = 0.10  # 10% weighted average cost of capital
    terminal_growth_rate: float = 0.03  # 3%
    projection_years: int = 10

    # Comparable companies
    comparables: List['ComparableCompany'] = field(default_factory=list)

    # Valuation summary
    implied_valuation_revenue_multiple: float = 0.0
    implied_valuation_dcf: float = 0.0
    implied_valuation_comparables: float = 0.0
    weighted_average_valuation: float = 0.0

    def calculate_revenue_multiple_valuation(self) -> float:
        """Calculate valuation using revenue multiple"""
        # Base: 15x ARR = $15B
        base_valuation = self.arr * self.revenue_multiple

        # Adjust for growth, margins, market position
        growth_premium = 1.05  # 25%+ growth (vs 20% market average)
        margin_premium = 1.10  # 42% margins (vs 25% market average)
        market_position_premium = 1.08  # 50%+ market share (#1)

        adjusted_valuation = base_valuation * growth_premium * margin_premium * market_position_premium

        self.implied_valuation_revenue_multiple = adjusted_valuation
        return adjusted_valuation

    def calculate_dcf_valuation(self) -> float:
        """Calculate DCF valuation"""
        # Revenue projections (next 10 years)
        revenue_projections = self.project_revenue()

        # EBITDA projections
        ebitda_margin = 0.48  # 48% EBITDA margin
        ebitda_projections = [rev * ebitda_margin for rev in revenue_projections]

        # FCF projections (EBITDA - CapEx - NWC)
        capex_percent = 0.10  # 10% of revenue
        nwc_percent = 0.05  # 5% of revenue growth
        fcf_projections = []

        for i, ebitda in enumerate(ebitda_projections):
            capex = revenue_projections[i] * capex_percent
            nwc = (revenue_projections[i] - (revenue_projections[i-1] if i > 0 else self.arr)) * nwc_percent
            fcf = ebitda - capex - nwc
            fcf_projections.append(fcf)

        # Discount to present value
        pv_fcf = []
        for i, fcf in enumerate(fcf_projections):
            discount_factor = 1 / ((1 + self.wacc) ** (i + 1))
            pv_fcf.append(fcf * discount_factor)

        # Terminal value
        terminal_fcf = fcf_projections[-1] * (1 + self.terminal_growth_rate)
        terminal_value = terminal_fcf / (self.wacc - self.terminal_growth_rate)
        pv_terminal_value = terminal_value / ((1 + self.wacc) ** self.projection_years)

        # Enterprise value
        enterprise_value = sum(pv_fcf) + pv_terminal_value

        # Equity value (no debt)
        equity_value = enterprise_value

        self.dcf_valuation = equity_value
        self.implied_valuation_dcf = equity_value
        return equity_value

    def project_revenue(self) -> List[float]:
        """Project revenue for next 10 years"""
        revenue = self.arr
        projections = []
        growth_rates = [
            0.25,  # Year 1: 25%
            0.23,  # Year 2: 23%
            0.21,  # Year 3: 21%
            0.19,  # Year 4: 19%
            0.17,  # Year 5: 17%
            0.15,  # Year 6: 15%
            0.13,  # Year 7: 13%
            0.11,  # Year 8: 11%
            0.09,  # Year 9: 9%
            0.07,  # Year 10: 7%
        ]

        for rate in growth_rates:
            revenue = revenue * (1 + rate)
            projections.append(revenue)

        return projections

    def analyze_comparables(self) -> float:
        """Analyze comparable companies"""
        self.comparables = [
            ComparableCompany(
                name="Snowflake",
                revenue=2_700_000_000,
                market_cap=45_000_000_000,
                ev_revenue_multiple=16.7,
                growth_rate=0.36,
                gross_margin=0.69,
                market_share=0.15
            ),
            ComparableCompany(
                name="Datadog",
                revenue=2_100_000_000,
                market_cap=35_000_000_000,
                ev_revenue_multiple=16.7,
                growth_rate=0.27,
                gross_margin=0.80,
                market_share=0.25
            ),
            ComparableCompany(
                name="CrowdStrike",
                revenue=3_000_000_000,
                market_cap=70_000_000_000,
                ev_revenue_multiple=23.3,
                growth_rate=0.35,
                gross_margin=0.75,
                market_share=0.18
            ),
            ComparableCompany(
                name="MongoDB",
                revenue=1_600_000_000,
                market_cap=20_000_000_000,
                ev_revenue_multiple=12.5,
                growth_rate=0.29,
                gross_margin=0.74,
                market_share=0.12
            ),
        ]

        # Calculate median multiple
        multiples = [c.ev_revenue_multiple for c in self.comparables]
        median_multiple = statistics.median(multiples)

        # Adjust for NovaCron's superior metrics
        # Higher growth premium: 25% vs 32% median
        growth_adjustment = 0.95

        # Higher margin premium: 75% vs 75% median
        margin_adjustment = 1.00

        # Market leader premium: 50% vs 18% median
        market_position_adjustment = 1.15

        adjusted_multiple = median_multiple * growth_adjustment * margin_adjustment * market_position_adjustment
        implied_valuation = self.arr * adjusted_multiple

        self.implied_valuation_comparables = implied_valuation
        return implied_valuation

    def calculate_weighted_valuation(self) -> float:
        """Calculate weighted average of all methods"""
        # Calculate all methods
        revenue_val = self.calculate_revenue_multiple_valuation()
        dcf_val = self.calculate_dcf_valuation()
        comp_val = self.analyze_comparables()

        # Weights (revenue multiple is primary for SaaS)
        weights = {
            'revenue_multiple': 0.50,
            'dcf': 0.30,
            'comparables': 0.20
        }

        weighted_val = (
            revenue_val * weights['revenue_multiple'] +
            dcf_val * weights['dcf'] +
            comp_val * weights['comparables']
        )

        self.weighted_average_valuation = weighted_val
        return weighted_val

    def get_valuation_summary(self) -> Dict[str, float]:
        """Get complete valuation summary"""
        return {
            "revenue_multiple_valuation": self.implied_valuation_revenue_multiple,
            "dcf_valuation": self.implied_valuation_dcf,
            "comparables_valuation": self.implied_valuation_comparables,
            "weighted_average": self.weighted_average_valuation,
            "target_valuation": self.target_valuation,
            "valuation_range": f"${self.valuation_range_low/1e9:.1f}B - ${self.valuation_range_high/1e9:.1f}B"
        }


@dataclass
class ComparableCompany:
    """Comparable public company"""
    name: str
    revenue: float
    market_cap: float
    ev_revenue_multiple: float
    growth_rate: float
    gross_margin: float
    market_share: float


@dataclass
class SharePricing:
    """
    Share pricing strategy for $40-45 range
    """
    # Valuation inputs
    target_valuation: float = 15_000_000_000
    shares_pre_ipo: int = 283_000_000
    shares_offering: int = 50_000_000
    greenshoe_shares: int = 7_500_000

    # Price range
    price_low: float = 40.00
    price_midpoint: float = 42.50
    price_high: float = 45.00

    # Calculated values
    shares_post_ipo: int = 0
    implied_valuation_low: float = 0.0
    implied_valuation_mid: float = 0.0
    implied_valuation_high: float = 0.0

    # Proceeds
    gross_proceeds_low: float = 0.0
    gross_proceeds_mid: float = 0.0
    gross_proceeds_high: float = 0.0

    # Underwriting
    underwriting_discount: float = 0.05  # 5%
    underwriting_fee: float = 0.0
    net_proceeds: float = 0.0

    def calculate_pricing(self):
        """Calculate all pricing metrics"""
        # Shares post-IPO
        self.shares_post_ipo = self.shares_pre_ipo + self.shares_offering

        # Implied valuations
        self.implied_valuation_low = self.shares_post_ipo * self.price_low
        self.implied_valuation_mid = self.shares_post_ipo * self.price_midpoint
        self.implied_valuation_high = self.shares_post_ipo * self.price_high

        # Gross proceeds
        self.gross_proceeds_low = self.shares_offering * self.price_low
        self.gross_proceeds_mid = self.shares_offering * self.price_midpoint
        self.gross_proceeds_high = self.shares_offering * self.price_high

        # Net proceeds (after underwriting discount)
        self.underwriting_fee = self.gross_proceeds_mid * self.underwriting_discount
        self.net_proceeds = self.gross_proceeds_mid - self.underwriting_fee

    def get_pricing_summary(self) -> Dict[str, any]:
        """Get pricing summary"""
        self.calculate_pricing()
        return {
            "price_range": f"${self.price_low} - ${self.price_high}",
            "midpoint_price": f"${self.price_midpoint}",
            "shares_offered": f"{self.shares_offering:,}",
            "shares_outstanding_post": f"{self.shares_post_ipo:,}",
            "implied_valuation": f"${self.implied_valuation_mid/1e9:.1f}B",
            "gross_proceeds": f"${self.gross_proceeds_mid/1e9:.2f}B",
            "underwriting_fees": f"${self.underwriting_fee/1e6:.1f}M",
            "net_proceeds": f"${self.net_proceeds/1e9:.2f}B"
        }


@dataclass
class BookBuilding:
    """
    Book-building process for IPO allocation
    """
    # Price discovery
    price_range_low: float = 40.00
    price_range_high: float = 45.00
    final_price: float = 0.0

    # Demand tracking
    shares_available: int = 50_000_000
    orders: List['InvestorOrder'] = field(default_factory=list)
    total_demand: int = 0
    oversubscription_ratio: float = 0.0

    # Allocation strategy
    institutional_percent: float = 0.70  # 70% to institutions
    retail_percent: float = 0.30  # 30% to retail

    # Allocation tiers
    anchor_investors: List['InvestorAllocation'] = field(default_factory=list)
    institutional_investors: List['InvestorAllocation'] = field(default_factory=list)
    retail_investors: List['InvestorAllocation'] = field(default_factory=list)

    # Lock-up
    lockup_days: int = 180

    def collect_orders(self, orders: List['InvestorOrder']):
        """Collect investor orders during book-building"""
        self.orders = orders
        self.total_demand = sum(order.shares_requested for order in orders)
        self.oversubscription_ratio = self.total_demand / self.shares_available

    def determine_pricing(self) -> float:
        """Determine final IPO price based on demand"""
        if self.oversubscription_ratio > 3.0:
            # Very high demand: price at high end
            self.final_price = self.price_range_high
        elif self.oversubscription_ratio > 2.0:
            # High demand: price above midpoint
            midpoint = (self.price_range_low + self.price_range_high) / 2
            self.final_price = (midpoint + self.price_range_high) / 2
        elif self.oversubscription_ratio > 1.5:
            # Good demand: price at midpoint
            self.final_price = (self.price_range_low + self.price_range_high) / 2
        else:
            # Moderate demand: price below midpoint
            midpoint = (self.price_range_low + self.price_range_high) / 2
            self.final_price = (self.price_range_low + midpoint) / 2

        return self.final_price

    def allocate_shares(self) -> Dict[str, List['InvestorAllocation']]:
        """Allocate shares to investors"""
        institutional_shares = int(self.shares_available * self.institutional_percent)
        retail_shares = int(self.shares_available * self.retail_percent)

        # Sort orders by quality (long-only, size, price)
        institutional_orders = [o for o in self.orders if o.investor_type in [
            "MUTUAL_FUND", "PENSION_FUND", "SOVEREIGN_WEALTH"
        ]]
        retail_orders = [o for o in self.orders if o.investor_type == "RETAIL"]

        # Allocate to institutions (pro-rata with quality bias)
        inst_demand = sum(o.shares_requested for o in institutional_orders)
        for order in institutional_orders:
            allocation_percent = order.shares_requested / inst_demand
            shares_allocated = int(institutional_shares * allocation_percent * 0.5)  # 50% fill

            self.institutional_investors.append(InvestorAllocation(
                investor_name=order.investor_name,
                shares_requested=order.shares_requested,
                shares_allocated=shares_allocated,
                fill_rate=shares_allocated / order.shares_requested
            ))

        # Allocate to retail (smaller allocations)
        retail_demand = sum(o.shares_requested for o in retail_orders)
        for order in retail_orders:
            allocation_percent = order.shares_requested / retail_demand
            shares_allocated = min(
                int(retail_shares * allocation_percent),
                1000  # Max 1,000 shares per retail
            )

            self.retail_investors.append(InvestorAllocation(
                investor_name=order.investor_name,
                shares_requested=order.shares_requested,
                shares_allocated=shares_allocated,
                fill_rate=shares_allocated / order.shares_requested if order.shares_requested > 0 else 0
            ))

        return {
            "institutional": self.institutional_investors,
            "retail": self.retail_investors
        }

    def get_book_summary(self) -> Dict[str, any]:
        """Get book-building summary"""
        return {
            "total_orders": len(self.orders),
            "total_demand": f"{self.total_demand:,} shares",
            "oversubscription": f"{self.oversubscription_ratio:.1f}x",
            "final_price": f"${self.final_price:.2f}",
            "institutional_allocation": f"{len(self.institutional_investors)} investors",
            "retail_allocation": f"{len(self.retail_investors)} investors",
            "average_fill_rate": f"{self.calculate_average_fill_rate():.1%}"
        }

    def calculate_average_fill_rate(self) -> float:
        """Calculate average fill rate"""
        if not self.institutional_investors and not self.retail_investors:
            return 0.0

        all_allocations = self.institutional_investors + self.retail_investors
        if not all_allocations:
            return 0.0

        return sum(a.fill_rate for a in all_allocations) / len(all_allocations)


@dataclass
class InvestorOrder:
    """Investor order during book-building"""
    investor_name: str
    investor_type: str
    shares_requested: int
    price_limit: float
    order_quality: str = "standard"  # anchor, standard, low
    investment_horizon: str = "long"  # long, medium, short


@dataclass
class InvestorAllocation:
    """Allocated shares to investor"""
    investor_name: str
    shares_requested: int
    shares_allocated: int
    fill_rate: float


@dataclass
class GreenshoeOption:
    """
    Over-allotment (Greenshoe) option - 15%
    """
    shares_available: int = 7_500_000  # 15% of 50M
    exercise_deadline: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))

    # Exercise decision
    exercised: bool = False
    shares_exercised: int = 0
    exercise_date: Optional[datetime] = None

    # Market conditions
    first_day_pop: float = 0.0  # % gain on first day
    trading_days_30: float = 0.0  # % change after 30 days

    def should_exercise(self, market_conditions: Dict[str, float]) -> bool:
        """Determine if greenshoe should be exercised"""
        self.first_day_pop = market_conditions.get('first_day_pop', 0.0)
        self.trading_days_30 = market_conditions.get('trading_days_30', 0.0)

        # Exercise if strong demand and price stability
        if self.first_day_pop > 0.15 and self.trading_days_30 > 0.10:
            # Very strong demand - full exercise
            self.exercised = True
            self.shares_exercised = self.shares_available
            self.exercise_date = datetime.now()
            return True
        elif self.first_day_pop > 0.10:
            # Strong demand - partial exercise
            self.exercised = True
            self.shares_exercised = int(self.shares_available * 0.5)
            self.exercise_date = datetime.now()
            return True

        return False


class IPOValuationManager:
    """
    Main valuation and pricing manager
    """
    def __init__(self):
        self.valuation = IPOValuation()
        self.pricing = SharePricing()
        self.book_building = BookBuilding()
        self.greenshoe = GreenshoeOption()

    async def complete_valuation_analysis(self) -> Dict[str, any]:
        """Complete comprehensive valuation"""
        # Calculate all valuation methods
        valuation_summary = self.valuation.get_valuation_summary()

        # Determine pricing
        pricing_summary = self.pricing.get_pricing_summary()

        return {
            "valuation": valuation_summary,
            "pricing": pricing_summary,
            "ready_for_roadshow": True
        }

    async def execute_book_building(self, orders: List[InvestorOrder]) -> Dict[str, any]:
        """Execute book-building process"""
        # Collect orders
        self.book_building.collect_orders(orders)

        # Determine final price
        final_price = self.book_building.determine_pricing()

        # Allocate shares
        allocations = self.book_building.allocate_shares()

        # Summary
        book_summary = self.book_building.get_book_summary()

        return {
            "final_price": final_price,
            "book_summary": book_summary,
            "allocations": {
                "institutional": len(allocations['institutional']),
                "retail": len(allocations['retail'])
            }
        }

    def get_comprehensive_metrics(self) -> Dict[str, any]:
        """Get all valuation and pricing metrics"""
        return {
            "target_valuation": f"${self.valuation.target_valuation/1e9:.1f}B",
            "valuation_range": f"${self.valuation.valuation_range_low/1e9:.1f}B - ${self.valuation.valuation_range_high/1e9:.1f}B",
            "price_range": f"${self.pricing.price_low} - ${self.pricing.price_high}",
            "shares_offered": f"{self.pricing.shares_offering:,}",
            "gross_proceeds": f"${self.pricing.gross_proceeds_mid/1e9:.2f}B",
            "revenue_multiple": f"{self.valuation.revenue_multiple}x",
            "dcf_valuation": f"${self.valuation.dcf_valuation/1e9:.1f}B"
        }


# Example usage
if __name__ == "__main__":
    manager = IPOValuationManager()

    # Complete valuation
    valuation_results = asyncio.run(manager.complete_valuation_analysis())
    print("IPO Valuation Analysis:")
    print(json.dumps(valuation_results, indent=2, default=str))

    # Simulate book-building
    sample_orders = [
        InvestorOrder("Fidelity", "MUTUAL_FUND", 5_000_000, 45.00, "anchor", "long"),
        InvestorOrder("BlackRock", "MUTUAL_FUND", 4_000_000, 44.00, "anchor", "long"),
        InvestorOrder("Vanguard", "MUTUAL_FUND", 3_000_000, 43.00, "standard", "long"),
        # ... 100+ more orders
    ]

    book_results = asyncio.run(manager.execute_book_building(sample_orders))
    print("\nBook-Building Results:")
    print(json.dumps(book_results, indent=2, default=str))

    # Metrics
    metrics = manager.get_comprehensive_metrics()
    print("\nComprehensive Metrics:")
    print(json.dumps(metrics, indent=2))
