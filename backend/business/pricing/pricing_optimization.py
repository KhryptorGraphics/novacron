"""
Package pricing provides value-based pricing and packaging optimization.

This module implements sophisticated pricing strategies to maximize $1B ARR
achievement while maintaining 42% margins and competitive positioning.
"""

import json
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class PricingModel(Enum):
    """Pricing model types"""
    FLAT_RATE = "flat_rate"
    PER_SEAT = "per_seat"
    USAGE_BASED = "usage_based"
    TIERED = "tiered"
    HYBRID = "hybrid"
    VALUE_BASED = "value_based"


class CustomerSegment(Enum):
    """Customer segment tiers"""
    SMB = "smb"
    MID_MARKET = "mid_market"
    ENTERPRISE = "enterprise"
    STRATEGIC = "strategic"
    FORTUNE_500 = "fortune_500"


@dataclass
class PricingTier:
    """Represents a pricing tier"""
    id: str
    name: str
    segment: CustomerSegment
    base_price: float
    min_seats: int
    max_seats: Optional[int]
    features: List[str]
    usage_limits: Dict[str, int]
    support_level: str
    sla_guarantee: float
    discount_eligible: bool
    popular: bool = False

    def calculate_price(self, seats: int, usage: Dict[str, int]) -> float:
        """Calculate total price for tier"""
        price = self.base_price * seats

        # Add usage-based charges
        for metric, value in usage.items():
            if metric in self.usage_limits:
                if value > self.usage_limits[metric]:
                    overage = value - self.usage_limits[metric]
                    price += overage * 0.01  # $0.01 per unit overage

        return price


@dataclass
class PricingPackage:
    """Represents a complete pricing package"""
    id: str
    name: str
    description: str
    tier: PricingTier
    products: List[str]
    annual_price: float
    monthly_price: float
    contract_length: int  # Months
    payment_terms: str
    auto_renewal: bool
    discount_schedule: Dict[int, float]  # Seats -> discount %
    incentives: List[str]
    created_at: datetime = field(default_factory=datetime.now)

    def calculate_total_contract_value(self, seats: int, months: int) -> float:
        """Calculate TCV"""
        base_price = self.annual_price if months >= 12 else self.monthly_price * months

        # Apply volume discount
        discount = 0.0
        for threshold, disc in sorted(self.discount_schedule.items()):
            if seats >= threshold:
                discount = disc

        return base_price * seats * (1 - discount)


@dataclass
class ValueMetric:
    """Metrics that drive pricing value"""
    name: str
    weight: float
    customer_value: float  # Customer's perceived value
    our_cost: float  # Our delivery cost
    competitive_value: float  # Competitor pricing

    def calculate_value_score(self) -> float:
        """Calculate value contribution"""
        return (self.customer_value - self.our_cost) * self.weight


@dataclass
class PricingProposal:
    """Custom pricing proposal for customer"""
    id: str
    customer_id: str
    package: PricingPackage
    seats: int
    usage_forecast: Dict[str, int]
    list_price: float
    discounts: List['Discount']
    final_price: float
    margin: float
    approval_required: bool
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    valid_until: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    notes: str = ""

    def calculate_margin(self, cost: float) -> float:
        """Calculate profit margin"""
        if self.final_price == 0:
            return 0.0
        return ((self.final_price - cost) / self.final_price) * 100


@dataclass
class Discount:
    """Represents a pricing discount"""
    type: str  # volume, promotional, competitive, strategic
    reason: str
    percentage: float
    amount: float
    approval_level: str  # sales, manager, vp, ceo
    approved: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CompetitivePricing:
    """Competitor pricing intelligence"""
    competitor: str
    product: str
    pricing_model: str
    starting_price: float
    average_price: float
    typical_discount: float
    strengths: List[str]
    weaknesses: List[str]
    market_position: str
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PriceSensitivity:
    """Customer price sensitivity analysis"""
    customer_id: str
    segment: CustomerSegment
    willingness_to_pay: float
    price_elasticity: float  # % demand change per % price change
    competitive_awareness: float  # 0-1
    value_perception: float  # 0-1
    urgency: float  # 0-1
    budget_authority: str
    decision_criteria: List[str]
    optimal_price_range: Tuple[float, float]


@dataclass
class PricingRule:
    """Automated pricing rule"""
    id: str
    name: str
    priority: int
    conditions: List[Dict[str, Any]]
    action: Dict[str, Any]
    enabled: bool = True
    execution_count: int = 0

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met"""
        for condition in self.conditions:
            field = condition['field']
            operator = condition['operator']
            value = condition['value']

            if field not in context:
                return False

            actual = context[field]

            if operator == '==' and actual != value:
                return False
            elif operator == '>' and actual <= value:
                return False
            elif operator == '<' and actual >= value:
                return False
            elif operator == '>=' and actual < value:
                return False
            elif operator == '<=' and actual > value:
                return False

        return True

    def apply(self, proposal: PricingProposal) -> PricingProposal:
        """Apply rule action to proposal"""
        action_type = self.action['type']

        if action_type == 'discount':
            discount = Discount(
                type=self.action.get('discount_type', 'automated'),
                reason=self.action.get('reason', self.name),
                percentage=self.action.get('percentage', 0),
                amount=self.action.get('amount', 0),
                approval_level=self.action.get('approval_level', 'automated')
            )
            proposal.discounts.append(discount)

        self.execution_count += 1
        return proposal


@dataclass
class PricingMetrics:
    """Pricing performance metrics"""
    total_proposals: int = 0
    approved_proposals: int = 0
    avg_deal_size: float = 0.0
    avg_margin: float = 42.0
    avg_discount: float = 0.0
    win_rate: float = 0.0
    price_realization: float = 100.0  # % of list price achieved
    competitive_win_rate: float = 0.0
    upsell_rate: float = 0.0
    renewal_rate: float = 97.0

    def calculate_win_rate(self, won: int, total: int) -> float:
        """Calculate win rate"""
        if total == 0:
            return 0.0
        return (won / total) * 100


class PricingOptimizer:
    """Main pricing optimization engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lock = threading.RLock()

        # Storage
        self.tiers: Dict[str, PricingTier] = {}
        self.packages: Dict[str, PricingPackage] = {}
        self.proposals: Dict[str, PricingProposal] = {}
        self.competitive_pricing: Dict[str, CompetitivePricing] = {}
        self.sensitivity_data: Dict[str, PriceSensitivity] = {}
        self.rules: List[PricingRule] = []

        # Metrics
        self.metrics = PricingMetrics()

        # Initialize
        self._initialize_tiers()
        self._initialize_packages()
        self._initialize_rules()
        self._initialize_competitive_data()

    def _initialize_tiers(self):
        """Initialize pricing tiers"""
        # Starter tier - SMB
        self.tiers['starter'] = PricingTier(
            id='starter',
            name='Starter',
            segment=CustomerSegment.SMB,
            base_price=50_000,
            min_seats=10,
            max_seats=100,
            features=['basic_features', 'email_support', 'standard_sla'],
            usage_limits={'api_calls': 1_000_000, 'storage_gb': 100},
            support_level='email',
            sla_guarantee=99.0,
            discount_eligible=True,
            popular=False
        )

        # Professional tier - Mid-market
        self.tiers['professional'] = PricingTier(
            id='professional',
            name='Professional',
            segment=CustomerSegment.MID_MARKET,
            base_price=250_000,
            min_seats=100,
            max_seats=1000,
            features=['advanced_features', 'priority_support', 'enhanced_sla', 'api_access'],
            usage_limits={'api_calls': 10_000_000, 'storage_gb': 1000},
            support_level='priority',
            sla_guarantee=99.5,
            discount_eligible=True,
            popular=True
        )

        # Enterprise tier - Enterprise
        self.tiers['enterprise'] = PricingTier(
            id='enterprise',
            name='Enterprise',
            segment=CustomerSegment.ENTERPRISE,
            base_price=1_000_000,
            min_seats=1000,
            max_seats=10000,
            features=['all_features', 'dedicated_support', 'premium_sla', 'custom_integrations'],
            usage_limits={'api_calls': 100_000_000, 'storage_gb': 10000},
            support_level='dedicated',
            sla_guarantee=99.9,
            discount_eligible=True,
            popular=True
        )

        # Strategic tier - Fortune 500
        self.tiers['strategic'] = PricingTier(
            id='strategic',
            name='Strategic',
            segment=CustomerSegment.FORTUNE_500,
            base_price=5_000_000,
            min_seats=10000,
            max_seats=None,
            features=['unlimited_features', 'executive_support', 'guaranteed_sla',
                     'custom_development', 'dedicated_tam'],
            usage_limits={},  # Unlimited
            support_level='executive',
            sla_guarantee=99.99,
            discount_eligible=True,
            popular=False
        )

    def _initialize_packages(self):
        """Initialize pricing packages"""
        for tier_id, tier in self.tiers.items():
            package = PricingPackage(
                id=f'package-{tier_id}',
                name=f'{tier.name} Package',
                description=f'{tier.name} tier package for {tier.segment.value}',
                tier=tier,
                products=['core_platform'],
                annual_price=tier.base_price,
                monthly_price=tier.base_price / 12 * 1.15,  # 15% premium for monthly
                contract_length=12,
                payment_terms='Net 30',
                auto_renewal=True,
                discount_schedule={
                    100: 0.05,    # 5% for 100+ seats
                    500: 0.10,    # 10% for 500+ seats
                    1000: 0.15,   # 15% for 1000+ seats
                    5000: 0.20,   # 20% for 5000+ seats
                },
                incentives=['annual_prepay_discount', 'multi_year_discount']
            )
            self.packages[package.id] = package

    def _initialize_rules(self):
        """Initialize pricing rules"""
        # Fortune 500 discount rule
        self.rules.append(PricingRule(
            id='fortune-500-strategic',
            name='Fortune 500 Strategic Discount',
            priority=1,
            conditions=[
                {'field': 'is_fortune_500', 'operator': '==', 'value': True},
                {'field': 'deal_size', 'operator': '>', 'value': 5_000_000}
            ],
            action={
                'type': 'discount',
                'discount_type': 'strategic',
                'percentage': 10,
                'reason': 'Fortune 500 strategic account',
                'approval_level': 'vp'
            }
        ))

        # Competitive displacement rule
        self.rules.append(PricingRule(
            id='competitive-displacement',
            name='Competitive Displacement Discount',
            priority=2,
            conditions=[
                {'field': 'has_competitor', 'operator': '==', 'value': True},
                {'field': 'competitive_threat_level', 'operator': '>', 'value': 0.7}
            ],
            action={
                'type': 'discount',
                'discount_type': 'competitive',
                'percentage': 15,
                'reason': 'Competitive displacement incentive',
                'approval_level': 'manager'
            }
        ))

        # Volume discount rule
        self.rules.append(PricingRule(
            id='volume-discount',
            name='High Volume Discount',
            priority=3,
            conditions=[
                {'field': 'seats', 'operator': '>', 'value': 10000}
            ],
            action={
                'type': 'discount',
                'discount_type': 'volume',
                'percentage': 20,
                'reason': 'High volume purchase',
                'approval_level': 'vp'
            }
        ))

        # Multi-year discount rule
        self.rules.append(PricingRule(
            id='multi-year-commitment',
            name='Multi-Year Commitment Discount',
            priority=4,
            conditions=[
                {'field': 'contract_years', 'operator': '>=', 'value': 3}
            ],
            action={
                'type': 'discount',
                'discount_type': 'promotional',
                'percentage': 12,
                'reason': '3+ year commitment',
                'approval_level': 'manager'
            }
        ))

    def _initialize_competitive_data(self):
        """Initialize competitive pricing data"""
        self.competitive_pricing['competitor-a'] = CompetitivePricing(
            competitor='Competitor A',
            product='Enterprise Platform',
            pricing_model='per_seat',
            starting_price=45_000,
            average_price=800_000,
            typical_discount=18.0,
            strengths=['Market leader', 'Brand recognition'],
            weaknesses=['Legacy technology', 'Poor support'],
            market_position='leader'
        )

        self.competitive_pricing['competitor-b'] = CompetitivePricing(
            competitor='Competitor B',
            product='Cloud Solution',
            pricing_model='usage_based',
            starting_price=25_000,
            average_price=500_000,
            typical_discount=12.0,
            strengths=['Modern architecture', 'Fast deployment'],
            weaknesses=['Limited features', 'Scaling issues'],
            market_position='challenger'
        )

    def create_proposal(self, customer_id: str, tier_id: str, seats: int,
                       context: Dict[str, Any]) -> PricingProposal:
        """Create pricing proposal for customer"""
        with self.lock:
            # Get package
            package_id = f'package-{tier_id}'
            if package_id not in self.packages:
                raise ValueError(f"Package not found: {package_id}")

            package = self.packages[package_id]

            # Calculate base price
            list_price = package.calculate_total_contract_value(
                seats=seats,
                months=package.contract_length
            )

            # Create proposal
            proposal = PricingProposal(
                id=f'proposal-{customer_id}-{datetime.now().timestamp()}',
                customer_id=customer_id,
                package=package,
                seats=seats,
                usage_forecast=context.get('usage_forecast', {}),
                list_price=list_price,
                discounts=[],
                final_price=list_price,
                margin=42.0,
                approval_required=False
            )

            # Apply pricing rules
            for rule in sorted(self.rules, key=lambda r: r.priority):
                if rule.enabled and rule.evaluate(context):
                    proposal = rule.apply(proposal)

            # Calculate final price with discounts
            total_discount = sum(d.percentage for d in proposal.discounts)
            proposal.final_price = list_price * (1 - total_discount / 100)

            # Calculate margin
            cost = list_price * 0.58  # 42% target margin
            proposal.margin = proposal.calculate_margin(cost)

            # Check if approval required (margin < 40% or discount > 20%)
            if proposal.margin < 40 or total_discount > 20:
                proposal.approval_required = True

            # Store proposal
            self.proposals[proposal.id] = proposal

            # Update metrics
            self.metrics.total_proposals += 1

            return proposal

    def optimize_pricing(self, customer_id: str, sensitivity: PriceSensitivity,
                        competitive_context: Dict[str, Any]) -> PricingProposal:
        """Optimize pricing based on value and competition"""
        with self.lock:
            # Determine optimal tier
            tier_id = self._select_optimal_tier(sensitivity)

            # Base proposal
            context = {
                'is_fortune_500': sensitivity.segment == CustomerSegment.FORTUNE_500,
                'deal_size': sensitivity.willingness_to_pay,
                'has_competitor': len(competitive_context.get('competitors', [])) > 0,
                'competitive_threat_level': competitive_context.get('threat_level', 0.5),
                'seats': competitive_context.get('seats', 1000),
                'contract_years': competitive_context.get('years', 1)
            }

            proposal = self.create_proposal(
                customer_id=customer_id,
                tier_id=tier_id,
                seats=context['seats'],
                context=context
            )

            # Value-based optimization
            if sensitivity.value_perception > 0.8:
                # High value perception - can price higher
                proposal.final_price *= 1.1
            elif sensitivity.price_elasticity > 1.5:
                # Highly price sensitive - need competitive pricing
                proposal.final_price *= 0.95

            # Competitive positioning
            if 'primary_competitor' in competitive_context:
                competitor = competitive_context['primary_competitor']
                if competitor in self.competitive_pricing:
                    comp_data = self.competitive_pricing[competitor]

                    # Price relative to competitor
                    if comp_data.market_position == 'leader':
                        # We can price 5% below leader
                        target_price = comp_data.average_price * 0.95
                        if proposal.final_price > target_price:
                            discount = Discount(
                                type='competitive',
                                reason=f'Competitive with {competitor}',
                                percentage=(proposal.final_price - target_price) / proposal.final_price * 100,
                                amount=proposal.final_price - target_price,
                                approval_level='manager'
                            )
                            proposal.discounts.append(discount)
                            proposal.final_price = target_price

            return proposal

    def _select_optimal_tier(self, sensitivity: PriceSensitivity) -> str:
        """Select optimal pricing tier for customer"""
        # Match segment to tier
        segment_tier_map = {
            CustomerSegment.SMB: 'starter',
            CustomerSegment.MID_MARKET: 'professional',
            CustomerSegment.ENTERPRISE: 'enterprise',
            CustomerSegment.FORTUNE_500: 'strategic',
            CustomerSegment.STRATEGIC: 'strategic'
        }

        return segment_tier_map.get(sensitivity.segment, 'professional')

    def analyze_price_sensitivity(self, customer_id: str,
                                  historical_data: Dict[str, Any]) -> PriceSensitivity:
        """Analyze customer price sensitivity"""
        # Simplified analysis - in production would use ML
        segment = CustomerSegment(historical_data.get('segment', 'enterprise'))

        # Estimate willingness to pay based on segment
        wtp_map = {
            CustomerSegment.SMB: 100_000,
            CustomerSegment.MID_MARKET: 500_000,
            CustomerSegment.ENTERPRISE: 2_000_000,
            CustomerSegment.FORTUNE_500: 5_000_000,
            CustomerSegment.STRATEGIC: 10_000_000
        }

        return PriceSensitivity(
            customer_id=customer_id,
            segment=segment,
            willingness_to_pay=wtp_map[segment],
            price_elasticity=historical_data.get('elasticity', 1.2),
            competitive_awareness=historical_data.get('competitive_awareness', 0.7),
            value_perception=historical_data.get('value_perception', 0.8),
            urgency=historical_data.get('urgency', 0.6),
            budget_authority=historical_data.get('budget_authority', 'manager'),
            decision_criteria=historical_data.get('criteria', ['price', 'features', 'support']),
            optimal_price_range=(wtp_map[segment] * 0.8, wtp_map[segment] * 1.2)
        )

    def calculate_value_based_price(self, customer_id: str,
                                   value_metrics: List[ValueMetric]) -> float:
        """Calculate price based on customer value"""
        total_value = sum(metric.calculate_value_score() for metric in value_metrics)

        # Price at 30% of total value delivered
        base_price = total_value * 0.30

        # Ensure minimum margin
        min_cost = sum(metric.our_cost for metric in value_metrics)
        min_price = min_cost / 0.58  # 42% margin

        return max(base_price, min_price)

    def get_competitive_positioning(self, proposal: PricingProposal,
                                   competitor_ids: List[str]) -> Dict[str, Any]:
        """Compare proposal to competitive pricing"""
        positioning = {
            'our_price': proposal.final_price,
            'competitors': {},
            'price_rank': 0,
            'value_rank': 0,
            'recommendation': ''
        }

        for comp_id in competitor_ids:
            if comp_id in self.competitive_pricing:
                comp = self.competitive_pricing[comp_id]
                positioning['competitors'][comp.competitor] = {
                    'price': comp.average_price,
                    'delta': proposal.final_price - comp.average_price,
                    'delta_pct': ((proposal.final_price - comp.average_price) / comp.average_price) * 100,
                    'position': comp.market_position
                }

        return positioning

    def export_metrics(self) -> Dict[str, Any]:
        """Export pricing metrics"""
        return {
            'total_proposals': self.metrics.total_proposals,
            'approved_proposals': self.metrics.approved_proposals,
            'avg_deal_size': self.metrics.avg_deal_size,
            'avg_margin': self.metrics.avg_margin,
            'avg_discount': self.metrics.avg_discount,
            'win_rate': self.metrics.win_rate,
            'price_realization': self.metrics.price_realization,
            'renewal_rate': self.metrics.renewal_rate
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        with self.lock:
            return {
                'tiers': len(self.tiers),
                'packages': len(self.packages),
                'proposals': len(self.proposals),
                'rules': len(self.rules),
                'metrics': self.export_metrics()
            }


# Example usage
if __name__ == '__main__':
    # Initialize optimizer
    config = {
        'target_margin': 42.0,
        'min_margin': 35.0,
        'max_discount': 25.0
    }

    optimizer = PricingOptimizer(config)

    # Create proposal
    proposal = optimizer.create_proposal(
        customer_id='acme-corp',
        tier_id='enterprise',
        seats=5000,
        context={
            'is_fortune_500': True,
            'deal_size': 8_000_000,
            'has_competitor': True,
            'competitive_threat_level': 0.8,
            'seats': 5000,
            'contract_years': 3
        }
    )

    print(f"Proposal: {proposal.id}")
    print(f"List Price: ${proposal.list_price:,.2f}")
    print(f"Final Price: ${proposal.final_price:,.2f}")
    print(f"Margin: {proposal.margin:.1f}%")
    print(f"Discounts: {len(proposal.discounts)}")
    for discount in proposal.discounts:
        print(f"  - {discount.type}: {discount.percentage}% ({discount.reason})")
    print(f"Approval Required: {proposal.approval_required}")
