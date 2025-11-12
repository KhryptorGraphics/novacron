"""
Comprehensive test suite for Enterprise Sales Intelligence Platform
Tests ML-powered deal scoring, competitive playbooks, and ABM automation
"""

import unittest
from datetime import datetime, timedelta

import sys
sys.path.append('../backend/sales')

from enterprise_intelligence import (
    EnterpriseSalesIntelligence,
    DealProfile,
    DealStage,
    CompetitorType,
    BuyingSignal,
    CompetitivePlaybook,
    ABMCampaign,
)


class TestEnterpriseSalesIntelligence(unittest.TestCase):
    """Test suite for Enterprise Sales Intelligence"""

    def setUp(self):
        """Set up test fixtures"""
        self.intelligence = EnterpriseSalesIntelligence()

    def test_initialization(self):
        """Test platform initialization"""
        self.assertIsNotNone(self.intelligence)
        self.assertEqual(len(self.intelligence.deals), 0)
        self.assertGreater(len(self.intelligence.playbooks), 0, "Should have competitive playbooks")

    def test_competitive_playbooks(self):
        """Test competitive displacement playbooks"""
        # Verify VMware playbook
        vmware_playbook = self.intelligence.get_playbook(CompetitorType.VMWARE)
        self.assertIsNotNone(vmware_playbook, "VMware playbook should exist")
        self.assertGreaterEqual(vmware_playbook.win_rate, 0.70, "70%+ win rate vs VMware")
        self.assertGreater(len(vmware_playbook.key_differentiators), 0, "Should have differentiators")
        self.assertGreater(len(vmware_playbook.discovery_questions), 0, "Should have discovery questions")

        # Verify AWS playbook
        aws_playbook = self.intelligence.get_playbook(CompetitorType.AWS)
        self.assertIsNotNone(aws_playbook, "AWS playbook should exist")
        self.assertGreaterEqual(aws_playbook.win_rate, 0.60, "60%+ win rate vs AWS")

        # Verify Kubernetes playbook
        k8s_playbook = self.intelligence.get_playbook(CompetitorType.KUBERNETES)
        self.assertIsNotNone(k8s_playbook, "K8s playbook should exist")
        self.assertGreaterEqual(k8s_playbook.win_rate, 0.80, "80%+ win rate vs K8s")

    def test_enterprise_deal_creation(self):
        """Test enterprise deal creation and tracking"""
        deal = DealProfile(
            deal_id="test-deal-001",
            account_name="Fortune 100 Company",
            deal_value=30_000_000,
            annual_value=10_000_000,
            stage=DealStage.PROPOSAL,
            probability=0.75,
            days_in_pipeline=60,
            fortune_500_rank=50,
            employee_count=100000,
            annual_revenue=50_000_000_000,
            industry_vertical="financial-services",
            incumbent_competitor=CompetitorType.VMWARE,
            competitor_contract_value=12_000_000,
            competitor_satisfaction=0.60,
            buying_signals=[
                BuyingSignal.BUDGET_ALLOCATED,
                BuyingSignal.RFP_ISSUED,
                BuyingSignal.EXECUTIVE_ENGAGEMENT,
            ],
            executive_engagement_count=3,
            technical_champions=2,
            economic_buyer_identified=True,
            use_cases=["core infrastructure", "disaster recovery", "hybrid cloud"],
            technical_fit_score=0.90,
            compliance_requirements_met=0.95,
            meetings_held=8,
            demos_completed=2,
            poc_status="successful",
            reference_potential=0.95,
            expansion_potential=20_000_000,
            strategic_account=True,
        )

        self.intelligence.add_deal(deal)
        self.assertEqual(len(self.intelligence.deals), 1, "Should have one deal")
        self.assertIn(deal.deal_id, self.intelligence.deals, "Deal should be tracked")

    def test_deal_scoring(self):
        """Test ML-powered deal scoring with 99%+ accuracy"""
        # Create a strong deal
        deal = DealProfile(
            deal_id="score-test-001",
            account_name="Top Bank",
            deal_value=25_000_000,
            annual_value=8_000_000,
            stage=DealStage.NEGOTIATION,
            probability=0.80,
            days_in_pipeline=90,
            fortune_500_rank=25,
            employee_count=150000,
            annual_revenue=100_000_000_000,
            industry_vertical="financial-services",
            incumbent_competitor=CompetitorType.VMWARE,
            competitor_satisfaction=0.50,
            buying_signals=[
                BuyingSignal.BUDGET_ALLOCATED,
                BuyingSignal.POC_REQUESTED,
                BuyingSignal.EXECUTIVE_ENGAGEMENT,
                BuyingSignal.COMPETITOR_DISSATISFACTION,
            ],
            executive_engagement_count=4,
            technical_champions=3,
            economic_buyer_identified=True,
            use_cases=["core banking", "trading", "analytics"],
            technical_fit_score=0.95,
            compliance_requirements_met=1.0,
            meetings_held=12,
            demos_completed=3,
            poc_status="successful",
            reference_potential=1.0,
            strategic_account=True,
        )

        self.intelligence.add_deal(deal)
        score = self.intelligence.score_deal(deal.deal_id)

        self.assertIsNotNone(score, "Should generate deal score")
        self.assertGreaterEqual(score.overall_score, 70.0, "Strong deal should score 70+")
        self.assertGreaterEqual(score.win_probability, 0.70, "Win probability should be 70%+")
        self.assertGreaterEqual(score.confidence, 0.99, "99%+ scoring confidence")

        # Verify score components
        self.assertGreater(score.account_fit_score, 0, "Should have account fit score")
        self.assertGreater(score.competitive_position_score, 0, "Should have competitive score")
        self.assertGreater(score.buying_intent_score, 0, "Should have buying intent score")
        self.assertGreater(score.solution_fit_score, 0, "Should have solution fit score")
        self.assertGreater(score.engagement_score, 0, "Should have engagement score")

        # Verify recommendations
        self.assertGreater(len(score.recommended_actions), 0, "Should have recommendations")

    def test_account_fit_scoring(self):
        """Test account fit scoring component"""
        # Fortune 100 deal
        f100_deal = DealProfile(
            deal_id="fit-001",
            account_name="Fortune 100",
            deal_value=50_000_000,
            annual_value=15_000_000,
            stage=DealStage.QUALIFICATION,
            probability=0.60,
            days_in_pipeline=30,
            fortune_500_rank=50,
            employee_count=200000,
            annual_revenue=150_000_000_000,
        )

        self.intelligence.add_deal(f100_deal)
        score = self.intelligence.score_deal(f100_deal.deal_id)

        self.assertGreaterEqual(score.account_fit_score, 80.0, "Fortune 100 should score 80+ on account fit")

    def test_competitive_position_scoring(self):
        """Test competitive position scoring"""
        # Competitor dissatisfaction deal
        competitive_deal = DealProfile(
            deal_id="comp-001",
            account_name="Dissatisfied VMware Customer",
            deal_value=10_000_000,
            annual_value=3_500_000,
            stage=DealStage.SOLUTION_DESIGN,
            probability=0.65,
            days_in_pipeline=45,
            incumbent_competitor=CompetitorType.VMWARE,
            competitor_satisfaction=0.30,
            buying_signals=[
                BuyingSignal.COMPETITOR_DISSATISFACTION,
                BuyingSignal.CONTRACT_EXPIRING,
            ],
        )

        self.intelligence.add_deal(competitive_deal)
        score = self.intelligence.score_deal(competitive_deal.deal_id)

        self.assertGreaterEqual(score.competitive_position_score, 70.0, "Competitive advantage should score 70+")

    def test_buying_intent_scoring(self):
        """Test buying intent scoring"""
        # Strong buying signals deal
        intent_deal = DealProfile(
            deal_id="intent-001",
            account_name="Strong Intent",
            deal_value=12_000_000,
            annual_value=4_000_000,
            stage=DealStage.PROPOSAL,
            probability=0.70,
            days_in_pipeline=60,
            buying_signals=[
                BuyingSignal.BUDGET_ALLOCATED,
                BuyingSignal.RFP_ISSUED,
                BuyingSignal.POC_REQUESTED,
                BuyingSignal.EXECUTIVE_ENGAGEMENT,
            ],
            executive_engagement_count=5,
            technical_champions=3,
            economic_buyer_identified=True,
        )

        self.intelligence.add_deal(intent_deal)
        score = self.intelligence.score_deal(intent_deal.deal_id)

        self.assertGreaterEqual(score.buying_intent_score, 75.0, "Strong buying intent should score 75+")

    def test_solution_fit_scoring(self):
        """Test solution fit scoring"""
        # High solution fit deal
        fit_deal = DealProfile(
            deal_id="solution-001",
            account_name="Perfect Fit",
            deal_value=8_000_000,
            annual_value=2_500_000,
            stage=DealStage.SOLUTION_DESIGN,
            probability=0.70,
            days_in_pipeline=50,
            technical_fit_score=0.95,
            compliance_requirements_met=1.0,
            use_cases=["core infra", "DR", "containers"],
            poc_status="successful",
        )

        self.intelligence.add_deal(fit_deal)
        score = self.intelligence.score_deal(fit_deal.deal_id)

        self.assertGreaterEqual(score.solution_fit_score, 80.0, "High solution fit should score 80+")

    def test_risk_identification(self):
        """Test deal risk factor identification"""
        # At-risk deal
        risk_deal = DealProfile(
            deal_id="risk-001",
            account_name="At Risk Deal",
            deal_value=5_000_000,
            annual_value=1_500_000,
            stage=DealStage.QUALIFICATION,
            probability=0.40,
            days_in_pipeline=200,
            technical_champions=0,
            economic_buyer_identified=False,
            demos_completed=0,
            poc_status="not_started",
        )

        self.intelligence.add_deal(risk_deal)
        score = self.intelligence.score_deal(risk_deal.deal_id)

        self.assertGreater(len(score.risk_factors), 0, "At-risk deal should have risk factors")
        self.assertLess(score.win_probability, 0.50, "At-risk deal should have <50% win probability")

    def test_deal_recommendations(self):
        """Test AI-powered deal recommendations"""
        deal = DealProfile(
            deal_id="rec-001",
            account_name="Needs Guidance",
            deal_value=10_000_000,
            annual_value=3_000_000,
            stage=DealStage.SOLUTION_DESIGN,
            probability=0.60,
            days_in_pipeline=75,
            incumbent_competitor=CompetitorType.VMWARE,
            executive_engagement_count=0,
            technical_champions=1,
            economic_buyer_identified=False,
            demos_completed=0,
            poc_status="not_started",
        )

        self.intelligence.add_deal(deal)
        score = self.intelligence.score_deal(deal.deal_id)

        self.assertGreater(len(score.recommended_actions), 0, "Should have recommendations")

        # Check for specific recommendations
        recommendations_text = " ".join(score.recommended_actions).lower()
        self.assertIn("playbook", recommendations_text, "Should recommend playbook")

    def test_top_deals_ranking(self):
        """Test top deals ranking by score"""
        # Add multiple deals
        for i in range(5):
            deal = DealProfile(
                deal_id=f"rank-{i:03d}",
                account_name=f"Deal {i}",
                deal_value=(i + 1) * 5_000_000,
                annual_value=(i + 1) * 1_500_000,
                stage=DealStage.PROPOSAL,
                probability=0.60 + (i * 0.05),
                days_in_pipeline=60,
            )
            self.intelligence.add_deal(deal)
            self.intelligence.score_deal(deal.deal_id)

        top_deals = self.intelligence.get_top_deals(3)
        self.assertEqual(len(top_deals), 3, "Should return top 3 deals")

        # Verify ranking by score
        for i in range(len(top_deals) - 1):
            self.assertGreaterEqual(
                top_deals[i][1].overall_score,
                top_deals[i + 1][1].overall_score,
                "Deals should be ranked by score"
            )

    def test_at_risk_deals(self):
        """Test at-risk deals identification"""
        # Add high-risk deal
        risk_deal = DealProfile(
            deal_id="high-risk-001",
            account_name="High Risk",
            deal_value=8_000_000,
            annual_value=2_500_000,
            stage=DealStage.QUALIFICATION,
            probability=0.30,
            days_in_pipeline=250,
            technical_champions=0,
            economic_buyer_identified=False,
        )

        self.intelligence.add_deal(risk_deal)
        self.intelligence.score_deal(risk_deal.deal_id)

        at_risk = self.intelligence.get_deals_at_risk()
        self.assertGreater(len(at_risk), 0, "Should identify at-risk deals")
        self.assertLess(at_risk[0][1].win_probability, 0.50, "At-risk deals should have <50% probability")

    def test_pipeline_metrics(self):
        """Test overall pipeline metrics calculation"""
        # Add sample deals
        for i in range(10):
            deal = DealProfile(
                deal_id=f"metric-{i:03d}",
                account_name=f"Company {i}",
                deal_value=5_000_000 + (i * 1_000_000),
                annual_value=1_500_000 + (i * 300_000),
                stage=DealStage.PROPOSAL,
                probability=0.50 + (i * 0.03),
                days_in_pipeline=60,
            )
            self.intelligence.add_deal(deal)
            self.intelligence.score_deal(deal.deal_id)

        metrics = self.intelligence.calculate_pipeline_metrics()

        self.assertEqual(metrics["total_deals"], 10, "Should have 10 deals")
        self.assertGreater(metrics["total_pipeline_value"], 0, "Should have pipeline value")
        self.assertGreater(metrics["weighted_pipeline_value"], 0, "Should have weighted value")
        self.assertGreater(metrics["average_deal_score"], 0, "Should have average score")
        self.assertIn("win_rate_by_competitor", metrics, "Should have win rates")

    def test_abm_campaign(self):
        """Test account-based marketing campaign creation"""
        campaign = ABMCampaign(
            campaign_id="abm-001",
            campaign_name="Fortune 500 Banking Campaign",
            target_accounts=["Bank A", "Bank B", "Bank C"],
            vertical_focus="financial-services",
            persona_targets=["CTO", "CIO", "CISO"],
            content_assets=["White paper", "Case study", "ROI calculator"],
            engagement_channels=["Email", "LinkedIn", "Executive events"],
        )

        self.intelligence.create_abm_campaign(campaign)
        self.assertIn(campaign.campaign_id, self.intelligence.abm_campaigns, "Campaign should be tracked")

    def test_export_metrics(self):
        """Test metrics export functionality"""
        # Add sample deal
        deal = DealProfile(
            deal_id="export-001",
            account_name="Export Test",
            deal_value=10_000_000,
            annual_value=3_000_000,
            stage=DealStage.PROPOSAL,
            probability=0.70,
            days_in_pipeline=60,
        )

        self.intelligence.add_deal(deal)
        self.intelligence.score_deal(deal.deal_id)

        metrics_json = self.intelligence.export_metrics()
        self.assertIsNotNone(metrics_json, "Should export metrics")
        self.assertIn("pipeline_metrics", metrics_json, "Should include pipeline metrics")
        self.assertIn("top_deals", metrics_json, "Should include top deals")

    def test_vmware_displacement_scenario(self):
        """Test complete VMware displacement scenario"""
        deal = DealProfile(
            deal_id="vmware-displace-001",
            account_name="VMware Customer",
            deal_value=20_000_000,
            annual_value=6_000_000,
            stage=DealStage.NEGOTIATION,
            probability=0.75,
            days_in_pipeline=100,
            fortune_500_rank=150,
            incumbent_competitor=CompetitorType.VMWARE,
            competitor_satisfaction=0.40,
            buying_signals=[
                BuyingSignal.COMPETITOR_DISSATISFACTION,
                BuyingSignal.CONTRACT_EXPIRING,
                BuyingSignal.BUDGET_ALLOCATED,
            ],
            technical_champions=2,
            poc_status="successful",
        )

        self.intelligence.add_deal(deal)
        score = self.intelligence.score_deal(deal.deal_id)

        # Should recommend VMware playbook
        recommendations_text = " ".join(score.recommended_actions)
        self.assertIn("vmware", recommendations_text.lower(), "Should recommend VMware playbook")

        # High win probability due to dissatisfaction and playbook
        self.assertGreaterEqual(score.win_probability, 0.60, "VMware displacement should have 60%+ win probability")


class TestPerformance(unittest.TestCase):
    """Performance tests for sales intelligence"""

    def test_scoring_performance(self):
        """Test deal scoring performance"""
        intelligence = EnterpriseSalesIntelligence()

        # Add 100 deals
        for i in range(100):
            deal = DealProfile(
                deal_id=f"perf-{i:03d}",
                account_name=f"Company {i}",
                deal_value=5_000_000,
                annual_value=1_500_000,
                stage=DealStage.PROPOSAL,
                probability=0.60,
                days_in_pipeline=60,
            )
            intelligence.add_deal(deal)

        # Score all deals
        import time
        start_time = time.time()

        for deal_id in intelligence.deals.keys():
            intelligence.score_deal(deal_id)

        elapsed_time = time.time() - start_time

        self.assertLess(elapsed_time, 1.0, "Should score 100 deals in <1 second")


if __name__ == '__main__':
    unittest.main()
