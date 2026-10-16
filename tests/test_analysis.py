"""Regression checks for the published, holdout-only campaign analysis."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_marketing_analysis import PRE_OUTREACH_FEATURES, chronological_split


class CampaignAnalysisTests(unittest.TestCase):
    def test_chronological_split_keeps_order(self) -> None:
        frame = pd.DataFrame({"contact_id": range(1, 9)})
        train, holdout = chronological_split(frame)
        self.assertEqual(train["contact_id"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(holdout["contact_id"].tolist(), [7, 8])

    def test_pre_outreach_features_exclude_current_campaign_information(self) -> None:
        forbidden = {"duration", "campaign", "contact", "month", "day_of_week", "converted", "y"}
        self.assertFalse(forbidden.intersection(PRE_OUTREACH_FEATURES))

    def test_deciles_reconcile_with_holdout(self) -> None:
        metrics = pd.read_csv(ROOT / "data/processed/model_metrics.csv").iloc[0]
        deciles = pd.read_csv(ROOT / "data/processed/propensity_decile_holdout.csv")
        scores = pd.read_csv(ROOT / "data/processed/propensity_scores_sample.csv")
        self.assertEqual(len(scores), int(metrics["holdout_rows"]))
        self.assertEqual(int(deciles["contacts"].sum()), len(scores))
        self.assertEqual(int(deciles["conversions"].sum()), int(scores["converted"].sum()))
        self.assertEqual(scores["contact_id"].min(), int(metrics["train_rows"]) + 1)
        self.assertAlmostEqual(float(scores["converted"].mean()), float(metrics["holdout_base_rate"]))
        self.assertAlmostEqual(float(deciles.iloc[-1]["cumulative_lift"]), 1.0)

    def test_scenario_cost_uses_one_planned_contact_per_selected_record(self) -> None:
        deciles = pd.read_csv(ROOT / "data/processed/propensity_decile_holdout.csv")
        for row in deciles.itertuples():
            self.assertAlmostEqual(row.modeled_contact_cost, row.contacts * 3.0)
            self.assertAlmostEqual(row.modeled_net_value, row.conversions * 650.0 - row.contacts * 3.0)

    def test_dashboard_carries_non_deployment_warning(self) -> None:
        dashboard = (ROOT / "dashboard/marketing_dashboard.html").read_text(encoding="utf-8")
        self.assertIn("Not deployment-ready", dashboard)
        self.assertIn("not realized profit or ROI", dashboard)
        self.assertIn("chronological holdout", dashboard)


if __name__ == "__main__":
    unittest.main()
