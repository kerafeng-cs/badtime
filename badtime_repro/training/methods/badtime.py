"""BadTime (v2): kế hoạch từ bốn khối thiết kế, GAT chấm điểm kênh, objective Công thức 15-22."""

from __future__ import annotations

from ...attack.gat import GatInfluenceScorer
from ...attack.objectives import BackdoorModelObjective, TriggerObjective
from ...attack.plan import AttackPlan, AttackPlanner
from .poisoning import PoisoningMethod


class BadTimeMethod(PoisoningMethod):
    name = "badtime"

    def build_plan(self) -> AttackPlan:
        att = self.config.attack
        scorer = GatInfluenceScorer(att.gat, self.ctx.device)
        return AttackPlanner(att, scorer).plan(self.ctx.dataset, self.ctx.seeds.influence)

    def make_model_objective(self) -> BackdoorModelObjective:
        att = self.config.attack
        return BackdoorModelObjective(
            self.ctx.dataset.n_channels, self.plan.target_idx, att.model_loss_weights, att.loss_reduction
        )

    def make_trigger_objective(self) -> TriggerObjective:
        att = self.config.attack
        return TriggerObjective(
            self.ctx.dataset.n_channels, self.plan.target_idx, att.trigger_loss_weights, att.loss_reduction,
            att.smooth_eps,
        )
