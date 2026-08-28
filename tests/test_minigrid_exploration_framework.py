import math
from pathlib import Path

from utils.minigrid_coverage_rewards import CoverageMilestoneTracker, CoverageRewardConfig
from utils.minigrid_deadlocks import DeadlockConfig, DeadlockTracker
from utils.minigrid_experiment_config import (
    ContinuationConfig,
    MiniGridExplorationExperimentConfig,
    ValidationConfig,
)
from utils.minigrid_exploration_episode import transition_is_done
from utils.minigrid_exploration_policy import episode_linear_epsilon
from utils.minigrid_exploration_rewards import ExplorationRewardConfig, compute_exploration_reward
from utils.minigrid_exploration_rewards import (
    compute_adaptive_frontier_resolution_reward,
)
from utils.minigrid_exploration_topology import (
    frontier_target_was_resolved,
    reachable_frontier_regions,
)
from utils.minigrid_exploration_trainer import MiniGridExplorationTrainer
from utils.minigrid_mapper import MapCell
from minigrid.core.constants import OBJECT_TO_IDX


def test_milestones_are_one_time_multi_crossing_and_episode_local():
    tracker = CoverageMilestoneTracker(
        CoverageRewardConfig(milestones={0.75: 5, 0.90: 10, 0.95: 20, 1.0: 50})
    )
    assert tracker.update(0.74, 0.76).bonus == 5
    assert tracker.update(0.76, 0.80).bonus == 0
    update = tracker.update(0.89, 1.0)
    assert update.crossed_milestones == (0.90, 0.95, 1.0)
    assert update.bonus == 80
    assert update.coverage_success
    assert tracker.update(1.0, 1.0).bonus == 0
    tracker.reset()
    assert tracker.update(0.74, 0.76).bonus == 5


def test_reward_decomposition_keeps_milestone_separate():
    reward = compute_exploration_reward(
        new_cells=2,
        frontier_distance_before=4,
        frontier_distance_after=5,
        action=2,
        position_before=(0, 0),
        position_after=(1, 0),
        previous_action=2,
        forward_was_known_blocked=False,
        coverage_milestone_bonus=10,
        config=ExplorationRewardConfig(novelty_beta=2),
    )
    assert reward["novelty"] == 4
    assert reward["coverage_milestone"] == 10
    assert reward["frontier"] == 0
    assert reward["total"] == 14


def test_oscillation_penalty_escalates_without_replacing_stationary_penalty():
    config = ExplorationRewardConfig(
        stationary_penalty=-0.5,
        oscillation_penalty=-2.0,
        escalate_oscillation_penalty=True,
        oscillation_penalty_max_multiplier=4,
    )
    oscillation_penalties = []
    for reversal_count in range(1, 6):
        reward = compute_exploration_reward(
            new_cells=0,
            frontier_distance_before=4,
            frontier_distance_after=4,
            action=1,
            position_before=(0, 0),
            position_after=(0, 0),
            previous_action=0,
            consecutive_reversals=reversal_count,
            forward_was_known_blocked=False,
            config=config,
        )
        oscillation_penalties.append(reward["oscillation"])
        assert reward["stationary"] == -0.5
        assert reward["total"] == reward["stationary"] + reward["oscillation"]
    assert oscillation_penalties == [-2.0, -4.0, -6.0, -8.0, -8.0]
    static_reward = compute_exploration_reward(
        new_cells=0,
        frontier_distance_before=4,
        frontier_distance_after=4,
        action=1,
        position_before=(0, 0),
        position_after=(0, 0),
        previous_action=0,
        consecutive_reversals=4,
        forward_was_known_blocked=False,
        config=ExplorationRewardConfig(oscillation_penalty=-2.0),
    )
    assert static_reward["oscillation"] == -2.0


def test_no_progress_resets_on_discovery_or_reduced_frontier_distance():
    tracker = DeadlockTracker(DeadlockConfig())
    tracker.reset(no_progress_timeout=4)
    common = dict(action=2, previous_action=2, position_changed=True)
    tracker.update(**common, new_cells=0, frontier_distance_before=12, frontier_distance_after=12)
    assert tracker.steps_since_exploration_progress == 1
    tracker.update(**common, new_cells=1, frontier_distance_before=12, frontier_distance_after=13)
    assert tracker.steps_since_exploration_progress == 0
    tracker.update(**common, new_cells=0, frontier_distance_before=12, frontier_distance_after=11)
    assert tracker.steps_since_exploration_progress == 0
    status = None
    for _ in range(4):
        status = tracker.update(
            **common, new_cells=0, frontier_distance_before=12, frontier_distance_after=12
        )
    assert status["deadlock_type"] == "no_progress_deadlock"


def test_episode_epsilon_and_coverage_terminal_flag():
    assert episode_linear_epsilon(0, 1001, 0.10, 0.02) == 0.10
    assert math.isclose(episode_linear_epsilon(500, 1001, 0.10, 0.02), 0.06)
    assert episode_linear_epsilon(1000, 1001, 0.10, 0.02) == 0.02
    assert transition_is_done(False, False, None, True)


def test_validation_levels_and_output_inference_are_configurable():
    validation = ValidationConfig(
        seeds=tuple(range(100)),
        checkpoint_seed_count=20,
        large_validation_frequency=500,
        large_validation_episode_count=100,
    )
    assert len(validation.checkpoint_seeds) == 20
    assert len(validation.large_validation_seeds) == 100
    config = MiniGridExplorationExperimentConfig(
        name="experiment_test", initial_checkpoint=Path("source.pth")
    )
    assert config.output.directory == Path("experiment_test")
    continuation = ContinuationConfig(
        enabled=True, checkpoint="BEST", additional_episodes=2_000, mode="fine_tune"
    )
    assert continuation.checkpoint == "best"
    assert continuation.additional_episodes == 2_000
    assert continuation.mode == "fine_tune"


def test_best_validation_prefers_large_then_coverage_tiebreaks():
    def record(validation_type, coverage, reached_100, reached_95, deadlocks):
        return {
            "validation_type": validation_type,
            "metrics": {
                "mean_final_coverage": coverage,
                "fraction_reaching_100": reached_100,
                "fraction_reaching_95": reached_95,
                "fraction_oscillation_deadlock": deadlocks,
                "fraction_stationary_deadlock": 0.0,
                "fraction_no_progress_deadlock": 0.0,
            },
        }

    lightweight = record("lightweight", 0.90, 0.10, 0.20, 0.10)
    large = record("large", 0.80, 0.05, 0.10, 0.20)
    assert MiniGridExplorationTrainer._is_better_validation(large, lightweight, 1e-3)
    better_completion = record("large", 0.8005, 0.06, 0.10, 0.20)
    assert MiniGridExplorationTrainer._is_better_validation(
        better_completion, large, 1e-3
    )
    worse_deadlocks = record("large", 0.8005, 0.05, 0.10, 0.30)
    assert not MiniGridExplorationTrainer._is_better_validation(
        worse_deadlocks, large, 1e-3
    )


def test_adaptive_frontier_reward_uses_log_growth_inverse_count_and_cap():
    config = ExplorationRewardConfig(
        adaptive_frontier_resolution_enabled=True,
        frontier_base_reward=0.5,
        frontier_map_scale=2.0,
        frontier_known_cell_scale=25.0,
        frontier_count_exponent=0.75,
        frontier_reward_cap=5.0,
    )
    small_map = compute_adaptive_frontier_resolution_reward(
        known_cells_before=25,
        reachable_frontier_count_before=4,
        resolved_frontier=True,
        config=config,
    )
    large_map = compute_adaptive_frontier_resolution_reward(
        known_cells_before=250,
        reachable_frontier_count_before=4,
        resolved_frontier=True,
        config=config,
    )
    few_frontiers = compute_adaptive_frontier_resolution_reward(
        known_cells_before=250,
        reachable_frontier_count_before=1,
        resolved_frontier=True,
        config=config,
    )
    capped = compute_adaptive_frontier_resolution_reward(
        known_cells_before=1_000_000,
        reachable_frontier_count_before=1,
        resolved_frontier=True,
        config=config,
    )
    assert small_map < large_map < few_frontiers
    assert capped == config.frontier_reward_cap
    assert compute_adaptive_frontier_resolution_reward(
        known_cells_before=250,
        reachable_frontier_count_before=0,
        resolved_frontier=True,
        config=config,
    ) == 0.0
    assert compute_adaptive_frontier_resolution_reward(
        known_cells_before=250,
        reachable_frontier_count_before=1,
        resolved_frontier=False,
        config=config,
    ) == 0.0


def test_reachable_frontier_regions_and_one_time_resolution_detection():
    class FakeMapper:
        position = (0, 0)

        def __init__(self):
            empty = MapCell(OBJECT_TO_IDX["empty"], 0, 0)
            self.cells = {
                (0, 0): empty,
                (1, 0): empty,
                (2, 0): empty,
                (0, 1): empty,
                (0, 2): empty,
            }
            self.frontiers = {(1, 0), (2, 0), (0, 2)}

        def frontier_cells(self):
            return set(self.frontiers)

    regions = reachable_frontier_regions(FakeMapper())
    assert len(regions) == 2
    assert not frontier_target_was_resolved(
        regions, {(1, 0), (0, 2)}, new_cells=2
    )
    assert frontier_target_was_resolved(regions, {(0, 2)}, new_cells=2)
    assert not frontier_target_was_resolved(regions, {(0, 2)}, new_cells=0)


def test_frontier_resolution_is_a_separate_reward_component():
    config = ExplorationRewardConfig(
        adaptive_frontier_resolution_enabled=True,
        novelty_beta=2.0,
    )
    reward = compute_exploration_reward(
        new_cells=3,
        frontier_distance_before=2,
        frontier_distance_after=3,
        action=2,
        position_before=(0, 0),
        position_after=(1, 0),
        previous_action=2,
        forward_was_known_blocked=False,
        known_cells_before=100,
        reachable_frontier_count_before=2,
        resolved_frontier=True,
        config=config,
    )
    assert reward["novelty"] == 6.0
    assert reward["frontier"] == 0.0
    assert reward["frontier_resolution"] > 0.0
    assert reward["frontier_resolved"]
    assert reward["total"] == reward["novelty"] + reward["frontier_resolution"]
