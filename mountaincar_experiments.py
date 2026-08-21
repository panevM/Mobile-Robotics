"""Systematic tabular Q-learning experiments for MountainCar-v0.

This module keeps the discretization and Q-learning behavior used in the
notebooks, while making resolution, exploration, evaluation, and visitation
analysis repeatable.  Run this file directly, or import it from
``mountaincar.ipynb`` and call ``run_all_experiments()``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Experiment configuration -- edit these values to scale the investigation.
# ---------------------------------------------------------------------------
RUN_EPSILON_COMPARISON = True
RUN_VISITATION_ANALYSIS = True
RUN_100X100_BUDGET_TEST = False  # Up to 250k episodes, so disabled by default.
SAVE_RESULTS = True
OUTPUT_DIR = Path("mountaincar_results")

DISCRETIZATIONS = [10, 20, 40, 100]
NUM_TRAIN_EPISODES = 25_000
NUM_TEST_EPISODES = 1_000
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
ROLLING_WINDOW = 500
TRAINING_BUDGETS = [25_000, 50_000, 100_000, 250_000]
SEED = 42

# epsilon(k) = epsilon_min + (epsilon_start - epsilon_min) * exp(-lambda*k)
# At episode 25,000 the exponential term is about 0.0006, 0.0067, and 0.0821.
EPSILON_DECAY_VALUES = {
    "fast": 3.0e-4,
    "medium": 2.0e-4,
    "slow": 1.0e-4,
}


@dataclass
class TrainingResult:
    discretization: int
    decay_name: str
    epsilon_decay: float
    training_episodes: int
    q_table: np.ndarray
    state_visits: np.ndarray
    state_action_visits: np.ndarray
    successes: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    rolling_success: np.ndarray
    test_success_rate: float = np.nan


def make_bin_edges(env, num_position_bins, num_velocity_bins):
    """Return internal bin edges, matching the original notebook logic."""
    position_bins = np.linspace(
        env.observation_space.low[0],
        env.observation_space.high[0],
        num_position_bins + 1,
    )[1:-1]
    velocity_bins = np.linspace(
        env.observation_space.low[1],
        env.observation_space.high[1],
        num_velocity_bins + 1,
    )[1:-1]
    return position_bins, velocity_bins


def discretize_state(state, position_bins, velocity_bins):
    """Map continuous (position, velocity) to a pair of discrete indices."""
    position, velocity = state
    return (
        int(np.digitize(position, position_bins)),
        int(np.digitize(velocity, velocity_bins)),
    )


def epsilon_at_episode(episode, epsilon_start, epsilon_min, decay_rate):
    """Exponential decay with a nonzero exploration floor."""
    return epsilon_min + (epsilon_start - epsilon_min) * np.exp(
        -decay_rate * episode
    )


def greedy_action(q_values, rng):
    """Greedy action selection with the notebook's random tie-breaking."""
    max_actions = np.flatnonzero(q_values == np.max(q_values))
    return int(rng.choice(max_actions))


def compute_rolling_success(successes, window=ROLLING_WINDOW):
    """Trailing success mean; early entries use all observations so far."""
    successes = np.asarray(successes, dtype=float)
    if successes.size == 0:
        return successes
    cumulative = np.cumsum(np.insert(successes, 0, 0.0))
    result = np.empty_like(successes)
    for end in range(1, successes.size + 1):
        start = max(0, end - window)
        result[end - 1] = (cumulative[end] - cumulative[start]) / (end - start)
    return result


def train_q_learning(
    discretization,
    num_episodes,
    decay_name,
    epsilon_decay,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_min=EPSILON_MIN,
    rolling_window=ROLLING_WINDOW,
    seed=SEED,
    progress_every=5_000,
):
    """Train one fresh agent and collect state and state-action coverage."""
    env = gym.make("MountainCar-v0")
    position_bins, velocity_bins = make_bin_edges(
        env, discretization, discretization
    )
    num_actions = env.action_space.n
    q_table = np.zeros((discretization, discretization, num_actions))
    state_visits = np.zeros((discretization, discretization), dtype=np.int64)
    state_action_visits = np.zeros_like(q_table, dtype=np.int64)
    successes = np.zeros(num_episodes, dtype=np.int8)
    episode_returns = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes, dtype=np.int16)
    rng = np.random.default_rng(seed)

    for episode in range(num_episodes):
        # Identical reset seeds across configurations make comparisons paired.
        continuous_state, _ = env.reset(seed=seed + episode)
        state = discretize_state(continuous_state, position_bins, velocity_bins)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        epsilon = epsilon_at_episode(
            episode, epsilon_start, epsilon_min, epsilon_decay
        )

        while not (terminated or truncated):
            state_visits[state] += 1
            if rng.random() < epsilon:
                # Using the same RNG (instead of action_space.sample) makes runs
                # exactly reproducible and fair across experiment settings.
                action = int(rng.integers(num_actions))
            else:
                action = greedy_action(q_table[state], rng)

            next_continuous_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(
                next_continuous_state, position_bins, velocity_bins
            )
            target = reward if terminated else reward + gamma * np.max(q_table[next_state])
            q_table[state][action] += alpha * (target - q_table[state][action])
            state_action_visits[state][action] += 1
            state = next_state
            total_reward += reward
            steps += 1

        successes[episode] = int(terminated)
        episode_returns[episode] = total_reward
        episode_lengths[episode] = steps
        if progress_every and (episode + 1) % progress_every == 0:
            recent = successes[max(0, episode + 1 - rolling_window) : episode + 1]
            print(
                f"N={discretization:3d} {decay_name:>6s} | "
                f"episode {episode + 1:7,d} | epsilon={epsilon:.3f} | "
                f"rolling success={np.mean(recent):.3f}"
            )

    env.close()
    return TrainingResult(
        discretization=discretization,
        decay_name=decay_name,
        epsilon_decay=epsilon_decay,
        training_episodes=num_episodes,
        q_table=q_table,
        state_visits=state_visits,
        state_action_visits=state_action_visits,
        successes=successes,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        rolling_success=compute_rolling_success(successes, rolling_window),
    )


def evaluate_policy(q_table, num_test_episodes=NUM_TEST_EPISODES, seed=SEED + 1_000_000):
    """Evaluate with epsilon=0 using fresh episodes and random greedy ties."""
    env = gym.make("MountainCar-v0")
    n_position, n_velocity, _ = q_table.shape
    position_bins, velocity_bins = make_bin_edges(env, n_position, n_velocity)
    rng = np.random.default_rng(seed)
    successes = np.zeros(num_test_episodes, dtype=np.int8)
    lengths = np.zeros(num_test_episodes, dtype=np.int16)

    for episode in range(num_test_episodes):
        continuous_state, _ = env.reset(seed=seed + episode)
        state = discretize_state(continuous_state, position_bins, velocity_bins)
        terminated = truncated = False
        while not (terminated or truncated):
            action = greedy_action(q_table[state], rng)
            continuous_state, _, terminated, truncated, _ = env.step(action)
            state = discretize_state(continuous_state, position_bins, velocity_bins)
            lengths[episode] += 1
        successes[episode] = int(terminated)
    env.close()
    return float(np.mean(successes)), successes, lengths


def visitation_statistics(result):
    """Coverage and sampling-density statistics for a completed run."""
    states = result.state_visits
    state_actions = result.state_action_visits
    visited_states = int(np.count_nonzero(states >= 1))
    updated_pairs = int(np.count_nonzero(state_actions >= 1))
    total_states = states.size
    total_pairs = state_actions.size
    return {
        "total_states": total_states,
        "states_visited": visited_states,
        "state_visit_percentage": 100.0 * visited_states / total_states,
        "states_visited_5_plus": int(np.count_nonzero(states >= 5)),
        "states_visited_10_plus": int(np.count_nonzero(states >= 10)),
        "total_state_actions": total_pairs,
        "state_actions_updated": updated_pairs,
        "state_action_update_percentage": 100.0 * updated_pairs / total_pairs,
        "state_actions_updated_5_plus": int(np.count_nonzero(state_actions >= 5)),
        "state_actions_updated_10_plus": int(np.count_nonzero(state_actions >= 10)),
        "average_visits_per_visited_state": (
            float(states.sum() / visited_states) if visited_states else 0.0
        ),
        "average_updates_per_visited_state_action": (
            float(state_actions.sum() / updated_pairs) if updated_pairs else 0.0
        ),
    }


def summary_row(result):
    stats = visitation_statistics(result)
    return {
        "discretization": result.discretization,
        "epsilon_schedule": result.decay_name,
        "epsilon_decay": result.epsilon_decay,
        "training_episodes": result.training_episodes,
        "test_success_rate": result.test_success_rate,
        "final_rolling_success_rate": float(result.rolling_success[-1]),
        **stats,
    }


def plot_rolling_success(results, output_dir=None):
    """Put all decay schedules for each resolution on the same axes."""
    for discretization in sorted({r.discretization for r in results}):
        fig, ax = plt.subplots(figsize=(10, 5))
        for result in results:
            if result.discretization == discretization:
                ax.plot(
                    np.arange(1, result.training_episodes + 1),
                    result.rolling_success,
                    label=f"{result.decay_name} (lambda={result.epsilon_decay:g})",
                )
        ax.set(xlabel="Training episode", ylabel="Rolling success rate", ylim=(0, 1.02))
        ax.set_title(f"MountainCar {discretization}x{discretization}: rolling success")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if output_dir:
            fig.savefig(output_dir / f"rolling_success_{discretization}x{discretization}.png", dpi=160)
        plt.show()


def plot_state_visitation(result, output_dir=None):
    """Plot log(1 + count) over the physical position-velocity plane."""
    env = gym.make("MountainCar-v0")
    extent = [*env.observation_space.low, *env.observation_space.high]
    # imshow expects [xmin, xmax, ymin, ymax], not low-position/low-velocity first.
    extent = [extent[0], extent[2], extent[1], extent[3]]
    env.close()
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(
        np.log1p(result.state_visits).T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="log(1 + state visits)")
    ax.set(xlabel="Position", ylabel="Velocity")
    ax.set_title(
        f"State visitation: {result.discretization}x{result.discretization}, "
        f"{result.decay_name} decay"
    )
    fig.tight_layout()
    if output_dir:
        fig.savefig(
            output_dir / f"visitation_{result.discretization}x{result.discretization}_{result.decay_name}.png",
            dpi=160,
        )
    plt.show()


def plot_policy(result, output_dir=None):
    """Reuse the original physical-plane greedy policy visualization."""
    env = gym.make("MountainCar-v0")
    low, high = env.observation_space.low, env.observation_space.high
    env.close()
    cmap = ListedColormap(["tab:blue", "tab:gray", "tab:orange"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(
        np.argmax(result.q_table, axis=2).T,
        origin="lower",
        aspect="auto",
        extent=[low[0], high[0], low[1], high[1]],
        cmap=cmap,
        norm=norm,
    )
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["Left", "Coast", "Right"])
    ax.axvline(0.5, linestyle="--", color="purple", label="Goal")
    ax.set(xlabel="Position", ylabel="Velocity")
    ax.set_title(
        f"Greedy policy: {result.discretization}x{result.discretization}, "
        f"{result.decay_name} decay"
    )
    ax.legend()
    fig.tight_layout()
    if output_dir:
        fig.savefig(
            output_dir / f"policy_{result.discretization}x{result.discretization}_{result.decay_name}.png",
            dpi=160,
        )
    plt.show()


def best_result_by_resolution(results):
    """Select schedules by independent test score, then final rolling score."""
    best = {}
    for result in results:
        key = result.discretization
        score = (result.test_success_rate, result.rolling_success[-1])
        if key not in best or score > (
            best[key].test_success_rate,
            best[key].rolling_success[-1],
        ):
            best[key] = result
    return best


def run_epsilon_experiment(output_dir: Optional[Path] = None):
    results = []
    for discretization in DISCRETIZATIONS:
        for decay_name, decay_rate in EPSILON_DECAY_VALUES.items():
            result = train_q_learning(
                discretization,
                NUM_TRAIN_EPISODES,
                decay_name,
                decay_rate,
            )
            result.test_success_rate, _, _ = evaluate_policy(result.q_table)
            results.append(result)
            print(pd.Series(summary_row(result)).to_string())
    plot_rolling_success(results, output_dir)
    return results


def run_training_budget_experiment(best_decay_name, best_decay_rate, output_dir=None):
    results = []
    for budget in TRAINING_BUDGETS:
        result = train_q_learning(
            100, budget, best_decay_name, best_decay_rate
        )
        result.test_success_rate, _, _ = evaluate_policy(result.q_table)
        results.append(result)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r.training_episodes for r in results],
        [r.test_success_rate for r in results],
        marker="o",
    )
    ax.set(xlabel="Training episodes", ylabel="Independent test success rate", ylim=(0, 1.02))
    ax.set_title("100x100 discretization: effect of training budget")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_dir:
        fig.savefig(output_dir / "100x100_training_budget.png", dpi=160)
    plt.show()
    return results


def run_all_experiments():
    """Run enabled experiments, produce plots, and save one summary table."""
    output_dir = OUTPUT_DIR if SAVE_RESULTS else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    epsilon_results = []
    best = {}
    if RUN_EPSILON_COMPARISON:
        epsilon_results = run_epsilon_experiment(output_dir)
        all_results.extend(epsilon_results)
        best = best_result_by_resolution(epsilon_results)

    if RUN_VISITATION_ANALYSIS and best:
        for result in best.values():
            plot_state_visitation(result, output_dir)
        # Policy comparison is most informative for the requested 20x20/100x100 pair.
        for discretization in (20, 100):
            if discretization in best:
                plot_policy(best[discretization], output_dir)

    if RUN_100X100_BUDGET_TEST:
        if 100 not in best:
            raise RuntimeError(
                "Enable RUN_EPSILON_COMPARISON first so the best 100x100 schedule "
                "can be selected independently."
            )
        budget_results = run_training_budget_experiment(
            best[100].decay_name, best[100].epsilon_decay, output_dir
        )
        all_results.extend(budget_results)

    summary = pd.DataFrame(summary_row(result) for result in all_results)
    if not summary.empty:
        summary = summary.sort_values(
            ["discretization", "training_episodes", "epsilon_decay"]
        ).reset_index(drop=True)
        print("\nExperiment summary:\n")
        print(summary.to_string(index=False))
        if output_dir:
            summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    return summary, all_results


if __name__ == "__main__":
    run_all_experiments()
