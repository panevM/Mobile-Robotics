import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

num_episodes = 25_000

gamma = 0.99

# This is the "overall" learning-rate parameter.
# Since num_tilings features are active at once,
# we divide it by num_tilings during each update.
alpha = 0.5

epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.00025

num_tilings = 8
tiles_per_dim = 8

seed = 42

rng = np.random.default_rng(seed)


# ============================================================
# ENVIRONMENT
# ============================================================

env = gym.make("MountainCar-v0")

state_low = env.observation_space.low
state_high = env.observation_space.high

num_actions = env.action_space.n

print("State low:", state_low)
print("State high:", state_high)
print("Number of actions:", num_actions)

# %%

# ============================================================
# TILE CODING
# ============================================================

num_features_per_tiling = tiles_per_dim * tiles_per_dim

num_features = num_tilings * num_features_per_tiling


def get_active_features(state):
    """
    Return the indices of the active tile features for a
    continuous MountainCar state.

    Exactly one feature is active in each tiling.
    """

    position, velocity = state

    active_features = []

    # Width of one tile in physical state coordinates
    tile_width_position = (
        state_high[0] - state_low[0]
    ) / tiles_per_dim

    tile_width_velocity = (
        state_high[1] - state_low[1]
    ) / tiles_per_dim

    for tiling in range(num_tilings):

        # Different offsets for the two dimensions
        offset_fraction_position = tiling / num_tilings

        offset_fraction_velocity = (
            (2 * tiling + 1) % num_tilings
        ) / num_tilings

        offset_position = (
            offset_fraction_position
            * tile_width_position
        )

        offset_velocity = (
            offset_fraction_velocity
            * tile_width_velocity
        )

        # Determine tile coordinates in this tiling
        position_index = int(
            np.floor(
                (
                    position
                    - state_low[0]
                    + offset_position
                )
                / tile_width_position
            )
        )

        velocity_index = int(
            np.floor(
                (
                    velocity
                    - state_low[1]
                    + offset_velocity
                )
                / tile_width_velocity
            )
        )

        # Keep indices inside the grid
        position_index = np.clip(
            position_index,
            0,
            tiles_per_dim - 1
        )

        velocity_index = np.clip(
            velocity_index,
            0,
            tiles_per_dim - 1
        )

        # Convert the 2D tile coordinate into one local index
        local_index = (
            position_index * tiles_per_dim
            + velocity_index
        )

        # Each tiling owns a different part of the feature vector
        global_index = (
            tiling * num_features_per_tiling
            + local_index
        )

        active_features.append(global_index)

    return np.array(active_features, dtype=int)

# %%

# ============================================================
# LINEAR FUNCTION APPROXIMATOR
# ============================================================

weights = np.zeros(
    (num_actions, num_features)
)

# %%

def q_value(state, action):
    active = get_active_features(state)

    return np.sum(
        weights[action, active]
    )

# %%

# ============================================================
# SEMI-GRADIENT Q-LEARNING
# ============================================================

rolling_window = 500
num_test_episodes = 1_000


def q_values(state):
    """Return Q(s, a) for every action using one feature lookup."""
    active = get_active_features(state)
    return np.sum(weights[:, active], axis=1)


def greedy_action(state, random_generator):
    """Greedy action with random tie-breaking."""
    values = q_values(state)
    max_actions = np.flatnonzero(values == np.max(values))
    return int(random_generator.choice(max_actions))


def epsilon_greedy_action(state, epsilon, random_generator):
    if random_generator.random() < epsilon:
        return int(random_generator.integers(num_actions))
    return greedy_action(state, random_generator)


def epsilon_for_episode(episode):
    """Exponential schedule with a configurable nonzero floor."""
    return epsilon_min + (epsilon_start - epsilon_min) * np.exp(
        -epsilon_decay * episode
    )


def rolling_mean(values, window):
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values)
    for end in range(1, len(values) + 1):
        start = max(0, end - window)
        result[end - 1] = (cumulative[end] - cumulative[start]) / (end - start)
    return result


def train_q_learning():
    """Train the shared linear approximator from freshly zeroed weights."""
    global weights

    weights = np.zeros((num_actions, num_features), dtype=float)
    train_rng = np.random.default_rng(seed)
    successes = np.zeros(num_episodes, dtype=np.int8)
    episode_returns = np.zeros(num_episodes, dtype=float)
    episode_lengths = np.zeros(num_episodes, dtype=np.int16)
    step_size = alpha / num_tilings

    for episode in range(num_episodes):
        # Seeding every reset gives reproducible initial states.
        state, _ = env.reset(seed=seed + episode)
        epsilon = epsilon_for_episode(episode)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = epsilon_greedy_action(state, epsilon, train_rng)
            active = get_active_features(state)
            prediction = np.sum(weights[action, active])

            next_state, reward, terminated, truncated, _ = env.step(action)

            # MountainCar's goal is a true terminal state, so it has no
            # continuation value. A time-limit truncation is not an MDP
            # terminal state, and therefore still bootstraps.
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(q_values(next_state))

            td_error = target - prediction
            weights[action, active] += step_size * td_error

            state = next_state
            total_reward += reward
            steps += 1

        successes[episode] = int(terminated)
        episode_returns[episode] = total_reward
        episode_lengths[episode] = steps

        if (episode + 1) % 1_000 == 0:
            recent_success = np.mean(successes[max(0, episode - 999):episode + 1])
            print(
                f"Episode {episode + 1:5d} | epsilon={epsilon:.3f} | "
                f"recent success={recent_success:.3f} | "
                f"recent steps={np.mean(episode_lengths[max(0, episode - 999):episode + 1]):.1f}"
            )

    return successes, episode_returns, episode_lengths


def evaluate_policy(test_episodes=num_test_episodes):
    """Independent epsilon=0 evaluation of the learned weights."""
    test_env = gym.make("MountainCar-v0")
    test_rng = np.random.default_rng(seed + 1_000_000)
    test_successes = np.zeros(test_episodes, dtype=np.int8)
    test_lengths = np.zeros(test_episodes, dtype=np.int16)

    for episode in range(test_episodes):
        state, _ = test_env.reset(seed=seed + 1_000_000 + episode)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = greedy_action(state, test_rng)
            state, _, terminated, truncated, _ = test_env.step(action)
            test_lengths[episode] += 1

        test_successes[episode] = int(terminated)

    test_env.close()
    return test_successes, test_lengths


successes, episode_returns, episode_lengths = train_q_learning()
rolling_success = rolling_mean(successes, rolling_window)
test_successes, test_episode_lengths = evaluate_policy()

print(f"\nTraining final rolling success: {rolling_success[-1]:.3f}")
print(f"Test success: {np.mean(test_successes):.3f}")
print(f"Test average episode length: {np.mean(test_episode_lengths):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(np.arange(1, num_episodes + 1), rolling_success)
axes[0].set(
    xlabel="Training episode",
    ylabel=f"Success rate over previous {rolling_window} episodes",
    title="Tile-coded Q-learning success",
    ylim=(0, 1.02),
)
axes[0].grid(alpha=0.3)

rolling_length = rolling_mean(episode_lengths, rolling_window)
axes[1].plot(np.arange(1, num_episodes + 1), rolling_length)
axes[1].set(
    xlabel="Training episode",
    ylabel=f"Steps over previous {rolling_window} episodes",
    title="Tile-coded Q-learning episode length",
)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
