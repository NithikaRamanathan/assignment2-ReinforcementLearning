"""
Lunar Lander
Made with Gymnasium
January 2026 - Machine Learning Classes
University Carlos III of Madrid

This version integrates:
- Phase 1: manual discretization + reward + Q-learning + tuning
- Phase 2: VQQL + experiments with number of clusters + training/evaluation
- Phase 3: comparison of Q-learning models + comparison with Tutorial 1 and Assignment 1

Modes:
- TRAIN: learn a Q-table
- PLAY: load a Q-table and watch the learned agent
- "VQQL" -> use the quantizer
- "MANUAL" -> use a hand-crafted discretization
"""

import gymnasium as gym
import time
import pygame
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================

# Moon gravity   -> -1.62
# Mars gravity   -> -3.72
# Earth gravity  -> -9.81
GRAVITY = -1.62

ENV_NAME = "LunarLander-v3"
QUANTIZER_FILE = "lunarlander_vq.pkl"
QTABLE_FILE = "qtable_highest.txt"

# MODE can be: "TRAIN", "PLAY"
MODE = "PLAY"

# State representation mode:
# "VQQL" -> use the quantizer
# "MANUAL" -> use a hand-crafted discretization
STATE_MODE = "MANUAL"

# If True, use the custom reward from compute_reward()
# If False, use the raw reward from Gymnasium
USE_CUSTOM_REWARD = True

# If True, render the environment during training (very slow)
RENDER_TRAINING = True

# Training parameters
EPISODES = 15000
LEARNING_RATE = 0.2
DISCOUNT_RATE = 0.98

# Exploration parameters
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.002

# ACTION_REPEAT reduces the frequency of decision-making by repeating
# the same action for multiple environment steps.
# This helps stabilize learning and reduce redundant state transitions.
ACTION_REPEAT = 5

# Evaluation / play parameters
PLAY_EPISODES = 10

# Action definitions
ACTION_NOTHING = 0
ACTION_LEFT_ENGINE = 1
ACTION_MAIN_ENGINE = 2
ACTION_RIGHT_ENGINE = 3


# GAME STATE CLASS
class GameState:
    def __init__(self, observation):
        """
        Initialize game state from Gymnasium observation.

        The LunarLander-v3 observation space consists of 8 values:
        - obs[0]: x position
        - obs[1]: y position
        - obs[2]: x velocity
        - obs[3]: y velocity
        - obs[4]: angle
        - obs[5]: angular velocity
        - obs[6]: left leg contact
        - obs[7]: right leg contact
        """
        self.x_position = observation[0]
        self.y_position = observation[1]
        self.x_velocity = observation[2]
        self.y_velocity = observation[3]
        self.angle = observation[4]
        self.angular_velocity = observation[5]
        self.left_leg_contact = observation[6]
        self.right_leg_contact = observation[7]

        # Store raw observation for convenience
        self.observation = observation

        # Score tracking
        self.score = 0.0
        self.episode_reward = 0.0

        # Current action
        self.action = ACTION_NOTHING

    def update(self, observation, reward):
        """Update state with new observation and reward."""
        self.x_position = observation[0]
        self.y_position = observation[1]
        self.x_velocity = observation[2]
        self.y_velocity = observation[3]
        self.angle = observation[4]
        self.angular_velocity = observation[5]
        self.left_leg_contact = observation[6]
        self.right_leg_contact = observation[7]
        self.observation = observation
        self.episode_reward += reward
        self.score = self.episode_reward

    def reset(self, observation):
        """Reset state for a new episode."""
        self.__init__(observation)

# ==========================================
# PHASE 1 STEP 1 - MANUAL DISCRETIZATION
# ==========================================
        
def get_manual_state_space():
    """
    TODO (Phase 1 - Step 1):

    Return the total number of discrete states.

    You must compute the size of the state space based on the number of bins
    defined for each variable.

    For example:
    - x_position: 5 bins
    - y_position: 5 bins
    - velocities: 3 bins each
    - angle: 3 bins
    - angular velocity: 3 bins
    - leg contact: 3 bins

    The total number of states is the product of all these values.

    This value determines the number of rows in the Q-table.
    """   

    # give more bins to the variables that are more relevant
    # order of significance determined from assignment 1 from InfoGain ranker:
        # angle, angular_velocity, y_position, x_position, x_velocity, y_velocity, right_leg_contact, left_leg_contact
 
    angle_bins = 5
    angular_velocity_bins = 4
    y_bins = 5
    x_bins = 5
    x_velocity_bins = 5
    y_velocity_bins = 3
    left_leg_bins = 2
    right_leg_bins = 2

    total_states = (
        angle_bins *
        angular_velocity_bins *
        y_bins *
        x_bins *
        x_velocity_bins *
        y_velocity_bins *
        left_leg_bins *
        right_leg_bins
    )

    return total_states


def manual_state_id(game):
    """
    TODO (Phase 1 - Step 1):

    Convert the continuous state of the environment into a unique discrete state ID.

    Steps:
    1. Discretize each variable (position, velocity, angle, etc.).
    2. Combine all discrete values into a single integer index.

    You should use a mixed-radix encoding:
    - Each variable contributes to the final index.
    - The order of combination must be consistent.

    IMPORTANT:
    The resulting state_id must be an integer in the range:
    [0, total_number_of_states - 1]

    This ID will be used as the row index in the Q-table.
    """

    x_pos = game.x_position
    y_pos = game.y_position
    x_vel = game.x_velocity
    y_vel = game.y_velocity
    angle = game.angle
    angular_vel = game.angular_velocity
    left_leg = game.left_leg_contact
    right_leg = game.right_leg_contact

    # using arbitrary interval discretiation bc i can  set intervals that make sense for a specific application
    # np.digitize returns the indices of the bins to which each value in input array belongs.


    angle_bin = np.digitize(angle, [-0.3, -0.1, 0.1, 0.3]) # digitize data into bins

    angular_vel_bin = np.digitize(angular_vel, [-0.5, 0.0, 0.5])

    y_bin = np.digitize(y_pos, [0, 0.5, 1.0, 1.5])

    x_bin = np.digitize(x_pos, [-1.0, -0.5, 0.0, 0.5])

    x_vel_bin = np.digitize(x_vel, [-1.0, -0.5, 0.0, 0.5])

    y_vel_edges = np.array([-0.5, 0.5])
    y_vel_bin = np.digitize(y_vel, y_vel_edges)

    left_leg_bin = int(left_leg)
    right_leg_bin = int(right_leg)

    # mixed radix encoding

    state_id = angle_bin
    state_id = state_id * 4 + angular_vel_bin
    state_id = state_id * 5 + y_bin
    state_id = state_id * 5 + x_bin
    state_id = state_id * 5 + x_vel_bin
    state_id = state_id * 3 + y_vel_bin
    state_id = state_id * 2 + left_leg_bin
    state_id = state_id * 2 + right_leg_bin



    return int(state_id)

# ==========================================
# PHASE 1 STEP 2 - NEW REWARD FUNCTION
# ==========================================

def compute_reward(observation, raw_reward, terminated, truncated):
    """
    TODO:
    Compute a custom reward based on the environment reward
    and additional shaping terms.

    observation format:
    [x_position, y_position, x_velocity, y_velocity,
     angle, angular_velocity, left_leg, right_leg]
    """

    x_position = observation[0]
    y_position = observation[1]
    x_velocity = observation[2]
    y_velocity = observation[3]
    angle = observation[4]
    angular_velocity = observation[5]
    left_leg = observation[6]
    right_leg = observation[7]

    # Start from the original Gymnasium reward
    reward = raw_reward

    """
    TODO (Phase 1 - Step 2):

    Design a custom reward function to guide the agent's learning.

    Start from the original reward provided by the environment and
    modify it by adding shaping terms.

    Possible ideas:
    - Penalize horizontal distance from the landing zone
    - Penalize high vertical speed (hard landings)
    - Penalize large angles (instability)
    - Reward stable landing (leg contact)
    - Reward smooth and controlled descent

    IMPORTANT:
    - Do not completely ignore the original reward
    - Keep the reward values within a reasonable range
    - Avoid overly large penalties that may prevent learning

    The goal is to make learning faster and more stable.
    """

    horizontal_penalization = -0.1 * abs(x_position)
    angle_penalization = -0.1 * abs(angle)
    leg_contact_reward = 0
    if left_leg and right_leg:
        leg_contact_reward = 0.2
    
    speed = (x_velocity**2 + y_velocity**2)**0.5
    descent_penalization = -0.1 * speed

    reward += horizontal_penalization + angle_penalization + leg_contact_reward + descent_penalization

    return reward
   
# ==========================================
# PHASE 1 STEP 3 - Q LEARNING
# ==========================================

def choose_action(state_id, qtable, epsilon, env):
    """
    Choose an action using an epsilon-greedy strategy.

    Epsilon-greedy works as follows:
    - with probability epsilon, choose a random action (exploration)
    - with probability 1 - epsilon, choose the best known action
      from the Q-table (exploitation)

    Parameters:
    - state_id: current discrete state
    - qtable: Q-value table
    - epsilon: exploration probability
    - env: Gymnasium environment (used to sample random actions)

    Returns:
    - action: integer in the action space
    """
    
    if np.random.uniform(0, 1) > epsilon:
        return int(np.argmax(qtable[state_id, :]))  # exploit
    return env.action_space.sample()                # explore


def update_qtable(qtable, state_id, action, reward, next_state_id,
                  learning_rate, discount_rate):
    """
    Apply the Q-learning Bellman update.

    Bellman equation:
    Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]

    Parameters:
    - qtable: Q-value table
    - state_id: current discrete state s
    - action: action a taken in state s
    - reward: reward received after taking action a
    - next_state_id: next discrete state s'
    - learning_rate: alpha
    - discount_rate: gamma
    """

    # ===========================================================
    # TODO: IMPLEMENT THE Q-LEARNING UPDATE FORMULA
    # ===========================================================
    # Replace the line below with the Bellman update.
    # The agent must learn from:
    # - current state
    # - chosen action
    # - received reward
    # - next state
    #
    # Tip:
    # np.max(qtable[next_state_id, :])
    
    max_q = np.max(qtable[next_state_id, :])
    qtable[state_id, action] = qtable[state_id, action] + learning_rate * (reward + discount_rate * max_q - qtable[state_id, action])


def decay_epsilon(episode, max_epsilon, min_epsilon, decay_rate):
    """
    Compute epsilon decay over time.

    The exploration rate should decrease as training progresses,
    so the agent gradually shifts from exploration to exploitation.

    A common exponential decay formula is:
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * exp(-decay_rate * episode)

    Parameters:
    - episode: current episode index
    - max_epsilon: initial epsilon
    - min_epsilon: minimum epsilon value
    - decay_rate: decay speed

    Returns:
    - epsilon value for the current episode
    """

    # ===========================================================
    # TODO: IMPLEMENT EPSILON DECAY
    # ===========================================================
    # Replace the line below with the exponential decay formula.
    
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    
    
 
# ==========================================
# PHASE 2 - VQQL DISCRETIZATION
# ==========================================

def load_quantizer(path):
    """
    Load the scaler and clustering model created in Phase 1.

    The .pkl file should contain at least:
    - scaler
    - quantizer
    - state_columns
    - n_clusters
    """
    data = joblib.load(path)

    scaler = data["scaler"]
    quantizer = data["quantizer"]
    state_columns = data["state_columns"]
    n_clusters = data["n_clusters"]

    return scaler, quantizer, state_columns, n_clusters


def extract_features(game, state_columns):
    """
    Build the feature vector in the same order used in Phase 1.

    IMPORTANT:
    The order of the variables must be exactly the same as the one used
    when the quantizer was trained.

    TODO:
    Students should verify that the state representation used here
    matches the one selected in the VQQL Discretization.
    """

    feature_map = {
        "x_position": game.x_position,
        "y_position": game.y_position,
        "x_velocity": game.x_velocity,
        "y_velocity": game.y_velocity,
        "angle": game.angle,
        "angular_velocity": game.angular_velocity,
        "left_leg": game.left_leg_contact,
        "right_leg": game.right_leg_contact,
    }

    # Build the feature vector following the order stored in the .pkl file
    features = np.array([feature_map[col] for col in state_columns], dtype=np.float64)

    return features


def discretize_state(game, scaler, quantizer, state_columns):
    """
    Convert the current continuous state into a discrete state id.

    Steps:
    1. Extract the selected features
    2. Apply the same scaling
    3. Find the nearest centroid
    4. Return its index
    """
    features = extract_features(game, state_columns)

    # Scale the features before clustering
    scaled_features = scaler.transform([features]).astype(np.float64)

    # Predict the nearest centroid index
    state_id = quantizer.predict(scaled_features)[0]

    return int(state_id)
    
    
def get_state_id(game, scaler=None, quantizer=None, state_columns=None):
    """
    Return the discrete state identifier used as row index in the Q-table.
    """
    if STATE_MODE == "VQQL":
        return discretize_state(game, scaler, quantizer, state_columns)
    elif STATE_MODE == "MANUAL":
        return manual_state_id(game)
    else:
        raise ValueError("STATE_MODE must be 'VQQL' or 'MANUAL'")
        
 
# ==========================================
# PLOTTING
# ==========================================

def plot_training_curve(rewards_all_episodes, window=100, title=None):
    if len(rewards_all_episodes) < window:
        window = max(1, len(rewards_all_episodes))

    moving_avg = np.convolve(
        rewards_all_episodes,
        np.ones(window) / window,
        mode='valid'
    )

    plt.plot(moving_avg)
    plt.title(title if title else f"Training Convergence: Moving Average of Rewards ({window} steps)")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()
    
# ==========================================
# QUIT GAME
# ==========================================

def handle_pygame_events():
    """
    Handle window close and keyboard input.
    Returns True if the program should exit.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                return True

    return False

# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 50)
    print("LUNAR LANDER - Machine Learning (UC3M)")
    print("=" * 50)
    print("\nInitializing environment...")

    pygame.init()

    # Render settings
    if MODE == "PLAY" or RENDER_TRAINING:
        render_mode = "human"
    else:
        render_mode = None

    env = gym.make(ENV_NAME, gravity=GRAVITY, render_mode=render_mode)

    print(f"Environment: {ENV_NAME}")
    print(f"Gravity: {GRAVITY}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    if STATE_MODE == "VQQL":
        scaler, quantizer, state_columns, n_clusters = load_quantizer(QUANTIZER_FILE)
        print(f"Loaded quantizer from {QUANTIZER_FILE}")
        print(f"Number of discrete states (clusters): {n_clusters}")
        print(f"State columns used: {state_columns}")
        n_states = n_clusters

    elif STATE_MODE == "MANUAL":
        scaler, quantizer, state_columns = None, None, None
        n_states = get_manual_state_space()
        print("Using manual state discretization.")

    else:
        raise ValueError("STATE_MODE must be 'VQQL' or 'MANUAL'")

    action_space = env.action_space.n

    print(f"STATE_MODE: {STATE_MODE}")
    print(f"Q-table state space size: {n_states}")

    # Load or initialize Q-table
    if os.path.exists(QTABLE_FILE):
        print(f"[{MODE} MODE] Existing Q-Table found. Loading from '{QTABLE_FILE}'...")
        qtable = np.loadtxt(QTABLE_FILE)

        if qtable.ndim == 1:
            qtable = qtable.reshape(1, -1)
    else:
        print(f"[{MODE} MODE] No previous Q-Table found. Initializing a new one with zeros...")
        qtable = np.zeros((n_states, action_space), dtype=np.float64)

    # Sanity check
    if qtable.shape != (n_states, action_space):
        raise ValueError(
            f"Loaded Q-table shape {qtable.shape} does not match expected {(n_states, action_space)}"
        )

    # ==========================================
    # ONLINE TRAINING MODE
    # ==========================================
    if MODE == "TRAIN":
        epsilon = MAX_EPSILON
        rewards_all_episodes = []

        print(f"Starting online training for {EPISODES} episodes...")

        for episode in range(EPISODES):
            observation, info = env.reset()
            game = GameState(observation)

            terminated = False
            truncated = False
            rewards_current_episode = 0.0

            while not (terminated or truncated):
                
                # Check for exit events
                if handle_pygame_events():
                    print("Exiting...")
                    env.close()
                    pygame.quit()
                    return
        
                # Current discrete state
                state_id = get_state_id(game, scaler, quantizer, state_columns)

                # Choose one action and keep it for several environment steps
                action = choose_action(state_id, qtable, epsilon, env)
                game.action = action

                total_reward = 0.0

                for _ in range(ACTION_REPEAT):
                    observation, raw_reward, terminated, truncated, info = env.step(action)

                    # Reward: raw or custom
                    if USE_CUSTOM_REWARD:
                        step_reward = compute_reward(observation, raw_reward, terminated, truncated)
                    else:
                        step_reward = raw_reward

                    total_reward += step_reward

                    if RENDER_TRAINING:
                        print(f"State: {state_id} | Action: {action} | Step reward: {step_reward:.3f}")
                        time.sleep(0.03)

                    # Stop repeating if episode ended
                    if terminated or truncated:
                        break

                # Update current game state
                game.update(observation, total_reward)

                next_state_id = get_state_id(game, scaler, quantizer, state_columns)

                # Bellman update using accumulated reward
                update_qtable(
                    qtable,
                    state_id,
                    action,
                    total_reward,
                    next_state_id,
                    LEARNING_RATE,
                    DISCOUNT_RATE
                )

                
                rewards_current_episode += total_reward

            # Epsilon decay
            epsilon = decay_epsilon(episode, MAX_EPSILON, MIN_EPSILON, DECAY_RATE)

            rewards_all_episodes.append(rewards_current_episode)

            if (episode + 1) % 100 == 0:
                last_rewards = rewards_all_episodes[-100:]
                avg_reward = np.mean(last_rewards)
                std_reward = np.std(last_rewards)

                print(
                    f"AVG last 100: {avg_reward:.2f} | "
                    f"STD: {std_reward:.2f}"
                )

            # Save Q-table after each episode
            header = "a0\ta1\ta2\ta3"
            np.savetxt(QTABLE_FILE, qtable, fmt="%12.4f", delimiter="\t", header=header)

            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Reward: {rewards_current_episode:.2f} | "
                f"Epsilon: {epsilon:.4f}"
            )

        print("Online training finished.")
        env.close()
        pygame.quit()

        plot_training_curve(
            rewards_all_episodes,
            title="Online Q-Learning Convergence"
        )
        return

    # ==========================================
    # PLAY MODE
    # ==========================================
    if MODE == "PLAY":
        print(f"Playing {PLAY_EPISODES} episodes using the loaded Q-Table (No exploration)...")

        for ep in range(PLAY_EPISODES):
            observation, info = env.reset()
            game = GameState(observation)
            terminated = False
            truncated = False

            print(f"\n--- Playing Episode {ep + 1} ---")

            while not (terminated or truncated):
            
                # Check for exit events
                if handle_pygame_events():
                    print("Exiting...")
                    env.close()
                    pygame.quit()
                    return
                    
                state_id = get_state_id(game, scaler, quantizer, state_columns)

                # Always exploit
                action = int(np.argmax(qtable[state_id, :]))
                game.action = action

                total_reward = 0.0

                for _ in range(ACTION_REPEAT):
                    observation, raw_reward, terminated, truncated, info = env.step(action)

                    if USE_CUSTOM_REWARD:
                        step_reward = compute_reward(observation, raw_reward, terminated, truncated)
                    else:
                        step_reward = raw_reward

                    total_reward += step_reward

                    if terminated or truncated:
                        break

                    time.sleep(0.05)

                game.update(observation, total_reward)

                print(
                    f"Discrete state id: {state_id} | "
                    f"Action: {action} | "
                    f"Accumulated reward: {total_reward:.3f}"
                )

            print(f"Final score: {game.score:.2f}")

        env.close()
        pygame.quit()
        return

    env.close()
    pygame.quit()
    raise ValueError("MODE must be one of: 'TRAIN', 'PLAY'")



if __name__ == "__main__":
    main()