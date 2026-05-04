"""
VQQL Discretization
January 2026 - Machine Learning Classes
University Carlos III of Madrid

This version integrates:
- Phase 1: manual discretization + reward + Q-learning + tuning
- Phase 2: VQQL + experiments with number of clusters + training/evaluation
- Phase 3: comparison of Q-learning models + comparison with Tutorial 1 and Assignment 1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import joblib

# =========================================================
# PHASE 1: State Representation with Vector Quantization
# =========================================================

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================

# Input CSV file
AGENT_CSV = "training_data.csv"

# Output file where the vector quantizer will be stored
OUTPUT_FILE = "lunarlander_vq.pkl"

# Columns that represent the STATE of the environment
# IMPORTANT: Do NOT include the action column here
STATE_COLUMNS = [
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg",
    "right_leg"
]

# Number of clusters (i.e., number of discrete states)
N_CLUSTERS = 256

# Number of samples to use from each dataset
# (keyboard and agent)
SAMPLES_PER_SOURCE = 25000


# ==========================================
# LOAD AND PREPARE DATA
# ==========================================

def load_and_mix_states(agent_csv, state_columns, samples_per_source):
    """
    Load state data from CSV file (keyboard and agent),
    select relevant columns, and mix them into a single dataset.
    """

    # Load CSV files into pandas DataFrames
    df_agent = pd.read_csv(agent_csv)

    # Keep only the state columns and remove any missing values
    # (this automatically ignores columns like "action")
    df_agent = df_agent[state_columns].dropna()

    # Determine how many samples we can take from each dataset
    # (we cannot take more than what exists)
    n_a = min(samples_per_source, len(df_agent))

    # Randomly sample from each dataset
    # This ensures contribute equally
    df_agent = df_agent.sample(n=n_a, random_state=42)

    # Shuffle the dataset
    # (so data is well mixed)
    df_agent = df_agent.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Convert to NumPy array (required by scikit-learn)
    return df_agent.to_numpy(dtype=np.float64)


# ==========================================
# BUILD VECTOR QUANTIZER
# ==========================================

def build_quantizer(states, n_clusters):
    """
    Train a vector quantizer (MiniBatchKMeans) on the state data.
    """

    # Sanity check: we need at least as many samples as clusters
    if len(states) < n_clusters:
        raise ValueError("Number of samples must be >= number of clusters.")

    # ------------------------------------------
    # STEP 1: SCALE THE DATA
    # ------------------------------------------
    # StandardScaler normalizes each feature so that:
    # - mean = 0
    # - standard deviation = 1
    #
    # This is important because KMeans uses distances,
    # and we want all variables to have similar influence.
    scaler = StandardScaler()
    states_scaled = scaler.fit_transform(states)

    # ------------------------------------------
    # STEP 2: TRAIN K-MEANS (VECTOR QUANTIZATION)
    # ------------------------------------------
    # MiniBatchKMeans is a faster version of KMeans
    # that works well for large datasets.
    quantizer = MiniBatchKMeans(
        n_clusters=n_clusters,   # number of discrete states
        random_state=42,         # reproducibility
        batch_size=1024,         # size of mini-batches
        n_init=10                # number of different initializations
    )

    # Fit the model to the scaled data
    quantizer.fit(states_scaled)

    # Return both scaler and quantizer
    # (we need both later!)
    return scaler, quantizer


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Loading and mixing state data...")

    # Load and combine data from both sources
    states = load_and_mix_states(
        AGENT_CSV,
        STATE_COLUMNS,
        SAMPLES_PER_SOURCE
    )

    # Print shape of dataset (rows, columns)
    print("States shape:", states.shape)

    print("Training vector quantizer...")

    # Train the scaler + quantizer
    scaler, quantizer = build_quantizer(states, N_CLUSTERS)
    
    # Print the centroids
    # Convert centroids back to original scale
    centroids = scaler.inverse_transform(quantizer.cluster_centers_)

    # Create a DataFrame for easier inspection
    df_centroids = pd.DataFrame(
        centroids,
        columns=STATE_COLUMNS
    )
    # Print a preview
    print("\nCentroids:")
    print(df_centroids)

    # Save the final model
    print("Saving quantizer model...")

    # Save everything needed to reuse the discretization
    joblib.dump(
        {
            "scaler": scaler,
            "quantizer": quantizer,
            "state_columns": STATE_COLUMNS,
            "n_clusters": N_CLUSTERS
        },
        OUTPUT_FILE
    )

    print(f"Quantizer saved to {OUTPUT_FILE}")
    print("Phase 1 completed successfully!")


# ==========================================
# SCRIPT ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()