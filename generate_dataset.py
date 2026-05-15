"""
Generate a synthetic stress level classification dataset.
1000 rows, 7 features (1-5 scale), 1 target label (Low/Moderate/High).
Class balance: ~30% Low, 40% Moderate, 30% High.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000
# Target distribution: 30% Low, 40% Moderate, 30% High
n_low = 300
n_moderate = 400
n_high = 300

def clamp(arr, lo=1, hi=5):
    return np.clip(np.round(arr).astype(int), lo, hi)

def generate_group(n, base_mean, base_std, noise_std=0.6):
    """Generate correlated features around a base mean with noise."""
    # Draw a per-person 'latent stress' score
    latent = np.random.normal(base_mean, base_std, size=n)
    
    features = {}
    feature_names = [
        'nervousness_frequency',
        'sleep_quality',
        'overwhelmed_frequency',
        'mood_description',
        'physical_symptoms',
        'concentration_ability',
        'irritability_frequency',
    ]
    
    for name in feature_names:
        # Each feature = latent + independent noise
        noise = np.random.normal(0, noise_std, size=n)
        features[name] = clamp(latent + noise)
    
    return pd.DataFrame(features)

# --- Low stress group (avg score 1–2) ---
df_low = generate_group(n_low, base_mean=1.5, base_std=0.4, noise_std=0.7)

# --- Moderate stress group (avg score 2.1–3.5) ---
df_mod = generate_group(n_moderate, base_mean=2.8, base_std=0.5, noise_std=0.7)

# --- High stress group (avg score 3.6–5) ---
df_high = generate_group(n_high, base_mean=4.2, base_std=0.4, noise_std=0.7)

# Combine
df = pd.concat([df_low, df_mod, df_high], ignore_index=True)

# Compute average score and assign label
feature_cols = [
    'nervousness_frequency', 'sleep_quality', 'overwhelmed_frequency',
    'mood_description', 'physical_symptoms', 'concentration_ability',
    'irritability_frequency'
]
df['avg_score'] = df[feature_cols].mean(axis=1)

def assign_label(avg):
    if avg <= 2.0:
        return 'Low'
    elif avg <= 3.5:
        return 'Moderate'
    else:
        return 'High'

df['stress_level'] = df['avg_score'].apply(assign_label)

# Drop helper column
df.drop(columns=['avg_score'], inplace=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv(r'c:\Users\ragha\Desktop\stress\stress_dataset.csv', index=False)

# Print summary
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['stress_level'].value_counts().to_string())
print(f"\nFeature statistics:")
print(df[feature_cols].describe().round(2).to_string())
print(f"\nFirst 5 rows:")
print(df.head().to_string())
