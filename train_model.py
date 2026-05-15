"""
Stress Level Classification Model
Trains and evaluates multiple ML models, selects the best one, and saves it.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import joblib
import os

# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
DATA_PATH = r'c:\Users\ragha\Desktop\stress\stress_dataset.csv'
OUTPUT_DIR = r'c:\Users\ragha\Desktop\stress'

df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("STRESS LEVEL CLASSIFICATION — MODEL TRAINING")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"Class distribution:\n{df['stress_level'].value_counts()}\n")

# ──────────────────────────────────────────────
# 2. Prepare Features & Target
# ──────────────────────────────────────────────
feature_cols = [
    'nervousness_frequency', 'sleep_quality', 'overwhelmed_frequency',
    'mood_description', 'physical_symptoms', 'concentration_ability',
    'irritability_frequency'
]

X = df[feature_cols]
le = LabelEncoder()
y = le.fit_transform(df['stress_level'])  # High=0, Low=1, Moderate=2
class_names = le.classes_
print(f"Label encoding: {dict(zip(class_names, le.transform(class_names)))}\n")

# ──────────────────────────────────────────────
# 3. Train/Test Split (80/20, stratified)
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples\n")

# ──────────────────────────────────────────────
# 4. Train & Compare Multiple Models
# ──────────────────────────────────────────────
models = {
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
print("-" * 60)
print(f"{'Model':<25} {'CV Accuracy (5-fold)':<22} {'Test Accuracy'}")
print("-" * 60)

for name, model in models.items():
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Fit on full training set & predict on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_acc': test_acc,
        'y_pred': y_pred,
    }
    print(f"{name:<25} {cv_scores.mean():.4f} ± {cv_scores.std():.4f}      {test_acc:.4f}")

# ──────────────────────────────────────────────
# 5. Select Best Model
# ──────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['test_acc'])
best = results[best_name]
print(f"\n{'=' * 60}")
print(f">> Best Model: {best_name} (Test Accuracy: {best['test_acc']:.4f})")
print(f"{'=' * 60}")

# ──────────────────────────────────────────────
# 6. Detailed Evaluation of Best Model
# ──────────────────────────────────────────────
print(f"\n--- Classification Report ({best_name}) ---\n")
print(classification_report(y_test, best['y_pred'], target_names=class_names))

# ──────────────────────────────────────────────
# 7. Visualizations
# ──────────────────────────────────────────────

# --- 7a. Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(results.keys())
cv_means = [results[m]['cv_mean'] for m in model_names]
test_accs = [results[m]['test_acc'] for m in model_names]
x = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x - width/2, cv_means, width, label='CV Accuracy (5-fold)', color='#5B8DEF')
bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='#F76E6E')
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison — Stress Level Classification')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylim(0.5, 1.0)
ax.legend()
ax.bar_label(bars1, fmt='%.3f', fontsize=8)
ax.bar_label(bars2, fmt='%.3f', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
print("\n[CHART] Saved: model_comparison.png")

# --- 7b. Confusion Matrix ---
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, best['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
print("[CHART] Saved: confusion_matrix.png")

# --- 7c. Feature Importance (if available) ---
if hasattr(best['model'], 'feature_importances_'):
    importances = best['model'].feature_importances_
    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_cols)[sorted_idx], importances[sorted_idx], color='#47B39C')
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance — {best_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    print("[CHART] Saved: feature_importance.png")

# ──────────────────────────────────────────────
# 8. Save Best Model & Label Encoder
# ──────────────────────────────────────────────
model_path = os.path.join(OUTPUT_DIR, 'stress_model.pkl')
encoder_path = os.path.join(OUTPUT_DIR, 'label_encoder.pkl')
joblib.dump(best['model'], model_path)
joblib.dump(le, encoder_path)
print(f"\n[SAVED] Model:   {model_path}")
print(f"[SAVED] Encoder: {encoder_path}")

# ──────────────────────────────────────────────
# 9. Quick Prediction Demo
# ──────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("PREDICTION DEMO")
print("=" * 60)
samples = {
    'Relaxed person':  [1, 1, 1, 1, 1, 1, 1],
    'Average person':  [3, 3, 3, 3, 3, 3, 3],
    'Stressed person': [5, 5, 5, 5, 5, 5, 5],
    'Mixed signals':   [4, 2, 3, 1, 4, 2, 3],
}
for label, values in samples.items():
    pred = le.inverse_transform(best['model'].predict([values]))[0]
    print(f"  {label:20s} {values} -> {pred}")

print("\nDone!")
