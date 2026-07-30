# =============================================================================
# CIPN IMPROVED DETECTION PIPELINE
# Improvements applied:
#   2. Reduced noise: lower sd_multiplier, higher apply_prob, lower mislabel_rate
#   3. More features: age, sex, cancer_type, drug_type + all cipn20 item scores
#   4. Larger / better model: deeper MLP with Dropout + BatchNorm
#   5. Tuned Random Forest: GridSearchCV for best hyperparameters
# =============================================================================

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    classification_report, precision_recall_fscore_support,
    roc_curve, auc as auc_fn
)

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: GENERATE IMPROVED SYNTHETIC DATASET
# Improvement #2: reduced noise parameters
# =============================================================================

# ---- Noise / signal parameters (IMPROVED) ----
vpt_shift      = 2.5    # same signal shift
cold_shift     = -0.7
sd_multiplier  = 1.0    # IMPROVED: was 1.5 → tighter distributions = cleaner signal
apply_prob     = 0.92   # IMPROVED: was 0.80 → 92% of CIPN cases now show the shift
mislabel_rate  = 0.02   # IMPROVED: was 0.05 → only 2% label noise
seed_main      = 1234

np.random.seed(seed_main)

n = 2200
df = pd.DataFrame({
    'cancer_type': np.random.choice(
        ['Breast cancer', 'Lung cancer', 'Colorectal cancer', 'Ovarian cancer'],
        size=n, p=[0.5, 0.2, 0.2, 0.1]
    ),
    'drug_type': np.random.choice(
        ['taxane', 'platinum', 'mixed (taxane, platinum, others)'],
        size=n
    ),
})
df['label'] = (np.random.rand(n) < 0.35).astype(int)

# Sex assignment
np.random.seed(42)
sexs = []
for _, row in df.iterrows():
    if str(row.get('cancer_type', '')).strip().lower() == 'breast cancer':
        sexs.append('M' if np.random.rand() <= 0.01 else 'F')
    else:
        sexs.append(np.random.choice(['M', 'F'], p=[0.4, 0.6]))
df['sex'] = sexs

# Age
np.random.seed(1)
df['age'] = np.random.randint(30, 75, size=len(df))
bins   = [30, 45, 60, 75]
labels = ['Young', 'Middle', 'Old']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Reference table
ref = {
    'Young':  {'M': {'vpt_mean': 3.0, 'vpt_sd': 0.6, 'cold_mean': 30.7, 'cold_sd': 0.8},
               'F': {'vpt_mean': 2.6, 'vpt_sd': 0.5, 'cold_mean': 31.0, 'cold_sd': 0.7}},
    'Middle': {'M': {'vpt_mean': 4.5, 'vpt_sd': 1.0, 'cold_mean': 30.2, 'cold_sd': 1.0},
               'F': {'vpt_mean': 4.0, 'vpt_sd': 0.9, 'cold_mean': 30.7, 'cold_sd': 0.8}},
    'Old':    {'M': {'vpt_mean': 7.0, 'vpt_sd': 1.5, 'cold_mean': 29.5, 'cold_sd': 0.9},
               'F': {'vpt_mean': 6.2, 'vpt_sd': 1.4, 'cold_mean': 30.0, 'cold_sd': 0.9}},
}

np.random.seed(seed_main)
vpt_vals, cold_vals = [], []
for _, row in df.iterrows():
    ag  = row['age_group']
    sex = row['sex']
    lbl = int(row['label'])
    if pd.isna(ag) or sex not in ['M', 'F']:
        ag = 'Middle'; sex = 'F'
    r   = ref[ag][sex]
    vsd = r['vpt_sd']  * sd_multiplier   # IMPROVED: sd_multiplier = 1.0
    csd = r['cold_sd'] * sd_multiplier
    if lbl == 1 and np.random.rand() < apply_prob:  # IMPROVED: apply_prob = 0.92
        vmean = r['vpt_mean']  + vpt_shift
        cmean = r['cold_mean'] + cold_shift
    else:
        vmean = r['vpt_mean']
        cmean = r['cold_mean']
    vpt_vals.append(round(max(0.1, np.random.normal(vmean, vsd)), 2))
    cold_vals.append(round(float(np.random.normal(cmean, csd)), 2))
df['vpt']       = vpt_vals
df['cold_temp'] = cold_vals

# Drives + CIPN-20 items
np.random.seed(2023)
drive_probs = df['age'].apply(lambda a: 0.85 if a < 60 else 0.45)
df['drives'] = (np.random.rand(len(df)) < drive_probs).astype(bool)

np.random.seed(2025)
def gen_item_probs(label, base_prob, shift=0.08):
    probs = np.array(base_prob, dtype=float)
    if label == 1:
        probs[0] = max(0, probs[0] - 0.7 * shift)
        probs[1] = max(0, probs[1] - 0.2 * shift)
        probs[2] = max(0, probs[2] + 0.6 * shift)
        probs[3] = max(0, probs[3] + 0.3 * shift)
    probs = probs / probs.sum()
    return probs

base_prob_sensory   = [0.55, 0.30, 0.10, 0.05]
base_prob_motor     = [0.60, 0.28, 0.08, 0.04]
base_prob_autonomic = [0.70, 0.22, 0.06, 0.02]
sensory_idx         = list(range(1, 10))
motor_idx           = list(range(10, 18))
autonomic_idx       = list(range(18, 21))

for i in range(1, 21):
    col  = f'cipn20_i{i}'
    vals = []
    for _, row in df.iterrows():
        lbl = int(row['label'])
        if i == 19:
            if not row['drives']:
                vals.append(np.nan); continue
            probs = gen_item_probs(lbl, base_prob_autonomic)
        elif i == 20:
            if row['sex'] != 'M':
                vals.append(np.nan); continue
            probs = gen_item_probs(lbl, base_prob_autonomic)
        elif i in sensory_idx:
            probs = gen_item_probs(lbl, base_prob_sensory)
        elif i in motor_idx:
            probs = gen_item_probs(lbl, base_prob_motor)
        else:
            probs = gen_item_probs(lbl, base_prob_autonomic)
        vals.append(np.random.choice([1, 2, 3, 4], p=probs))
    df[col] = vals

# Subscales
def compute_subscale(df, indices, prefix):
    cols     = [f'cipn20_i{i}' for i in indices]
    present  = df[cols].notna().sum(axis=1)
    mean_raw = df[cols].mean(axis=1)
    required = np.ceil(len(cols) / 2.0)
    transformed = ((mean_raw - 1.0) / 3.0) * 100.0
    transformed[present < required] = np.nan
    df[f'cipn20_{prefix}_raw_mean'] = mean_raw
    df[f'cipn20_{prefix}']          = np.round(transformed, 1)
    return df

df = compute_subscale(df, sensory_idx,   'sensory')
df = compute_subscale(df, motor_idx,     'motor')
df = compute_subscale(df, autonomic_idx, 'autonomic')

subscale_cols = ['cipn20_sensory', 'cipn20_motor', 'cipn20_autonomic']
df['cipn20_num_present_subscales'] = df[subscale_cols].notna().sum(axis=1)
df['cipn20_total']                 = df[subscale_cols].mean(axis=1)
df.loc[df['cipn20_num_present_subscales'] < 2, 'cipn20_total'] = np.nan
df['cipn20_total']                 = df['cipn20_total'].round(1)

# IMPROVED: Reduced label noise (mislabel_rate = 0.02)
np.random.seed(999)
n_mis   = int(round(len(df) * mislabel_rate))
mis_idx = np.random.choice(df.index, size=n_mis, replace=False)
df.loc[mis_idx, 'label'] = 1 - df.loc[mis_idx, 'label']

print("Dataset generated.")
print("Label counts:\n", df['label'].value_counts())
print()

# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# Improvement #3: use ALL available features (demographics + all CIPN-20 items)
# =============================================================================

TARGET = 'label'
y      = df[TARGET].astype(int).values

# --- Encode categorical columns ---
le_cancer = LabelEncoder()
le_drug   = LabelEncoder()
le_sex    = LabelEncoder()
le_age_g  = LabelEncoder()

df['cancer_type_enc'] = le_cancer.fit_transform(df['cancer_type'])
df['drug_type_enc']   = le_drug.fit_transform(df['drug_type'])
df['sex_enc']         = le_sex.fit_transform(df['sex'])
df['age_group_enc']   = le_age_g.fit_transform(df['age_group'].astype(str))
df['drives_enc']      = df['drives'].astype(int)

# Quantitative sensory test features
qst_features = ['vpt', 'cold_temp']

# Subscale summary features
subscale_features = ['cipn20_sensory', 'cipn20_motor', 'cipn20_autonomic', 'cipn20_total']

# All individual CIPN-20 item scores (Improvement #3: these were NOT used before)
item_features = [f'cipn20_i{i}' for i in range(1, 21)]

# Demographic features (Improvement #3: these were NOT used before)
demographic_features = ['age', 'cancer_type_enc', 'drug_type_enc', 'sex_enc',
                        'age_group_enc', 'drives_enc']

ALL_FEATURES = qst_features + subscale_features + item_features + demographic_features

print(f"Total features used: {len(ALL_FEATURES)}")
print(f"  QST features:         {qst_features}")
print(f"  Subscale features:    {subscale_features}")
print(f"  Item features (20):   cipn20_i1 ... cipn20_i20")
print(f"  Demographic features: {demographic_features}")
print()

X = df[ALL_FEATURES].copy()

# Impute missing values
imputer  = SimpleImputer(strategy='median')
X_imp    = pd.DataFrame(imputer.fit_transform(X), columns=ALL_FEATURES)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.15, random_state=42, stratify=y
)

# Scale
scaler     = StandardScaler().fit(X_train)
X_train_s  = scaler.transform(X_train)
X_test_s   = scaler.transform(X_test)

print(f"Train samples: {len(X_train)}   Test samples: {len(X_test)}")
print()

# =============================================================================
# SECTION 3: TUNED RANDOM FOREST  (Improvement #5: GridSearchCV)
# =============================================================================

print("=" * 60)
print("RANDOM FOREST with GridSearchCV hyperparameter tuning")
print("=" * 60)

param_grid = {
    'n_estimators':      [200, 400],
    'max_depth':         [None, 15, 25],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2],
}

rf_base = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_s, y_train)

print(f"\nBest RF params: {grid_search.best_params_}")
print(f"Best CV AUC:    {grid_search.best_score_:.4f}")

best_rf  = grid_search.best_estimator_
y_prob_rf = best_rf.predict_proba(X_test_s)[:, 1]
y_pred_rf = (y_prob_rf >= 0.5).astype(int)

acc_rf   = accuracy_score(y_test, y_pred_rf)
auc_rf   = roc_auc_score(y_test, y_prob_rf)
prec_rf, recall_rf, f1_rf, _ = precision_recall_fscore_support(
    y_test, y_pred_rf, average='binary', zero_division=0
)
cm_rf    = confusion_matrix(y_test, y_pred_rf)

print(f"\n--- Tuned Random Forest Test Results ---")
print(f"Accuracy  : {acc_rf:.4f}")
print(f"AUC       : {auc_rf:.4f}")
print(f"Precision : {prec_rf:.4f}")
print(f"Recall    : {recall_rf:.4f}")
print(f"F1        : {f1_rf:.4f}")
print(f"Confusion matrix (tn, fp, fn, tp): {cm_rf.ravel().tolist()}")
print("\nClassification report:\n",
      classification_report(y_test, y_pred_rf, digits=4))

# Feature importance (top 15)
importances = pd.Series(best_rf.feature_importances_, index=ALL_FEATURES)
print("\nTop 15 most important features:")
print(importances.nlargest(15).to_string())
print()

# =============================================================================
# SECTION 4: IMPROVED KERAS MLP  (Improvement #4: deeper model + Dropout + BN)
# =============================================================================

print("=" * 60)
print("IMPROVED KERAS MLP  (deeper, Dropout, BatchNormalization)")
print("=" * 60)

# TF import inside try/except for environments that may not have it
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available — skipping MLP section.")

if TF_AVAILABLE:
    tf.random.set_seed(42)

    # IMPROVED model: 3 hidden layers, Dropout, BatchNormalization
    def make_improved_model(input_dim):
        model = keras.Sequential([
            keras.layers.InputLayer(shape=(input_dim,)),

            # Hidden layer 1
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),

            # Hidden layer 2
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            # Hidden layer 3  (was not present in original)
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.1),

            # Output
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=[keras.metrics.AUC(name='auc')]
        )
        return model

    model = make_improved_model(X_train_s.shape[1])
    model.summary()

    # Class weighting to handle imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg_count / pos_count}
    print(f"\nClass weights — 0: {class_weight[0]:.3f}, 1: {class_weight[1]:.3f}")

    # Early stopping to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max',
        patience=10, restore_best_weights=True, verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5,
        min_lr=1e-5, verbose=1
    )

    history = model.fit(
        X_train_s, y_train,
        validation_split=0.12,
        epochs=80,              # More epochs; early stopping will control this
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    keras_probs = model.predict(X_test_s).flatten()
    keras_preds = (keras_probs >= 0.5).astype(int)

    acc_mlp   = accuracy_score(y_test, keras_preds)
    auc_mlp   = roc_auc_score(y_test, keras_probs)
    prec_mlp, recall_mlp, f1_mlp, _ = precision_recall_fscore_support(
        y_test, keras_preds, average='binary', zero_division=0
    )
    cm_mlp    = confusion_matrix(y_test, keras_preds)

    print(f"\n--- Improved MLP Test Results ---")
    print(f"Accuracy  : {acc_mlp:.4f}")
    print(f"AUC       : {auc_mlp:.4f}")
    print(f"Precision : {prec_mlp:.4f}")
    print(f"Recall    : {recall_mlp:.4f}")
    print(f"F1        : {f1_mlp:.4f}")
    print(f"Confusion matrix (tn, fp, fn, tp): {cm_mlp.ravel().tolist()}")
    print("\nClassification report:\n",
          classification_report(y_test, keras_preds, digits=4))

# =============================================================================
# SECTION 5: ROC CURVE COMPARISON PLOT
# =============================================================================

plt.figure(figsize=(8, 6))

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc_fn(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, lw=2,
         label=f'Tuned Random Forest (AUC = {roc_auc_rf:.3f})')

# MLP ROC
if TF_AVAILABLE:
    fpr_mlp, tpr_mlp, _ = roc_curve(y_test, keras_probs)
    roc_auc_mlp = auc_fn(fpr_mlp, tpr_mlp)
    plt.plot(fpr_mlp, tpr_mlp, lw=2, linestyle='--',
             label=f'Improved MLP (AUC = {roc_auc_mlp:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random chance')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison — Improved CIPN Models', fontsize=13)
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150)
print("\nSaved ROC comparison plot: roc_comparison.png")
plt.show()

# =============================================================================
# SECTION 6: SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY: Original vs Improved")
print("=" * 60)
print(f"{'Metric':<20} {'Original RF':>15} {'Tuned RF':>12}", end="")
if TF_AVAILABLE:
    print(f" {'Improved MLP':>14}")
else:
    print()

print("-" * 60)
print(f"{'Accuracy':<20} {'~71.5%':>15} {acc_rf*100:>11.1f}%", end="")
if TF_AVAILABLE:
    print(f" {acc_mlp*100:>13.1f}%")
else:
    print()

print(f"{'AUC':<20} {'~0.74':>15} {auc_rf:>12.4f}", end="")
if TF_AVAILABLE:
    print(f" {auc_mlp:>14.4f}")
else:
    print()

print(f"{'Recall (CIPN+)':<20} {'~52.9%':>15} {recall_rf*100:>11.1f}%", end="")
if TF_AVAILABLE:
    print(f" {recall_mlp*100:>13.1f}%")
else:
    print()

print(f"{'F1 score':<20} {'~0.573':>15} {f1_rf:>12.4f}", end="")
if TF_AVAILABLE:
    print(f" {f1_mlp:>14.4f}")
else:
    print()

print("=" * 60)
print()
print("Key changes made vs original notebook:")
print("  #2 - Noise reduced: sd_multiplier 1.5→1.0, apply_prob 0.80→0.92,")
print("       mislabel_rate 0.05→0.02")
print("  #3 - Features expanded: 5 → 32 (added demographics + all 20 CIPN items)")
print("  #4 - MLP improved: 1 hidden layer(16) → 3 layers(64→32→16)")
print("       + BatchNormalization + Dropout + EarlyStopping + class weights")
print("  #5 - RF improved: GridSearchCV over n_estimators, max_depth,")
print("       min_samples_split, min_samples_leaf")
