import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight

def get_rf_combined_mae():
    # 1. Load the data
    df = pd.read_csv("BABE_features_normalized.csv")
    df = df[df['label_bias'] != 'No agreement'].copy()
    
    # 2. Extract target
    df['y_cls'] = df['label_bias'].map({'Biased': 1, 'Non-biased': 0})
    df['y_reg'] = df['target_score']
    df = df.dropna(subset=['y_cls', 'y_reg', 'text']).reset_index(drop=True)
    
    y_cls = df['y_cls'].values.astype(int)
    y_reg = df['y_reg'].values.astype(float)
    
    # 3. Handcrafted features
    HANDCRAFTED_COLS = [
        'char_len', 'word_count', 'avg_word_len', 'punct_count', 'caps_word_count',
        'biased_word_count', 'biased_word_ratio', 'biased_word_present',
        'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu', 'subjectivity',
        'polarity', 'intensifier_count', 'hedge_count', 'negative_word_count',
        'is_question', 'is_exclamation', 'entity_count', 'has_person', 'has_org',
        'has_place', 'person_count', 'adj_count', 'adj_ratio', 'verb_count', 'noun_count'
    ]
    X_hand_full = df[HANDCRAFTED_COLS].values.astype(float)
    
    # 4. Load cached embeddings
    embed_file = "babe_embeddings.npy"
    if not os.path.exists(embed_file):
        print("Embeddings not found. Please run run_pipeline.py first.")
        return
    embeddings = np.load(embed_file)
    
    # 5. Interaction features
    bias_x_sentiment = (df['biased_word_ratio'] * df['vader_compound']).values.reshape(-1, 1)
    subj_x_polarity = (df['subjectivity'] * df['polarity']).values.reshape(-1, 1)
    caps_ratio = (df['caps_word_count'] / (df['word_count'] + 1e-5)).values.reshape(-1, 1)
    interactions = np.hstack([bias_x_sentiment, subj_x_polarity, caps_ratio])
    
    # 6. Build the 'Combined' (Embeddings + Features) matrix (799 dimensions)
    X_combined_full = np.hstack([embeddings, X_hand_full, interactions])
    
    # 7. Stratified Train/Test split using the same logic as run_pipeline.py
    indices = np.arange(len(df))
    idx_train, idx_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        indices, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    X_train = X_combined_full[idx_train]
    X_test = X_combined_full[idx_test]
    
    # 8. Compute sample weights for regression
    sample_weights_train = compute_sample_weight('balanced', np.round(y_reg_train).astype(int))
    
    # 9. Initialize and Train Random Forest Regressor
    print("Training RF on Embeddings + Features... (This may take a minute)", flush=True)
    model = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_reg_train, sample_weight=sample_weights_train)
    
    # 10. Predict and evaluate MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_pred)
    
    print(f"\n--- RESULTS ---")
    print(f"Random Forest Regressor MAE (Embeddings + Features): {mae:.4f}")

if __name__ == "__main__":
    get_rf_combined_mae()
