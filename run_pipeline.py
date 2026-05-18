import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import random

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.manifold import TSNE

try:
    import mord
    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False
    print("Warning: 'mord' library not found. Ordinal Ridge will be skipped.")

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(42)
    
    # =====================================================================
    # TASK 1 — DATA PREPARATION
    # =====================================================================
    print("TASK 1: Data Preparation...")
    csv_path = "BABE_features_normalized.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Drop "No agreement"
    df = df[df['label_bias'] != 'No agreement'].copy()
    
    # Encode label_bias
    df['y_cls'] = df['label_bias'].map({'Biased': 1, 'Non-biased': 0})
    
    # Keep target_score
    df['y_reg'] = df['target_score']
    
    # Drop rows where target or text is missing
    df = df.dropna(subset=['y_cls', 'y_reg', 'text']).reset_index(drop=True)
    
    y_cls = df['y_cls'].values.astype(int)
    y_reg = df['y_reg'].values.astype(float)
    texts = df['text'].tolist()
    
    HANDCRAFTED_COLS = [
        'char_len', 'word_count', 'avg_word_len', 'punct_count', 'caps_word_count',
        'biased_word_count', 'biased_word_ratio', 'biased_word_present',
        'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu', 'subjectivity',
        'polarity', 'intensifier_count', 'hedge_count', 'negative_word_count',
        'is_question', 'is_exclamation', 'entity_count', 'has_person', 'has_org',
        'has_place', 'person_count', 'adj_count', 'adj_ratio', 'verb_count', 'noun_count'
    ]
    X_hand_full = df[HANDCRAFTED_COLS].values.astype(float)
    
    indices = np.arange(len(df))
    idx_train, idx_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        indices, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    # =====================================================================
    # TASK 2 — TEXT EMBEDDINGS
    # =====================================================================
    print("TASK 2: Text Embeddings...")
    embed_file = "babe_embeddings.npy"
    if os.path.exists(embed_file):
        print(f"Loading cached embeddings from {embed_file}")
        embeddings = np.load(embed_file)
    else:
        print("Computing embeddings with all-mpnet-base-v2...")
        model = SentenceTransformer('all-mpnet-base-v2')
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        np.save(embed_file, embeddings)
        print(f"Saved embeddings to {embed_file}")
    
    # Engineer interaction features
    # 'bias_x_sentiment' = biased_word_ratio * vader_compound
    bias_x_sentiment = (df['biased_word_ratio'] * df['vader_compound']).values.reshape(-1, 1)
    # 'subj_x_polarity' = subjectivity * polarity
    subj_x_polarity = (df['subjectivity'] * df['polarity']).values.reshape(-1, 1)
    # 'caps_ratio' = caps_word_count / (word_count + 1e-5)
    caps_ratio = (df['caps_word_count'] / (df['word_count'] + 1e-5)).values.reshape(-1, 1)
    
    interactions = np.hstack([bias_x_sentiment, subj_x_polarity, caps_ratio])
    
    # Feature matrices for all data
    X_emb_full = embeddings
    X_combined_full = np.hstack([embeddings, X_hand_full, interactions])
    
    # Train/Test splits for all feature sets
    feature_sets = {
        'Emb': (X_emb_full[idx_train], X_emb_full[idx_test]),
        'Hand': (X_hand_full[idx_train], X_hand_full[idx_test]),
        'Combined': (X_combined_full[idx_train], X_combined_full[idx_test])
    }
    
    # Prepare sample weights for regression
    sample_weights_train = compute_sample_weight('balanced', np.round(y_reg_train).astype(int))
    
    # Data structures for results
    cls_results = []
    reg_results = []
    summary_data = []
    
    # Store best models for plotting
    best_cls_model = None
    best_cls_f1 = -1
    best_cls_name = ""
    best_reg_model = None
    best_reg_rmse = float('inf')
    best_reg_name = ""
    
    # =====================================================================
    # TASK 3 — CLASSIFICATION
    # =====================================================================
    print("\nTASK 3: Classification...")
    cls_models = {
        "LR": LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             n_estimators=300, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=42),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42),
        "SVM": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=5)
    }
    
    # Grid search for LR
    lr_params = {'C': [0.01, 0.1, 1, 10]}
    
    for fs_name, (X_tr, X_te) in feature_sets.items():
        for m_name, model in cls_models.items():
            print(f"  Training {m_name} on {fs_name}...")
            
            if m_name == "LR":
                clf = GridSearchCV(model, lr_params, cv=StratifiedKFold(n_splits=5), scoring='f1_macro', n_jobs=-1)
                clf.fit(X_tr, y_cls_train)
                best_model = clf.best_estimator_
            else:
                best_model = model
                best_model.fit(X_tr, y_cls_train)
            
            y_pred = best_model.predict(X_te)
            y_prob = best_model.predict_proba(X_te)[:, 1] if hasattr(best_model, "predict_proba") else y_pred
            
            acc = accuracy_score(y_cls_test, y_pred)
            f1 = f1_score(y_cls_test, y_pred, average='macro')
            try:
                auc_score = roc_auc_score(y_cls_test, y_prob)
            except:
                auc_score = np.nan
            mcc = matthews_corrcoef(y_cls_test, y_pred)
            
            cls_results.append({
                'Model': m_name, 'Feature Set': fs_name,
                'Acc': acc, 'F1': f1, 'AUC': auc_score, 'MCC': mcc
            })
            
            summary_data.append([m_name, fs_name, "Classification", "F1 Macro", f1])
            
            if fs_name == 'Combined' and f1 > best_cls_f1:
                best_cls_f1 = f1
                best_cls_model = best_model
                best_cls_name = m_name
                
    # =====================================================================
    # TASK 4 — REGRESSION
    # =====================================================================
    print("\nTASK 4: Regression...")
    reg_models = {
        "XGB": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0, random_state=42),
        "RF": RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42)
    }
    if MORD_AVAILABLE:
        reg_models["Ordinal"] = mord.OrdinalRidge(alpha=1.0)
        
    for fs_name, (X_tr, X_te) in feature_sets.items():
        for m_name, model in reg_models.items():
            print(f"  Training {m_name} on {fs_name}...")
            
            if m_name == "Ordinal":
                model.fit(X_tr, np.round(y_reg_train).astype(int))
            else:
                model.fit(X_tr, y_reg_train, sample_weight=sample_weights_train)
                
            y_pred = model.predict(X_te)
            
            rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
            mae = mean_absolute_error(y_reg_test, y_pred)
            r2 = r2_score(y_reg_test, y_pred)
            
            reg_results.append({
                'Model': m_name, 'Feature Set': fs_name,
                'RMSE': rmse, 'MAE': mae, 'R2': r2
            })
            
            summary_data.append([m_name, fs_name, "Regression", "RMSE", rmse])
            
            if fs_name == 'Combined' and rmse < best_reg_rmse:
                best_reg_rmse = rmse
                best_reg_model = model
                best_reg_name = m_name

    # =====================================================================
    # FINAL SUMMARY TABLE
    # =====================================================================
    summary_df = pd.DataFrame(summary_data, columns=["Model", "Feature Set", "Task", "Key Metric", "Value"])
    cls_summary = summary_df[summary_df["Task"] == "Classification"].sort_values(by="Value", ascending=False)
    reg_summary = summary_df[summary_df["Task"] == "Regression"].sort_values(by="Value", ascending=True)
    
    final_summary = pd.concat([cls_summary, reg_summary])
    final_summary.to_csv("babe_results_summary.csv", index=False)
    
    print("\n" + "="*55)
    print("FINAL SUMMARY TABLE")
    print("="*55)
    print(tabulate(final_summary, headers='keys', tablefmt='psql', showindex=False))

    # =====================================================================
    # TASK 5 — PLOTS
    # =====================================================================
    print("\nTASK 5: Generating Plots...")
    
    # PLOT 1 — Classification Results Heatmap
    cls_df = pd.DataFrame(cls_results)
    f1_pivot = cls_df.pivot(index='Model', columns='Feature Set', values='F1')
    annot_df = pd.DataFrame(index=f1_pivot.index, columns=f1_pivot.columns)
    for idx, row in cls_df.iterrows():
        annot_df.loc[row['Model'], row['Feature Set']] = f"Acc: {row['Acc']:.3f}\nF1: {row['F1']:.3f}\nAUC: {row['AUC']:.3f}\nMCC: {row['MCC']:.3f}"
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(f1_pivot, annot=annot_df, fmt="", cmap='YlGn', cbar_kws={'label': 'F1 Macro'})
    plt.title("Classification Performance by Model and Feature Set")
    plt.tight_layout()
    plt.savefig("plot1_cls_heatmap.png", dpi=150)
    plt.close()
    
    # PLOT 2 — Regression Results Heatmap
    reg_df = pd.DataFrame(reg_results)
    rmse_pivot = reg_df.pivot(index='Model', columns='Feature Set', values='RMSE')
    annot_reg_df = pd.DataFrame(index=rmse_pivot.index, columns=rmse_pivot.columns)
    for idx, row in reg_df.iterrows():
        annot_reg_df.loc[row['Model'], row['Feature Set']] = f"RMSE: {row['RMSE']:.3f}\nMAE: {row['MAE']:.3f}\nR2: {row['R2']:.3f}"
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(rmse_pivot, annot=annot_reg_df, fmt="", cmap='YlGn_r', cbar_kws={'label': 'RMSE'})
    plt.title("Regression Performance by Model and Feature Set")
    plt.tight_layout()
    plt.savefig("plot2_reg_heatmap.png", dpi=150)
    plt.close()

    # PLOT 3 — Best Classifier: Confusion Matrix + ROC Curve
    if best_cls_model is not None:
        X_te_combined = feature_sets['Combined'][1]
        y_pred = best_cls_model.predict(X_te_combined)
        y_prob = best_cls_model.predict_proba(X_te_combined)[:, 1] if hasattr(best_cls_model, "predict_proba") else y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        cm = confusion_matrix(y_cls_test, y_pred, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0])
        axes[0].set_title("Normalized Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_xticklabels(['Non-biased', 'Biased'])
        axes[0].set_yticklabels(['Non-biased', 'Biased'])
        
        fpr, tpr, _ = roc_curve(y_cls_test, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Receiver Operating Characteristic')
        axes[1].legend(loc="lower right")
        
        plt.suptitle(f"Best Classifier — {best_cls_name} on Combined Features", fontsize=16)
        plt.tight_layout()
        plt.savefig("plot3_best_cls.png", dpi=150)
        plt.close()

    # PLOT 4 — Best Regressor: Actual vs Predicted Scatter
    if best_reg_model is not None:
        X_te_combined = feature_sets['Combined'][1]
        y_pred = best_reg_model.predict(X_te_combined)
        
        residuals = np.abs(y_reg_test - y_pred)
        jitter_x = np.random.normal(0, 0.08, size=len(y_reg_test))
        jitter_y = np.random.normal(0, 0.08, size=len(y_pred))
        
        plt.figure(figsize=(9, 7))
        sc = plt.scatter(y_reg_test + jitter_x, y_pred + jitter_y, c=residuals, cmap='coolwarm', alpha=0.7)
        plt.colorbar(sc, label="Absolute Error")
        
        min_val = min(y_reg_test.min(), y_pred.min())
        max_val = max(y_reg_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        # Metrics text box
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
        mae = mean_absolute_error(y_reg_test, y_pred)
        r2 = r2_score(y_reg_test, y_pred)
        textstr = f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}"
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel("Actual Score (with jitter)")
        plt.ylabel("Predicted Score (with jitter)")
        plt.title(f"Best Regressor — {best_reg_name} on Combined Features")
        plt.tight_layout()
        plt.savefig("plot4_best_reg.png", dpi=150)
        plt.close()

    # PLOT 5 — Feature Importance
    if "XGB" in cls_models and "XGB" in reg_models:
        xgb_cls = cls_models["XGB"]
        xgb_reg = reg_models["XGB"]
        if hasattr(xgb_cls, "feature_importances_") and hasattr(xgb_reg, "feature_importances_"):
            # Feature names
            feature_names = [f"emb_{i}" for i in range(768)] + HANDCRAFTED_COLS + ['bias_x_sentiment', 'subj_x_polarity', 'caps_ratio']
            
            def get_agg_importances(model):
                imp = model.feature_importances_
                emb_imp = imp[:768].mean()
                other_imp = imp[768:]
                names = ["Embeddings (avg)"] + feature_names[768:]
                values = [emb_imp] + list(other_imp)
                
                df_imp = pd.DataFrame({'Feature': names, 'Importance': values})
                return df_imp.sort_values(by='Importance', ascending=False).head(20)
            
            top20_cls = get_agg_importances(xgb_cls)
            top20_reg = get_agg_importances(xgb_reg)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            sns.barplot(data=top20_cls, x='Importance', y='Feature', ax=axes[0], color='skyblue')
            axes[0].set_title("XGBoost Classifier - Top 20 Features")
            
            sns.barplot(data=top20_reg, x='Importance', y='Feature', ax=axes[1], color='lightgreen')
            axes[1].set_title("XGBoost Regressor - Top 20 Features")
            
            plt.suptitle("Top 20 Feature Importances — XGBoost Classifier vs Regressor", fontsize=16)
            plt.tight_layout()
            plt.savefig("plot5_feature_importance.png", dpi=150)
            plt.close()

    # PLOT 6 — Embedding Space Visualization (t-SNE)
    print("Running t-SNE for plot 6...")
    X_emb_te = feature_sets['Emb'][1]
    tsne = TSNE(perplexity=30, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_emb_te)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_cls_test, cmap='coolwarm', alpha=0.5)
    plt.title("t-SNE of Sentence Embeddings — Colored by Bias Label")
    
    handles, labels = scatter.legend_elements()
    plt.legend(handles, ['Non-biased', 'Biased'], loc="best")
    plt.text(0.02, 0.02, "t-SNE is for visualization only", transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("plot6_tsne.png", dpi=150)
    plt.close()
    
    print("Pipeline completed successfully. Output files generated.")

if __name__ == '__main__':
    main()