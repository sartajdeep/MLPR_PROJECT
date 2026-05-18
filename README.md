# Explanation of the Machine Learning Pipeline: End-to-End Walkthrough

This document provides a comprehensive, end-to-end explanation of the machine learning pipeline generated for your dataset (`BABE_features_normalized.csv`). It meticulously breaks down exactly what features were used as inputs, what target values the models were trying to predict, and how each model was trained.

---

## 1. THE TARGET VALUES (What are the models predicting?)
The raw dataset contains annotations for media bias. For this pipeline, we have two distinct predictive tasks, which means we have two separate "target" values (the `y` variables in machine learning):

1. **Classification Target (`y_cls`): Predicting if an article is biased.**
   - Derived from the raw `label_bias` column.
   - We dropped any rows where the annotators could not agree (where `label_bias == 'No agreement'`).
   - We mapped the remaining text labels into binary integers: **'Biased' becomes 1**, and **'Non-biased' becomes 0**.
   - **All Classifiers** (Logistic Regression, XGBoost Classifier, Random Forest Classifier, SVM) are trained exclusively against this `0` or `1` value.

2. **Regression Target (`y_reg`): Predicting the severity/score of the bias.**
   - Derived from the raw `target_score` column.
   - This is a continuous/ordinal value ranging from 1 to 10.
   - **All Regressors** (XGBoost Regressor, Random Forest Regressor, Ordinal Ridge) are trained exclusively against this 1–10 numeric score.

---

## 2. THE INPUT FEATURES (What data do the models use to learn?)
To predict the target values above, the models need input data (the `X` variables). The script extracts, generates, and engineers three distinct types of features from the raw dataset:

### A. Handcrafted Features (28 Dimensions)
These are statistical and linguistic properties of the headline text that were already pre-calculated and normalized (0–1) in your CSV. All 28 were extracted:
1. `char_len` (Length of the headline in characters)
2. `word_count` (Number of words)
3. `avg_word_len` (Average length of words)
4. `punct_count` (Amount of punctuation)
5. `caps_word_count` (Number of capitalized words)
6. `biased_word_count` (Count of known biased words)
7. `biased_word_ratio` (Ratio of biased words to total words)
8. `biased_word_present` (Binary indicator if a biased word exists)
9. `vader_compound` (VADER sentiment compound score)
10. `vader_pos` (VADER positive sentiment score)
11. `vader_neg` (VADER negative sentiment score)
12. `vader_neu` (VADER neutral sentiment score)
13. `subjectivity` (TextBlob subjectivity score)
14. `polarity` (TextBlob polarity score)
15. `intensifier_count` (Number of intensifier words, e.g., "very", "extremely")
16. `hedge_count` (Number of hedging words, e.g., "might", "possibly")
17. `negative_word_count` (Count of negative words)
18. `is_question` (Binary indicator if headline is a question)
19. `is_exclamation` (Binary indicator if headline is an exclamation)
20. `entity_count` (Total number of Named Entities)
21. `has_person` (Binary indicator if a Person entity is mentioned)
22. `has_org` (Binary indicator if an Organization entity is mentioned)
23. `has_place` (Binary indicator if a Location/Place entity is mentioned)
24. `person_count` (Number of Person entities)
25. `adj_count` (Number of adjectives)
26. `adj_ratio` (Ratio of adjectives to total words)
27. `verb_count` (Number of verbs)
28. `noun_count` (Number of nouns)

### B. Text Embeddings (768 Dimensions)
Instead of just relying on the handcrafted linguistic stats, the script uses the `sentence-transformers` library (specifically the `all-mpnet-base-v2` model).
- It takes the raw text string from the `text` column.
- It passes it through the deep learning transformer model to generate a **768-dimensional dense vector** representing the semantic meaning of the sentence.
- These 768 numbers are cached into `babe_embeddings.npy` to speed up future runs.

### C. Engineered Interaction Features (3 Dimensions)
To capture complex relationships between the handcrafted features, the script multiplies/divides specific columns to create 3 brand new features:
1. `bias_x_sentiment`: `biased_word_ratio` × `vader_compound`
2. `subj_x_polarity`: `subjectivity` × `polarity`
3. `caps_ratio`: `caps_word_count` / (`word_count` + 1e-5) (the small number prevents division by zero).

---

## 3. FEATURE SET COMBINATIONS
To see which data representation works best, the pipeline groups the above features into three distinct Feature Sets (the `X` matrices):

1. **`Emb` Feature Set (768 columns):** Only uses the Text Embeddings.
2. **`Hand` Feature Set (28 columns):** Only uses the Handcrafted Features.
3. **`Combined` Feature Set (799 columns):** Stacks everything together horizontally (768 Embeddings + 28 Handcrafted + 3 Interactions).

---

## 4. END-TO-END TRAINING WORKFLOW
Here is the exact timeline of how the data flows through the models:

### Step 1: Data Splitting
Before any training happens, the script takes all the rows, the 3 Feature Sets (`X`), and the 2 targets (`y_cls`, `y_reg`) and splits them into:
- **Training Set (80%):** Used to teach the models.
- **Testing Set (20%):** Hidden from the models during training. Used strictly to evaluate performance.
- *Note: A stratified split is used to ensure the 80/20 ratio maintains the exact same proportion of "Biased" vs "Non-biased" headlines as the original dataset.*

### Step 2: Classification (Predicting `y_cls`: 0 or 1)
The script trains 4 different classifiers. **Every single classifier is trained 3 separate times** (once for `Emb`, once for `Hand`, once for `Combined`), resulting in 12 total models.
- **Logistic Regression:** Uses a GridSearch to find the best `C` parameter (0.01, 0.1, 1, or 10) using 5-fold cross-validation.
- **XGBoost Classifier:** Trained with 300 decision trees, learning rate 0.05, and max tree depth of 6.
- **Random Forest Classifier:** Trained with 300 decision trees, requiring at least 2 samples per leaf.
- **Support Vector Machine (SVM):** A `LinearSVC` wrapped in a probability calibrator to output confidence percentages.

**Evaluation:** After training on the 80%, each model is tested on the 20% holdout set. The script records Accuracy, F1 Macro score, ROC-AUC, and the Matthews Correlation Coefficient (MCC).

### Step 3: Regression (Predicting `y_reg`: 1 through 10)
Because the extreme scores (like 1 or 10) are rarer than the middle scores (like 5), the script calculates **Sample Weights**. This tells the regressors to pay extra attention when they get an extreme score wrong.

The script trains 3 regressors. Again, every regressor is trained 3 separate times (once for each Feature Set), resulting in 9 total models.
- **XGBoost Regressor:** 500 decision trees, max depth 6, with L1/L2 mathematical regularization to prevent overfitting.
- **Random Forest Regressor:** 300 trees, max depth 12.
- **Ordinal Ridge:** (From the `mord` library). Specifically designed to predict ranked ordinal numbers rather than pure continuous decimals.

**Evaluation:** Tested on the 20% holdout set. The script records Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (variance explained).

### Step 4: Outputs and Plots
Once all 21 models (12 classifiers + 9 regressors) are trained and evaluated, the script outputs the results:
- **`babe_results_summary.csv`**: A spreadsheet of all the metrics.
- **`plot1_cls_heatmap.png` & `plot2_reg_heatmap.png`**: Visual grids comparing the models against the 3 feature sets.
- **`plot3_best_cls.png`**: Shows the Confusion Matrix and ROC Curve for whichever classifier got the highest F1 score on the `Combined` 799-feature dataset.
- **`plot4_best_reg.png`**: Shows an Actual vs Predicted scatterplot (with added visual jitter and color-coded error mapping) for whichever regressor got the lowest RMSE on the `Combined` dataset.
- **`plot5_feature_importance.png`**: Looks inside the XGBoost models trained on the `Combined` dataset to see which of the 799 features it relied on the most (averaging the 768 embeddings into a single bar so the handcrafted features are still visible).
- **`plot6_tsne.png`**: Mathematically squashes the 768-dimensional embeddings down to a 2D graph so human eyes can see if the "Biased" and "Non-biased" sentences naturally cluster apart based on semantics alone.
