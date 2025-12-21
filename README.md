# 📱 Smart Phone Price Prediction (Classification)

An end-to-end machine learning project for classifying smartphones into **Expensive** and **Non-Expensive** categories based on technical specifications and market-related features.

This project focuses on **robust evaluation metrics** (AUC-ROC & Average Precision) rather than accuracy alone and follows a complete ML pipeline from data preprocessing to model persistence.

---

## 👥 Team

**Team Name:** Bakary Gasama  

- Ali Mohamed Gaber  
- Mostafa Alaa  
- Ali Mohamed Mohamed  
- Youssef Eldiasty  
- Aly Mahmoud
- Mohamed Sherif  

---

## 📊 Dataset Overview

- Each record represents a smartphone model
- Features include:
  - Processor details
  - RAM & storage
  - Camera specifications
  - Battery & display
  - Connectivity (4G, 5G, NFC)
  - OS & brand information
- Target variable: **Price Category (Expensive / Non-Expensive)**

**Dataset Size:**
- Train: 863 rows × 32 columns  
- Test: 153 rows × 32 columns  

---

## 🧹 Data Cleaning & Preprocessing

Key preprocessing steps included:

- Processor series standardization
- Invalid RAM & storage value correction/removal
- RAM tier and notch type imputation
- Camera count logical corrections
- Brand name normalization
- Feature transformation (OS version, memory card size)
- Binary & ordinal encoding
- Target encoding for high-cardinality brand feature
- Correlation-based feature selection

📌 **Example: Correlation Heatmap (Before Feature Selection)**  
<img width="1976" height="1698" alt="image" src="https://github.com/user-attachments/assets/c6abaa3c-9ee5-4f90-8d0f-8da7766fb137" />


---

## 📈 Exploratory Data Analysis (EDA)

EDA was conducted to understand feature relationships with the price category.

Key insights:
- Premium processors (Snapdragon, Bionic, Tensor) → Expensive
- 5G & NFC → Strong indicators of high price
- Higher RAM & storage → Expensive cluster
- iOS devices → Exclusively expensive

📊 **Sample Visualizations**

![RAM vs Storage](images/ram_storage_pairplot.png)  
![Price by Brand](images/price_by_brand.png)  
![Processor vs Price](images/processor_price_countplot.png)

---

## ⚖️ Handling Class Imbalance (SMOTE)

To address class imbalance, **SMOTE** was applied **only to the training set** to avoid data leakage.

📊 **Class Distribution Before SMOTE**  
![Before SMOTE](images/class_distribution_before_smote.png)

📊 **Class Distribution After SMOTE**  
![After SMOTE](images/class_distribution_after_smote.png)

---

## 🤖 Models Trained

The following classification models were implemented and optimized:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- Voting Classifier (Ensemble)

---

## 🔧 Hyperparameter Tuning

- GridSearchCV used for hyperparameter optimization
- Cross-validation applied during tuning
- Best parameters used for final training

📸 **Example: Hyperparameter Search Output**  
![Grid Search](images/hyperparameter_tuning.png)

---

## ⏱️ Model Training Time Analysis

Training duration was recorded for selected models to compare computational cost vs performance.

📊 **Training Time Comparison**  
![Training Time](images/model_training_time.png)

---

## 📊 Evaluation Strategy

Instead of accuracy alone, we focused on **threshold-independent metrics**:

- **ROC–AUC**
- **Average Precision (AP)**

These metrics provide better insight, especially under class imbalance.

---

## 📈 ROC–AUC Curves

### Combined ROC–AUC Curve
![Combined ROC](images/roc_auc_all_models.png)

### Individual ROC–AUC Curves
![LR ROC](images/roc_logistic.png)  
![RF ROC](images/roc_random_forest.png)  
![SVM ROC](images/roc_svm.png)

---

## 📉 Average Precision (AP) Curves

### Combined AP Curve
![Combined AP](images/ap_all_models.png)

### Individual AP Curves
![GB AP](images/ap_gradient_boosting.png)  
![Voting AP](images/ap_voting_classifier.png)

---

## 🌲 Feature Importance Analysis

Feature importance was extracted for tree-based models to improve interpretability.

![Random Forest Feature Importance](images/feature_importance_rf.png)  
![Gradient Boosting Feature Importance](images/feature_importance_gb.png)

---

## 🧪 K-Fold Cross Validation

- K-Fold CV applied to validate model stability
- Consistent performance across folds observed

📊 **Cross-Validation Scores**  
![KFold](images/kfold_scores.png)

---

## 🗳️ Voting Classifier

An ensemble Voting Classifier was implemented to combine the strengths of top-performing models.

📸 **Voting Classifier Results**  
![Voting CM](images/voting_confusion_matrix.png)

---

## 💾 Model Persistence

- Trained models were saved using serialization
- Enables reuse without retraining
- Ready for future deployment

📸 **Model Saving Output**  
![Model Save](images/model_saving.png)

---

## 🏁 Final Model Selection

Models were compared based on:
- AUC-ROC
- Average Precision
- Precision, Recall, F1-score
- Training time
- Cross-validation stability

The final selected model demonstrated strong generalization and robust class separation.

---

## ✅ Conclusion

This project demonstrates a complete and production-ready machine learning workflow for smartphone price classification.  
By focusing on **AUC-ROC and Average Precision**, the evaluation reflects real-world model reliability rather than surface-level accuracy.

---

## 🚀 Future Work

- Multi-class price prediction
- Larger and newer datasets
- Automated hyperparameter optimization
- Web or API-based deployment

---

## 📎 Repository Structure

```text
├── data/
├── notebooks/
├── models/
├── images/
├── README.md
