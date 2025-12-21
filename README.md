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

![RAM vs Storage] <img width="326" height="248" alt="image" src="https://github.com/user-attachments/assets/ef20bac5-5e0b-49c2-a131-2d6013f77cde" />
![Price by Brand] <img width="940" height="286" alt="image" src="https://github.com/user-attachments/assets/99b6d507-edfe-4cb3-b45a-3ec2e672fae1" />
![Processor vs Price] <img width="428" height="208" alt="image" src="https://github.com/user-attachments/assets/ff0c73ab-c7a7-4d74-bc39-4a7f24cd7b92" />

---

## ⚖️ Handling Class Imbalance (SMOTE)

To address class imbalance, **SMOTE** was applied **only to the training set** to avoid data leakage.

📊 **Class Distribution Before SMOTE**  
![Before SMOTE] <img width="975" height="218" alt="image" src="https://github.com/user-attachments/assets/8a341258-491d-4758-b8dc-102401f0261b" />

📊 **Class Distribution After SMOTE**  
![After SMOTE] <img width="627" height="364" alt="image" src="https://github.com/user-attachments/assets/01c0aa50-9690-4928-96fc-f5a478e800b4" />

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
![Grid Search] <img width="975" height="914" alt="image" src="https://github.com/user-attachments/assets/167de21e-712f-4970-b551-9f103aa008a4" />
<img width="975" height="750" alt="image" src="https://github.com/user-attachments/assets/2ad1fb62-9793-4765-a4cc-198b4ec88c50" />

---

## ⏱️ Model Training Time Analysis

Training duration was recorded for selected models to compare computational cost vs performance.

📊 **Training Time Comparison**  
![Training Time] <img width="975" height="456" alt="image" src="https://github.com/user-attachments/assets/7606c1ba-a246-4ee9-a707-a4ef5bc0c4f2" />

---

## 📊 Evaluation Strategy

Instead of accuracy alone, we focused on **threshold-independent metrics**:

- **ROC–AUC**
- **Average Precision (AP)**

These metrics provide better insight, especially under class imbalance.

---

## 📈 ROC–AUC Curves

### Combined ROC–AUC Curve
![Combined ROC] <img width="764" height="496" alt="image" src="https://github.com/user-attachments/assets/06c31fe0-56ad-41a8-9ae3-5c24d2e48afa" />

### Individual ROC–AUC Curves
![RF ROC]  <img width="680" height="595" alt="image" src="https://github.com/user-attachments/assets/4889d919-d5ea-4828-9851-4a8d4844b327" />
![SVM ROC] <img width="682" height="597" alt="image" src="https://github.com/user-attachments/assets/febcc839-c034-4b20-92da-8349fec49127" />

---

## 📉 Average Precision (AP) Curves

### Combined AP Curve
![Combined AP] <img width="975" height="772" alt="image" src="https://github.com/user-attachments/assets/3eb104e2-7311-4cbc-ae26-82d1d6161fb8" />

### Individual AP Curves
![RF AP] <img width="975" height="772" alt="image" src="https://github.com/user-attachments/assets/d975f7be-118a-4d6b-bad7-2607ed23bb16" />

---

## 🌲 Feature Importance Analysis

Feature importance was extracted for tree-based models to improve interpretability.

![Random Forest Feature Importance] <img width="905" height="1034" alt="image" src="https://github.com/user-attachments/assets/24c2d615-37b9-4f85-918b-fc53e9c58320" />
![Gradient Boosting Feature Importance] <img width="975" height="592" alt="image" src="https://github.com/user-attachments/assets/70bafd74-4f60-430d-9a83-6b002d7c319f" />

---

## 🧪 K-Fold Cross Validation

- K-Fold CV applied to validate model stability
- Consistent performance across folds observed

📊 **Cross-Validation Scores**  
![KFold] <img width="975" height="192" alt="image" src="https://github.com/user-attachments/assets/ed5bef15-bf6f-444f-bd61-64d8b8a8cef2" />

---

## 🗳️ Voting Classifier

An ensemble Voting Classifier was implemented to combine the strengths of top-performing models.

📸 **Voting Classifier Results**  
![Voting CM] <img width="466" height="488" alt="image" src="https://github.com/user-attachments/assets/318c6707-42fe-418b-be91-e4324b56b4e0" />

---

## 💾 Model Persistence

- Trained models were saved using serialization
- Enables reuse without retraining
- Ready for future deployment

📸 **Model Saving Output**  
![Model Save] <img width="975" height="416" alt="image" src="https://github.com/user-attachments/assets/3b0c031a-152d-426d-97e9-fbe09a654001" />

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

📸 **Model Saving Output**

<img width="975" height="439" alt="image" src="https://github.com/user-attachments/assets/1f605d08-9076-4afd-ab61-81f22bb0166f" />
<img width="975" height="266" alt="image" src="https://github.com/user-attachments/assets/aa989087-5b6a-4516-92ae-bf07c31443c4" />
<img width="975" height="298" alt="image" src="https://github.com/user-attachments/assets/1f90c042-ebfd-4008-ac64-24c3ffc9ea1d" />

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
