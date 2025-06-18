# 📞 Telecom Customer Churn Prediction - MLOps Project

## 📜 Project Overview
**Objective**:  
Reproduce and extend key research papers on churn prediction in telecom using ML, following CRISP-DM phases. Focus on model interpretability and deployment.

**Papers**:  
1. *Customer Churn Prediction in Telecom Using ML*  
2. *ML Techniques for Customer Retention in Telecom*  
3. *Explaining Churn with Tabular ML Models*  

---

## 🏗️ Project Structure (CRISP-DM)
```bash
Telecom_Churn_ML/
├── data/
│   ├── raw/                  # Original dataset (e.g., `telecom_churn.csv`)
│   └── processed/            # Cleaned data (DVC-tracked)
├── notebooks/
│   ├── 1_Business_Understanding.ipynb  # Paper analysis + EDA
│   ├── 2_Data_Preprocessing.ipynb      # Feature engineering
│   ├── 3_Modeling_Comparison.ipynb     # Paper reproduction (3 models)
│   └── 4_Evaluation_Explainability.ipynb  # SHAP/LIME analysis
├── src/
│   ├── data_preprocessing/   # Pipeline scripts
│   ├── models/               # ML models (LogReg, XGBoost, etc.)
│   ├── evaluation/           # Metrics + plots
│   └── api/                  # FastAPI deployment
├── papers/                   # PDFs of the 3 research papers
├── reports/                  # Final presentation/PDF report
└── tests/                    # Unit tests
