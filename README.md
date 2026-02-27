🧬 PrognosAI — AI-Powered Cervical Cancer Risk Intelligence

An AI-powered healthcare data science application combining machine learning, clinical data, and interactive visualisation to predict cervical cancer risk.

📌 Problem Statement
Cervical cancer remains one of the most prevalent cancers in women worldwide, yet it is largely preventable through timely detection. This project builds a predictive model that:

Identifies the most influential risk factors for cervical cancer
Generates personalized risk scores based on patient demographics, lifestyle, and medical history
Compares four ML algorithms on clinical performance metrics
Demonstrates real-world data science techniques including handling class imbalance, feature engineering, and model interpretation

Target variable: Biopsy result — 0 (no cancer) · 1 (cancer detected)

🧠 Hypothesis

If a woman has HPV infection, smokes, has more sexual partners, or had an earlier age at first sexual intercourse, then the likelihood of an abnormal cervical biopsy is expected to increase.

🔄 Data Pipeline
StepDescriptionRaw Data858 patients · 36 features · Kaggle cervical cancer datasetCleaningReplace ? → NaN · KNN imputation · Drop columns with >80% missingDeduplicationRemoved 27 duplicate rows → 831 unique patientsFeature EngineeringYears_Since_First_Intercourse, Has_Any_STD, Risk_Score, Age_GroupLog TransformApplied log(1+x) to 6 right-skewed featuresScalingStandardScaler (z-score normalization)SMOTEBalanced the 93:7 class ratio via synthetic oversamplingModelingXGBoost · Logistic Regression · Random Forest · SVM

📊 Model Results
ModelAccuracyPrecisionRecallF1ROC-AUCXGBoost ⭐96.9%75.0%81.8%0.7830.912Logistic Regression97.0%71.4%90.9%0.8000.960Random Forest97.0%80.0%72.7%0.7620.913SVM (RBF)95.8%75.0%54.5%0.6320.962
XGBoost was selected for its best overall balance between recall and precision — critical in medical screening where both false negatives (missed cancers) and false positives (unnecessary biopsies) have real consequences.

🌟 App Features
PageDescription🩺 Risk AssessmentInput patient data → get personalized cancer risk score + key risk drivers📊 Model PerformanceConfusion matrix, ROC curve, metric comparison across all 4 models🔬 Feature InsightsFeature importance, Simpson's Paradox demo, log-transform impactℹ️ AboutProject background, pipeline overview, ethical considerations

🚀 Run Locally
bash# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/prognosai-cervical-risk.git
cd prognosai-cervical-risk

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
App will open at http://localhost:8501

☁️ Deploy to Streamlit Cloud (Free)

Push this repo to GitHub
Go to share.streamlit.io
Click "New app" → select your repo → set app.py as main file
Click Deploy — your app is live in ~2 minutes!


📦 Dependencies
streamlit>=1.32.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0

⚠️ Ethical Disclaimer
This tool is for educational purposes only and is not validated for clinical use.

All patient inputs are processed locally and never stored
Predictions should support — never replace — professional medical judgment
Model performance may vary across demographic subgroups
Always consult a qualified healthcare provider for medical decisions


📚 Dataset
Source: UCI ML Repository / Kaggle – Cervical Cancer Risk Factors
Fernandes, K., Cardoso, J.S., Fernandes, J. (2017). Transfer Learning with Partial Observability Applied to Cervical Cancer Screening. Iberian Conference on Pattern Recognition and Image Analysis.

Note: Due to patient privacy, the raw dataset is not included in this repo. Download it from Kaggle and place it at data/risk_factors_cervical_cancer.csv.
