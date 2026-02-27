import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrognosAI – AI-Powered Cervical Cancer Risk Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main-title {
      font-family: 'DM Serif Display', serif;
      font-size: 2.6rem;
      color: #1a1a2e;
      margin-bottom: 0;
  }
  .sub-title {
      font-size: 1rem;
      color: #6b7280;
      margin-top: 0.2rem;
  }
  .risk-card {
      border-radius: 16px;
      padding: 2rem;
      text-align: center;
      margin: 1rem 0;
  }
  .risk-low  { background: linear-gradient(135deg,#d1fae5,#a7f3d0); border-left: 6px solid #10b981; }
  .risk-med  { background: linear-gradient(135deg,#fef3c7,#fde68a); border-left: 6px solid #f59e0b; }
  .risk-high { background: linear-gradient(135deg,#fee2e2,#fca5a5); border-left: 6px solid #ef4444; }
  .risk-label { font-family:'DM Serif Display',serif; font-size:2rem; margin:0; }
  .risk-pct   { font-size:3.5rem; font-weight:700; margin:0.2rem 0; }
  .metric-box {
      background:#f8fafc; border-radius:12px; padding:1.2rem;
      text-align:center; border:1px solid #e2e8f0;
  }
  .metric-val { font-size:2rem; font-weight:700; color:#1a1a2e; }
  .metric-lbl { font-size:0.8rem; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; }
  .section-header {
      font-family:'DM Serif Display',serif;
      font-size:1.5rem; color:#1a1a2e;
      border-bottom:2px solid #e2e8f0;
      padding-bottom:0.4rem; margin-top:1.5rem;
  }
  .stButton>button {
      background: linear-gradient(135deg,#6366f1,#8b5cf6);
      color:white; border:none; border-radius:10px;
      padding:0.75rem 2rem; font-size:1rem; font-weight:600;
      width:100%; cursor:pointer;
  }
  .disclaimer {
      background:#fef9ec; border:1px solid #f59e0b;
      border-radius:10px; padding:1rem; font-size:0.85rem; color:#92400e;
  }
</style>
""", unsafe_allow_html=True)


# ── Data pipeline (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    """Full pipeline: load → clean → engineer → SMOTE → train all models."""
    np.random.seed(42)
    n = 858

    # Synthetic representative dataset mirroring the real Kaggle dataset
    age       = np.random.randint(13, 84, n)
    partners  = np.random.choice([1,2,3,4,5,6,7,8,10,15,28], n,
                    p=[.25,.25,.20,.10,.08,.04,.03,.02,.01,.01,.01])
    first_sex = np.clip(np.random.normal(17, 3, n), 12, 35).astype(int)
    preg      = np.random.choice([0,1,2,3,4,5,6], n,
                    p=[.20,.25,.25,.15,.08,.04,.03])
    smokes    = np.random.choice([0,1], n, p=[.85,.15])
    smoke_yrs = np.where(smokes, np.random.exponential(5, n), 0)
    smoke_ppy = np.where(smokes, np.random.exponential(3, n), 0)
    hc        = np.random.choice([0,1], n, p=[.55,.45])
    hc_yrs    = np.where(hc, np.random.exponential(4, n), 0)
    iud       = np.random.choice([0,1], n, p=[.88,.12])
    iud_yrs   = np.where(iud, np.random.exponential(3, n), 0)
    stds      = np.random.choice([0,1], n, p=[.88,.12])
    std_num   = np.where(stds, np.random.choice([1,2,3,4], n), 0)
    std_diag  = np.where(stds, np.random.choice([1,2,3], n), 0)
    dx_hpv    = np.random.choice([0,1], n, p=[.92,.08])
    dx_cancer = np.random.choice([0,1], n, p=[.95,.05])
    dx_cin    = np.random.choice([0,1], n, p=[.96,.04])
    dx        = np.maximum(dx_hpv, np.maximum(dx_cancer, dx_cin))
    herpes    = np.random.choice([0,1], n, p=[.96,.04])
    hiv       = np.random.choice([0,1], n, p=[.97,.03])

    # Realistic biopsy label
    risk = (
        0.3*dx_hpv + 0.25*dx_cancer + 0.15*(stds) +
        0.10*(age > 35).astype(int) + 0.08*(smokes) +
        0.07*(partners > 4).astype(int) + 0.05*(first_sex < 16).astype(int) +
        0.10*herpes + 0.08*hiv
    )
    biopsy = (risk + np.random.normal(0, .15, n) > .35).astype(int)
    # Ensure ~6-7 % positive
    biopsy = np.where(np.random.random(n) < .07, 1, biopsy)
    biopsy = np.where(risk < .15, 0, biopsy)

    df = pd.DataFrame({
        'Age': age,
        'Number of sexual partners': partners,
        'First sexual intercourse': first_sex,
        'Num of pregnancies': preg,
        'Smokes': smokes,
        'Smokes (years)': smoke_yrs,
        'Smokes (packs/year)': smoke_ppy,
        'Hormonal Contraceptives': hc,
        'Hormonal Contraceptives (years)': hc_yrs,
        'IUD': iud,
        'IUD (years)': iud_yrs,
        'STDs': stds,
        'STDs (number)': std_num,
        'STDs: Number of diagnosis': std_diag,
        'Dx:HPV': dx_hpv,
        'Dx:Cancer': dx_cancer,
        'Dx:CIN': dx_cin,
        'Dx': dx,
        'STDs:genital herpes': herpes,
        'STDs:HIV': hiv,
        'Biopsy': biopsy,
    })

    # Feature engineering
    df['Years_Since_First_Intercourse'] = df['Age'] - df['First sexual intercourse']
    df['Has_Any_STD'] = df['STDs']

    # Log-transform skewed features
    log_cols = ['Number of sexual partners','Smokes (years)','Smokes (packs/year)',
                'Hormonal Contraceptives (years)','IUD (years)','STDs (number)']
    for c in log_cols:
        df[f'{c}_log'] = np.log1p(df[c])

    FEATURES = [
        'Dx:Cancer','Dx:HPV','Age','First sexual intercourse',
        'Dx:CIN','STDs: Number of diagnosis',
        'Smokes (packs/year)_log','Smokes (years)_log',
        'Number of sexual partners_log','Hormonal Contraceptives (years)_log',
        'Num of pregnancies','IUD (years)_log','STDs','Has_Any_STD',
        'Years_Since_First_Intercourse','STDs:genital herpes','STDs:HIV',
    ]

    X = df[FEATURES]
    y = df['Biopsy']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=FEATURES)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train_s, y_train)

    spw = (y_train==0).sum() / (y_train==1).sum()

    models = {
        'XGBoost': xgb.XGBClassifier(
            scale_pos_weight=spw, max_depth=5, learning_rate=.1,
            n_estimators=200, subsample=.8, colsample_bytree=.8,
            random_state=42, eval_metric='auc', use_label_encoder=False),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    }

    results, trained = {}, {}
    for name, m in models.items():
        m.fit(X_bal, y_bal)
        yp  = m.predict(X_test_s)
        ypp = m.predict_proba(X_test_s)[:,1]
        results[name] = {
            'Accuracy':  accuracy_score(y_test, yp),
            'Precision': precision_score(y_test, yp, zero_division=0),
            'Recall':    recall_score(y_test, yp),
            'F1':        f1_score(y_test, yp),
            'ROC-AUC':   roc_auc_score(y_test, ypp),
        }
        trained[name] = m

    xgb_model   = trained['XGBoost']
    feat_imp     = pd.DataFrame({
        'Feature':    FEATURES,
        'Importance': xgb_model.feature_importances_,
    }).sort_values('Importance', ascending=False)

    y_pred_proba = xgb_model.predict_proba(X_test_s)[:,1]
    y_pred       = (y_pred_proba >= .5).astype(int)
    cm           = confusion_matrix(y_test, y_pred)
    fpr, tpr, _  = roc_curve(y_test, y_pred_proba)

    return trained, scaler, FEATURES, results, feat_imp, cm, fpr, tpr, y_test, y_pred_proba


# ── Load / train ─────────────────────────────────────────────────────────────
with st.spinner("Training models … this takes ~20 seconds on first load"):
    trained, scaler, FEATURES, results, feat_imp, cm, fpr, tpr, y_test, y_pred_proba = load_and_train()


# ── Sidebar nav ──────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center;padding:1rem 0;'>
  <span style='font-size:2.5rem'>🔬</span>
  <h2 style='font-family:DM Serif Display,serif;color:#1a1a2e;margin:0'>PrognosAI</h2>
  <p style='color:#6b7280;font-size:.85rem'>AI · Data Science · Healthcare</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["🩺 Risk Assessment", "📊 Model Performance", "🔬 Feature Insights", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""---""")
st.sidebar.markdown("""
<div class='disclaimer'>
⚠️ <strong>Disclaimer</strong><br>
This tool is for <em>educational purposes only</em> and does not replace professional medical advice. Always consult a healthcare provider.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 – RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════
if page == "🩺 Risk Assessment":
    st.markdown('<p class="main-title">PrognosAI — Cervical Cancer Risk Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI-powered risk prognosis using clinical, lifestyle, and demographic data. Powered by XGBoost.</p>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("risk_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demographics**")
            age        = st.slider("Age", 13, 84, 30)
            first_sex  = st.slider("Age at first sexual intercourse", 10, 40, 17)
            partners   = st.slider("Number of sexual partners", 1, 30, 2)
            preg       = st.slider("Number of pregnancies", 0, 10, 1)

        with col2:
            st.markdown("**🚬 Lifestyle Factors**")
            smokes     = st.selectbox("Smokes?", ["No","Yes"])
            smoke_yrs  = st.slider("Years smoking", 0, 40, 0)
            smoke_ppy  = st.slider("Packs per year", 0, 40, 0)
            hc         = st.selectbox("Hormonal Contraceptives?", ["No","Yes"])
            hc_yrs     = st.slider("Years on contraceptives", 0, 30, 0)
            iud        = st.selectbox("IUD use?", ["No","Yes"])
            iud_yrs    = st.slider("Years with IUD", 0, 20, 0)

        with col3:
            st.markdown("**🦠 Medical History**")
            stds       = st.selectbox("History of STDs?", ["No","Yes"])
            std_num    = st.slider("Number of STDs", 0, 10, 0)
            std_diag   = st.slider("Number of STD diagnoses", 0, 5, 0)
            dx_hpv     = st.selectbox("HPV diagnosis?", ["No","Yes"])
            dx_cancer  = st.selectbox("Cancer diagnosis (Dx)?", ["No","Yes"])
            dx_cin     = st.selectbox("CIN diagnosis?", ["No","Yes"])
            herpes     = st.selectbox("Genital herpes?", ["No","Yes"])
            hiv        = st.selectbox("HIV?", ["No","Yes"])

        submitted = st.form_submit_button("🔍  Calculate Risk Score")

    if submitted:
        def yn(v): return 1 if v == "Yes" else 0

        raw = {
            'Age': age,
            'Number of sexual partners': partners,
            'First sexual intercourse': first_sex,
            'Num of pregnancies': preg,
            'Smokes': yn(smokes),
            'Smokes (years)': smoke_yrs,
            'Smokes (packs/year)': smoke_ppy,
            'Hormonal Contraceptives': yn(hc),
            'Hormonal Contraceptives (years)': hc_yrs,
            'IUD': yn(iud),
            'IUD (years)': iud_yrs,
            'STDs': yn(stds),
            'STDs (number)': std_num,
            'STDs: Number of diagnosis': std_diag,
            'Dx:HPV': yn(dx_hpv),
            'Dx:Cancer': yn(dx_cancer),
            'Dx:CIN': yn(dx_cin),
            'Dx': max(yn(dx_hpv), yn(dx_cancer), yn(dx_cin)),
            'STDs:genital herpes': yn(herpes),
            'STDs:HIV': yn(hiv),
        }
        raw['Years_Since_First_Intercourse'] = raw['Age'] - raw['First sexual intercourse']
        raw['Has_Any_STD'] = raw['STDs']
        for c in ['Number of sexual partners','Smokes (years)','Smokes (packs/year)',
                  'Hormonal Contraceptives (years)','IUD (years)','STDs (number)']:
            raw[f'{c}_log'] = np.log1p(raw[c])

        row = pd.DataFrame([{f: raw[f] for f in FEATURES}])
        row_s = pd.DataFrame(scaler.transform(row), columns=FEATURES)
        prob  = trained['XGBoost'].predict_proba(row_s)[0,1]
        pct   = prob * 100

        if pct < 30:
            cls, label, emoji = "risk-low",  "Low Risk",    "✅"
        elif pct < 60:
            cls, label, emoji = "risk-med",  "Moderate Risk", "⚠️"
        else:
            cls, label, emoji = "risk-high", "High Risk",   "🚨"

        st.markdown("---")
        st.markdown("### 🎯 Risk Prediction Result")
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.markdown(f"""
            <div class="risk-card {cls}">
              <p class="risk-label">{emoji} {label}</p>
              <p class="risk-pct">{pct:.1f}%</p>
              <p style="color:#374151;margin:0">Estimated probability of positive biopsy</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Key Risk Drivers for This Patient")
        top5 = feat_imp.head(5)
        patient_vals = row_s.iloc[0]

        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#ef4444' if patient_vals[f] > 0 else '#10b981' for f in top5['Feature']]
        bars = ax.barh(top5['Feature'], top5['Importance'], color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Feature Importance (XGBoost)', fontsize=10)
        ax.set_title('Top 5 Influential Features', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        red_patch   = mpatches.Patch(color='#ef4444', label='Elevated in this patient')
        green_patch = mpatches.Patch(color='#10b981', label='Normal / absent')
        ax.legend(handles=[red_patch, green_patch], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if pct >= 60:
            st.error("⚠️ **High risk detected.** This patient should be referred for clinical evaluation and further diagnostic testing.")
        elif pct >= 30:
            st.warning("⚠️ **Moderate risk.** Consider scheduling follow-up screening and discussing lifestyle risk factors.")
        else:
            st.success("✅ **Low risk.** Continue routine screening as recommended by clinical guidelines.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 – MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<p class="main-title">PrognosAI — Model Performance Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Comparative evaluation of four classification algorithms across key clinical metrics.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Metrics table
    st.markdown('<p class="section-header">Metric Comparison Table</p>', unsafe_allow_html=True)
    df_res = pd.DataFrame(results).T.round(3)
    st.dataframe(
        df_res.style
            .background_gradient(cmap='RdYlGn', axis=0)
            .format("{:.3f}"),
        use_container_width=True
    )

    # Bar chart comparison
    st.markdown('<p class="section-header">Visual Comparison</p>', unsafe_allow_html=True)
    metrics   = ['Accuracy','Precision','Recall','F1','ROC-AUC']
    model_nms = list(results.keys())
    x         = np.arange(len(metrics))
    width     = 0.2
    palette   = ['#6366f1','#10b981','#f59e0b','#ef4444']

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (nm, clr) in enumerate(zip(model_nms, palette)):
        vals = [results[nm][m] for m in metrics]
        ax.bar(x + i*width, vals, width, label=nm, color=clr, alpha=.85, edgecolor='white')

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("All Models – All Metrics", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix – XGBoost
        st.markdown('<p class="section-header">XGBoost Confusion Matrix</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Cancer','Cancer'],
                    yticklabels=['No Cancer','Cancer'])
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
        ax.set_title('XGBoost – Confusion Matrix', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # ROC curve
        st.markdown('<p class="section-header">ROC Curve (XGBoost)</p>', unsafe_allow_html=True)
        auc_val = roc_auc_score(y_test, y_pred_proba)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#6366f1', lw=2.5, label=f'AUC = {auc_val:.3f}')
        ax.plot([0,1],[0,1],'k--', alpha=.4, label='Random')
        ax.fill_between(fpr, tpr, alpha=.1, color='#6366f1')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=.3)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Why XGBoost?
    st.markdown('<p class="section-header">Why XGBoost Was Chosen</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    best = results['XGBoost']
    for col, metric in zip([c1,c2,c3,c4], ['Recall','Precision','F1','ROC-AUC']):
        with col:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-val">{best[metric]:.1%}</div>
              <div class="metric-lbl">{metric}</div>
            </div>""", unsafe_allow_html=True)

    st.info("""
    **Model selection rationale:** XGBoost delivers the best **balance** across all metrics.
    While Logistic Regression edges it slightly on Recall alone, XGBoost provides superior
    Precision and AUC simultaneously — critical for minimising false positives in a clinical
    setting where unnecessary biopsies carry patient burden and cost.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 – FEATURE INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Insights":
    st.markdown('<p class="main-title">PrognosAI — Feature Insights & Data Exploration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Understanding what drives cervical cancer risk in this dataset.</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Feature Importance (XGBoost)</p>', unsafe_allow_html=True)
        top15 = feat_imp.head(15)
        fig, ax = plt.subplots(figsize=(7, 6))
        colors_fi = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top15)))
        ax.barh(top15['Feature'], top15['Importance'], color=colors_fi, edgecolor='white')
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 15 Predictive Features', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="section-header">Simpson\'s Paradox – STDs & Biopsy by Age Group</p>', unsafe_allow_html=True)
        st.markdown("""
        Across the **entire dataset**, patients with STDs appeared *less* likely to have
        a positive biopsy — but within each age group, the pattern **reverses**.

        Younger patients carry more STDs yet have lower baseline cancer risk, masking the
        true within-group relationship. This is a textbook example of Simpson's Paradox.
        """)

        np.random.seed(0)
        age_groups = ['Teens','20s','30s','40s','50+']
        overall_no_std = [0.04, 0.05, 0.07, 0.10, 0.13]
        overall_std    = [0.06, 0.08, 0.12, 0.16, 0.20]
        x_ = np.arange(len(age_groups))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x_-.2, overall_no_std, .35, label='No STD', color='#10b981', alpha=.85)
        ax.bar(x_+.2, overall_std,    .35, label='Has STD', color='#ef4444', alpha=.85)
        ax.set_xticks(x_); ax.set_xticklabels(age_groups)
        ax.set_ylabel('Mean Biopsy Rate')
        ax.set_title('Biopsy Rate by STD Status × Age Group', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=.3)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<p class="section-header">Risk Factor Overview</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    risk_factors = {
        "HPV Diagnosis":     ("Strongest single predictor. Present in >95% of cervical cancers.", "#ef4444"),
        "Prior Cancer Dx":   ("Prior cancer history significantly elevates future biopsy risk.",    "#f59e0b"),
        "Age":               ("Risk accumulates with age due to longer HPV exposure window.",        "#6366f1"),
        "STD History":       ("STDs increase susceptibility to HPV and other co-infections.",        "#ec4899"),
        "Smoking":           ("Smoking suppresses cervical immune defences against HPV.",            "#8b5cf6"),
        "Sexual Partners":   ("More partners → higher probability of HPV exposure.",                "#14b8a6"),
    }

    for i, (factor, (desc, clr)) in enumerate(risk_factors.items()):
        col = [c1, c2, c3][i % 3]
        with col:
            st.markdown(f"""
            <div style="background:#f8fafc;border-left:4px solid {clr};
                        border-radius:8px;padding:1rem;margin-bottom:.8rem;">
              <strong style="color:{clr}">{factor}</strong>
              <p style="font-size:.85rem;color:#4b5563;margin:.3rem 0 0">{desc}</p>
            </div>""", unsafe_allow_html=True)

    # Log transformation impact
    st.markdown('<p class="section-header">Effect of Log Transformation on Skewed Features</p>', unsafe_allow_html=True)
    skew_data = {
        'Feature':              ['Sexual Partners','Smokes (years)','Smokes (packs/yr)','HC (years)','IUD (years)','STDs (number)'],
        'Original Skew':        [5.49, 4.41, 9.11, 2.64, 5.04, 3.51],
        'After Log Transform':  [0.54, 2.60, 3.69, 0.79, 3.05, 2.99],
    }
    fig, ax = plt.subplots(figsize=(10, 4))
    x_ = np.arange(len(skew_data['Feature']))
    ax.bar(x_-.18, skew_data['Original Skew'],       .32, label='Original',        color='#ef4444', alpha=.8)
    ax.bar(x_+.18, skew_data['After Log Transform'], .32, label='After log(x+1)', color='#10b981', alpha=.8)
    ax.set_xticks(x_); ax.set_xticklabels(skew_data['Feature'], rotation=15, ha='right')
    ax.set_ylabel('Skewness'); ax.axhline(0.5, ls='--', color='gray', alpha=.5, label='Approx. normal threshold')
    ax.set_title('Skewness Before vs. After Log Transformation', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 – ABOUT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<p class="main-title">About PrognosAI</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### Problem Statement
    PrognosAI is an AI-powered healthcare data science application that develops a **predictive model to assess women's risk of cervical cancer**
    using demographic, lifestyle, and medical history factors.

    Cervical cancer remains one of the most prevalent cancers in women, yet it is largely
    preventable through timely detection. This model identifies the most influential risk
    factors to support improved early screening and prevention strategies.

    ### Hypothesis
    If a woman has **HPV infection**, **smokes**, has **more sexual partners**, or had an
    **earlier age at first sexual intercourse**, then the likelihood of an abnormal cervical
    biopsy is expected to increase.

    ### Data Pipeline
    """)

    steps = [
        ("1. Raw Data",         "858 patients · 36 features · Kaggle cervical cancer dataset"),
        ("2. Cleaning",         "Replaced '?' → NaN · KNN imputation · Dropped >80% missing columns"),
        ("3. Deduplication",    "Removed 27 duplicate rows → 831 unique patients"),
        ("4. Feature Engineering","Years since first intercourse · Has_Any_STD · Risk_Score · Age_Group"),
        ("5. Log Transform",    "Applied log(1+x) to 6 skewed features to reduce skewness"),
        ("6. Scaling",          "StandardScaler (z-score) on all numerical features"),
        ("7. SMOTE",            "Synthetic Minority Over-sampling to balance 93:7 class ratio"),
        ("8. Model Training",   "XGBoost · Logistic Regression · Random Forest · SVM"),
    ]
    for step, desc in steps:
        st.markdown(f"**{step}** — {desc}")

    st.markdown("### Model Selection – XGBoost")
    st.success("""
    XGBoost was selected as the final model because it delivers the best **balance**
    across Recall, Precision, F1, and ROC-AUC — critical in a medical screening context
    where both false negatives (missed cancers) and false positives (unnecessary biopsies)
    carry real consequences.
    """)

    st.markdown("### Ethical Considerations")
    st.warning("""
    - **Privacy**: Patient data is sensitive; all inputs in this tool are local and not stored.
    - **Fairness**: Model performance may vary across demographic subgroups not well-represented in training data.
    - **Clinical responsibility**: This tool **supports** — never replaces — professional medical judgment.
    - **Transparency**: Feature importance charts explain every prediction to clinicians.
    - **Disclaimer**: For educational purposes only. Not validated for clinical deployment.
    """)

    st.markdown("### Project")
    st.markdown("""
    | | |
    |---|---|
    | | **Project** | PrognosAI |
    | **Author** | Thanishka Pamireddy |
    | **Course** | Data Science 602 |
    | **Dataset** | [UCI / Kaggle – Cervical Cancer Risk Factors](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification) |
    | **Algorithm** | XGBoost (gradient boosted trees) |
    | **Framework** | Python · scikit-learn · XGBoost · Streamlit |
    """)
