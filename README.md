# 🫁 LiverGuard Predict

LiverGuard Predict is a web-based machine learning application designed to assess liver disease risk using patient demographics and clinical laboratory results. The app utilizes a Logistic Regression model trained on 6 key clinical biomarkers.

---

## 📌 Features
- **Real-Time Risk Assessment:** Instant probability scores and risk categorizations (Low Risk vs. High Risk).
- **Tailored Recommendations:** Dynamic medical next steps and lifestyle guidelines based on estimated risk probability.
- **Interactive Controls:** User-friendly sliders and dropdowns for smooth clinical data input.
- **Pre-Scaled Pipeline:** Automatic normalization using `StandardScaler` fitted during model training.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Frontend / Framework:** Streamlit
- **Machine Learning:** Scikit-Learn (Logistic Regression, StandardScaler)
- **Data Processing:** Pandas, Joblib

---

## 📊 Selected Biomarkers (6 Features)
1. **Age** (Years)
2. **Gender** (Male / Female)
3. **Total Bilirubin** (mg/dL)
4. **Alamine Aminotransferase - ALT** (IU/L)
5. **Aspartate Aminotransferase - AST** (IU/L)
6. **Total Proteins** (g/dL)

---

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
