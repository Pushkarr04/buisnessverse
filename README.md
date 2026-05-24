# BusinessPulse AI – Smart Business Analytics & Forecasting Platform

**Created by Pushkar Lal Teli**

BusinessPulse AI is an enterprise-level SaaS-style analytics dashboard, machine learning workspace, and AI-driven business decision engine. Built entirely using Python, SQLite, Scikit-learn, XGBoost, and Streamlit, it provides a high-fidelity glassmorphic reporting dashboard that fuses raw SQL queries, forecasting algorithms, automated risk diagnostics, and client retention predictors into a cohesive corporate suite.

---

## 🚀 Key Platform Features

1. **Executive Dashboard (Business Health Portal)**: High-level KPI monitoring (Total Revenue, Margins, Retention Rates) coupled with a dynamic **Business Health Score** index speed-gauge. Shows active AI anomalies.
2. **Sales Analytics & Simulated Planner**: Maps category sales trees alongside interactive price elasticity what-if scenario panels modeled across varied client tiers (SMB, Mid-Market, Enterprise).
3. **Customer Segmentations & RFM**: Multi-dimensional RFM calculations mapped in a **3D interactive scatter plot** using K-Means clustering, accompanied by dynamic persona profiling.
4. **Product Diagnostics & Recommendations**: Direct tabular ABC inventory categorization and Pareto contribution maps, linked to a custom-engineered, dependency-free **Apriori Association recommender** for bundle deals.
5. **Marketing Multi-Channel Heatmaps**: Dynamic correlation matrixes checking marketing allocations, clicks, impressions, conversions, and actual sales revenues across LinkedIn, Google, Social, and Email campaigns.
6. **Financial Performance Waterfall**: Dynamic Waterfall corporate operating ledger mapping Gross Revenue -> COGS -> EBITDA -> Rent -> Payroll -> Taxes -> Net Profit, linked to staff overhead scenario models.
7. **AI & AutoML Sandbox**: 
   - Hyperparameter training logs for customer churn classifiers (Gradient Boosting / Random Forest).
   - Real-time profile churn threat evaluator inputs.
   - **AutoML upload**: Upload *any* external CSV dataset, automatically hot-encode categoricals, select target column, and trigger classifier or regressor training dynamically.
8. **AI Chatbot & HTML5 Voice Assistant**: 
   - Dynamic *PulseAdvisor* chat CFO.
   - **Speech-to-Text mic button**: An HTML5/SpeechRecognition iframe widget that captures voice queries locally and copies them to the clipboard for instant query input.
   - **NL-to-SQL compiler**: Translates plain text into executable SQLite queries, displays SQL codes, and plots tabular visual charts.
9. **Expert SQL Sandbox Editor**: Complete SQL console with active schema explorer tree, prebuilt analytical templates (CTE growths, CLV ranks, Region running totals), and dynamic automated business takeaways digests.
10. **Report Export Center**: Styled multi-sheet openpyxl Excel workbooks and beautiful, highly printable executive HTML briefing PDF layouts.

---

## 🛠️ Technology Stack & Dependencies

- **Core Framework**: Python 3.9+ & Streamlit
- **Data Engineering**: Pandas, NumPy, SQLAlchemy
- **Database Engine**: SQLite3
- **Machine Learning**: Scikit-learn, XGBoost, Statsmodels
- **Visualizations**: Plotly, Seaborn, Matplotlib
- **Reporting & Exporters**: Openpyxl, Requests

---

## 📂 Project Architecture

```
buisnessverse DA/
├── app.py                      # Master application entrypoint, page routing, login gates
├── requirements.txt            # System dependencies manifest
├── README.md                   # Platform deployment instructions and design blueprint
├── assets/
│   └── custom.css              # Dark-mode glassmorphic CSS styling sheet
├── database/
│   ├── __init__.py
│   ├── connection.py           # SQLite db initialization, table creations, performance indexings
│   └── queries.py              # Advanced CTE, Join, and Window queries prebuilt templates
├── models/
│   ├── __init__.py
│   ├── forecaster.py           # Holt-Winters Exponential Smoothing & ML lagged regression forecasters
│   ├── churn.py                # RandomForest classification training, confusion matrix, ROC plots
│   ├── segmentation.py         # Customer RFM aggregations, K-Means clustering, and 3D visual plotting
│   ├── recommender.py          # Custom pandas-native Apriori basket rule recommendations
│   └── financial_models.py     # What-If strategic elasticity simulators and Opex planners
└── utils/
    ├── __init__.py
    ├── data_generator.py       # Multi-year correlated mock transactional relational generator
    ├── ai_engine.py            # Natural Language SQL compilers, Gemini/OpenAI HTTP wrappers, anomaly checks
    ├── pdf_generator.py        # openpyxl sheet bundlers & inline-styled corporate HTML-briefing exports
    └── auth.py                 # Multi-role authentication (Admin/Analyst) and SHA-256 password hashes
```

---

## 🔑 Platform Access Accounts

The application has pre-configured roles that can be modified inside `utils/auth.py`.

* **Admin Role (Full Access Control)**:
  - **Username**: `admin`
  - **Password**: `admin123`
* **Analyst Role (Standard Analytics)**:
  - **Username**: `analyst`
  - **Password**: `analyst123`

---

## ⚙️ Deployment & Setup Instructions

### 1. Clone the Directory
Ensure your shell is positioned in the project workspace:
```powershell
cd "d:\Desktop\buisnessverse DA"
```

### 2. Install Project Dependencies
Run `pip` to install all libraries specified in the manifest:
```powershell
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
Execute the dev server command:
```powershell
streamlit run app.py
```
Upon startup, the database seeding script will automatically run in the background. It will generate **3 years of correlated, seasonal transactional details** and compile them into `database/business_pulse.db` with performance indexes in under 2 seconds. The login page will then load.

---

## 🤖 Cognitive LLM Integrations (Optional)

BusinessPulse AI operates **100% out-of-the-box in local fallback mode** using statistical keyword semantic dictionaries if no API key is specified. 

To enable full conversational dynamic capabilities:
1. Open the **AI Insights Provider Settings** in the sidebar settings.
2. Select your provider (`Gemini` or `OpenAI`).
3. Paste your API key (e.g., Gemini `AIza...` or OpenAI `sk-...`).
4. The chatbot will instantly transform into a cognitive executive consultant capable of compiling complex SQL join syntaxes for *any* query and holding conversational context.
