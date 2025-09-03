# Healthcare Denial Management 

A Streamlit-powered machine learning application designed for Revenue Cycle Management (RCM) professionals to analyze healthcare claim denials, predict missing denial reasons, and identify actionable insights to reduce denials and improve collections.

##  Problem Statement

Medical billing analysts at RCM companies need to:
- Identify top CPT codes getting denied
- Understand root causes for lack of payment
- Develop targeted strategies to reduce denials
- Predict denial reasons for incomplete records

This tool automates the analysis process by training machine learning models on labeled data and predicting missing denial reasons while providing comprehensive insights.

##  Features

###  Smart Data Processing
- **Automatic header detection** across CSV/XLSX files
- **Intelligent column mapping** using keyword matching
- **Currency parsing** with support for various formats ($, commas, parentheses)
- **Missing data handling** with median imputation for numerical fields

###  Machine Learning Prediction
- **Logistic Regression** with balanced class weights
- **One-hot encoding** for categorical variables
- **Cross-validation** with stratified sampling
- **Automatic prediction** for rows with missing denial reasons

###  Comprehensive Analytics
- **Model performance metrics** (accuracy, F1-score, classification report)
- **Visual dashboards** with interactive charts
- **Top denied CPT codes** analysis
- **Insurance company** denial patterns
- **Physician-specific** denial trends

###  Actionable Insights
- **Root cause analysis** for high-denial patterns
- **Corrective action recommendations**
- **Payer-specific insights**
- **Revenue recovery opportunities**

##  Getting Started

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn plotly openpyxl
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/infancia03/denial-reason-predictor.git
cd denial-reason-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run denial_reason_predictor_fixed.py
```

##  Usage

### Input Data Format

Your CSV/XLSX file should contain the following columns (flexible naming supported):

| Column | Expected Names | Description |
|--------|----------------|-------------|
| CPT Code | `cpt`, `cpt_code`, `cptcode` | Medical procedure codes |
| Insurance | `insurance`, `insurance_company`, `payer` | Insurance provider names |
| Physician | `physician`, `provider`, `doctor` | Healthcare provider names |
| Payment | `payment`, `amount_paid`, `paid` | Payment amounts received |
| Balance | `balance`, `balance_amount` | Outstanding balance amounts |
| Denial Reason | `denial`, `denial_reason`, `reason` | Denial codes/descriptions |

### Sample Data Structure

```csv
CPT Code,Insurance Company,Physician Name,Payment Amount,Balance,Denial Reason
99213,Medicare,Dr. Smith,$0.00,$100.00,16 - Missing information
99214,Aetna,Dr. Johnson,$80.00,$20.00,
99215,Cigna,Dr. Lee,$0.00,$150.00,45 - Charge exceeds fee schedule
```

### Step-by-Step Process

1. **Upload File**: Choose your CSV or XLSX file
2. **Review Data**: Check the automatically detected headers and preview
3. **Model Training**: The app trains on rows with existing denial reasons
4. **Prediction**: Missing denial reasons are predicted automatically
5. **Analysis**: Review model performance and insights
6. **Export**: Download the completed dataset with predictions

## 🔧 Technical Details

### Machine Learning Pipeline

```python
# Feature preprocessing
ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numerical_features)
])

# Classification model
LogisticRegression(
    solver="saga",
    multi_class="multinomial", 
    class_weight="balanced",
    max_iter=2000
)
```

### Key Functions

- `detect_header_row()`: Automatically identifies header location using keyword matching
- `_canonize()`: Standardizes column names for consistent processing  
- `currency_to_float()`: Robust currency parsing with error handling
- `train_and_predict()`: Complete ML pipeline for training and prediction

##  Output & Insights

### Model Evaluation
- **Accuracy Score**: Overall prediction accuracy
- **F1-Score**: Balanced precision and recall metrics
- **Classification Report**: Per-class performance breakdown
- **Confusion Matrix**: Detailed prediction analysis

### Visual Analytics
- **Top 10 Denied CPT Codes**: Bar chart of most frequently denied procedures
- **Denials by Insurance**: Payer-specific denial patterns
- **Denials by Physician**: Provider-specific trends

### Actionable Recommendations
- **Root Cause Identification**: Common denial patterns and triggers
- **Corrective Actions**: Specific steps to reduce denials
- **Process Improvements**: Workflow and documentation recommendations

##  Common Healthcare Denial Codes

| Code | Description | Typical Root Cause |
|------|-------------|-------------------|
| 16 | Missing information | Incomplete documentation |
| 45 | Charge exceeds fee schedule | Pricing/contract issues |
| 96 | Non-covered service | Authorization/coverage problems |





---

