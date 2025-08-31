# AI-Powered Data Analytics Portal

An **end-to-end Streamlit AutoML dashboard** for data exploration, preprocessing, feature selection, and machine learning.  
Upload your datasets, visualize them, preprocess, select features, and train models — all in one interactive portal!

---

## Features

- **Data Upload:** CSV or Excel files.
- **Data Exploration:**
  - Dataset summary (rows, columns, statistics)
  - Top/Bottom rows
  - Column data types
  - Column value counts
- **Custom Visualizations:** Line, Bar, Scatter, Pie, Sunburst, Pairplots.
- **Preprocessing & Feature Engineering:**
  - Handle missing values
  - Encode categorical features
  - Scale numeric features
  - Optional outlier removal
  - Remove duplicates
- **Feature Selection:** Select top features automatically.
- **Machine Learning Models:**
  - Classification: Random Forest Classifier
  - Regression: Linear Regression, Random Forest Regressor
- **Model Evaluation:**
  - Accuracy for classification
  - MSE & R² for regression
  - Actual vs Predicted plots
- **Download:** Preprocessed and feature-selected datasets.

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/YashHaval/ai-powered-analytics.git
cd ai-powered-analytics
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

ai-powered-analytics/
│── app.py
│── requirements.txt
│── README.md
