import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pandas.api.types as ptypes

# Streamlit Config
st.set_page_config(page_title='AI-Powered Analytics', page_icon='📊', layout='wide')
st.title('📊 :rainbow[AI-Powered Data Analytics Portal] 🚀')
st.subheader('Upload your dataset, preprocess, and run AutoML')

# File Upload
file = st.file_uploader('Upload CSV or Excel file', type=['csv', 'xlsx'])
if file:
    try:
        data = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Basic Info Tabs
    st.subheader(':rainbow[Dataset Overview]', divider='rainbow')
    tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Top/Bottom Rows', 'Data Types', 'Columns'])

    with tab1:
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        st.dataframe(data.describe())
    with tab2:
        st.dataframe(data.head(10))
        st.dataframe(data.tail(10))
    with tab3:
        st.dataframe(data.dtypes.astype(str))
    with tab4:
        st.write(list(data.columns))

    # Value Counts
    st.subheader(':rainbow[Value Counts]', divider='rainbow')
    with st.expander('Explore Column Value Counts'):
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox('Select Column', options=list(data.columns), key='value_counts_column')
        with col2:
            top_n = st.number_input('Top N Values', min_value=1, value=5)

        if st.button('Show Counts'):
            result = data[column].value_counts().reset_index().head(top_n)
            result.columns = [column, 'Count']
            st.dataframe(result)
            st.plotly_chart(px.bar(result, x=column, y='Count', text='Count'))
            st.plotly_chart(px.pie(result, names=column, values='Count'))

    # Visualization
    st.subheader(':gray[Custom Visualization]', divider='gray')
    chart_type = st.selectbox('Choose Chart Type', ['line', 'bar', 'scatter', 'pie', 'sunburst'])

    if chart_type in ['line', 'bar', 'scatter']:
        x_axis = st.selectbox('X Axis', options=list(data.columns))
        y_axis = st.selectbox('Y Axis', options=list(data.columns))
        color = st.selectbox('Color', options=[None] + list(data.columns))
        if chart_type == 'line':
            st.plotly_chart(px.line(data, x=x_axis, y=y_axis, color=color))
        elif chart_type == 'bar':
            st.plotly_chart(px.bar(data, x=x_axis, y=y_axis, color=color))
        elif chart_type == 'scatter':
            size = st.selectbox('Size', options=[None] + list(data.columns))
            st.plotly_chart(px.scatter(data, x=x_axis, y=y_axis, color=color, size=size))
    elif chart_type == 'pie':
        values = st.selectbox('Values', options=list(data.columns))
        names = st.selectbox('Labels', options=list(data.columns))
        st.plotly_chart(px.pie(data, values=values, names=names))
    elif chart_type == 'sunburst':
        path = st.multiselect('Hierarchy Path', options=list(data.columns))
        if path:
            st.plotly_chart(px.sunburst(data, path=path))

    # --- Preprocessing ---
    st.subheader('🛠 Preprocessing & Feature Engineering')
    
    preprocessed_data = data.copy()

    # Drop datetime
    preprocessed_data.drop(columns=preprocessed_data.select_dtypes(include='datetime64').columns, inplace=True)

    # Handle missing values
    preprocessed_data.fillna(preprocessed_data.mean(numeric_only=True), inplace=True)

    # Encode categoricals
    for col in preprocessed_data.select_dtypes(include='object').columns:
        preprocessed_data[col] = LabelEncoder().fit_transform(preprocessed_data[col].astype(str))

    # Outlier removal (optional toggle)
    if st.checkbox("Remove outliers (Z-Score > 3)", value=True):
        z_scores = (preprocessed_data - preprocessed_data.mean(numeric_only=True)) / preprocessed_data.std(numeric_only=True)
        cleaned_data = preprocessed_data[(z_scores < 3).all(axis=1)]
        if cleaned_data.empty:
            st.warning("⚠️ Dataset became empty after outlier removal. Skipping this step.")
        else:
            preprocessed_data = cleaned_data

    # Remove duplicates
    preprocessed_data.drop_duplicates(inplace=True)

    st.write("### Preprocessed Data Preview")
    st.dataframe(preprocessed_data.head())

    # Pairplot
    st.subheader('📊 Pairplot (After Preprocessing)')
    numerical_cols = preprocessed_data.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        st.plotly_chart(px.scatter_matrix(preprocessed_data, dimensions=numerical_cols[:5]))

    # --- Feature Selection ---
    st.subheader('📌 Feature Selection')
    target_column = st.selectbox('Select Target Column', options=list(preprocessed_data.columns))
    
    if target_column:
        try:
            X = preprocessed_data.drop(columns=[target_column])
            y = preprocessed_data[target_column]

            # Detect task type safely
            if (ptypes.is_integer_dtype(y) and y.nunique() <= 20) or y.dtype == "object":
                task_type = "classification"
            else:
                task_type = "regression"

            if task_type == "classification" and y.nunique() > 20:
                st.warning("⚠️ Target has too many unique values. This looks like regression data.")

            # Scale features only (not target)
            scaler = StandardScaler()
            num_cols = X.select_dtypes(include=[np.number]).columns
            X[num_cols] = scaler.fit_transform(X[num_cols])

            # Feature selection
            score_func = f_classif if task_type == 'classification' else f_regression
            k_best = min(5, X.shape[1])
            selector = SelectKBest(score_func=score_func, k=k_best)
            X_new = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()

            st.write(f"Selected Features: {selected_features}")
            final_data = pd.DataFrame(X_new, columns=selected_features)
            final_data[target_column] = y.values
            
            st.write("### Data After Feature Selection")
            st.dataframe(final_data.head())

            # --- Model Training ---
            st.subheader('🔍 Train & Evaluate Models')
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

            # Define models
            if task_type == 'classification':
                models = {"Random Forest Classifier": RandomForestClassifier()}
            else:
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor()
                }

            results = {}
            best_model_name = None
            best_score = float('-inf') if task_type == 'regression' else 0
            predictions = None

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                if task_type == 'classification':
                    acc = accuracy_score(y_test, predictions)
                    st.write(f"{model_name}: Accuracy = {acc:.4f}")
                    metric = acc
                    if metric > best_score:
                        best_score = metric
                        best_model_name = model_name
                else: # Regression
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    st.write(f"{model_name}: MSE = {mse:.4f}, R² = {r2:.4f}")
                    metric = r2
                    if metric > best_score:
                        best_score = metric
                        best_model_name = model_name
                
                results[model_name] = metric

            st.success(f"🎯 Best Model: **{best_model_name}** with a score of: {best_score:.4f}")

            # Prediction Plot
            if predictions is not None:
                st.subheader("📊 Actual vs Predicted")
                if task_type == "regression":
                    plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
                    st.plotly_chart(px.scatter(plot_df, x='Actual', y='Predicted',
                                               title=f"Actual vs. Predicted for {best_model_name}"))
                else:
                    plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
                    st.plotly_chart(px.histogram(plot_df, x='Actual', color='Predicted', barmode='group'))

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
