import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os

# Set page config with custom dark mode and page icon
st.set_page_config(page_title="ML Big Data Analysis", layout="wide", page_icon="🤖", initial_sidebar_state="expanded")

# Apply custom CSS for a modern and clean design
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #3a3a3a, #111);
            color: #EAEAEA;
            font-family: 'Roboto', sans-serif;
        }

        /* Sidebar Styling */
        .css-1d391kg {
            background: linear-gradient(145deg, #1f1f1f, #121212);
            color: #EAEAEA;
            font-family: 'Roboto', sans-serif;
            box-shadow: 4px 0px 6px rgba(0, 0, 0, 0.2);
        }

        /* Login Container */
        .login-container {
            background-color: rgba(0, 0, 0, 0.8);
            border-radius: 10px;
            padding: 40px 60px;
            width: 350px;
            margin: 80px auto;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            text-align: center;
            transform: translateY(-10%);
            transition: transform 0.5s ease-in-out;
        }

        .login-container:hover {
            transform: translateY(-15%);
        }

        .login-logo {
            font-size: 36px;
            font-weight: 600;
            color: #4CAF50;
            margin-bottom: 30px;
            text-transform: uppercase;
        }

        /* Input Boxes */
        .stTextInput>div>div>input {
            background-color: white;
            border: 1px solid #444;
            color: black;
            border-radius: 5px;
            padding: 1px;
            margin: 1px 0;
            font-size: 16px;
        }

        .stTextInput>div>div>input:focus {
            border-color: #4CAF50;
        }

        /* Login Button */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 50px;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            cursor: pointer;
            width: 100%;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            transition: background-color 0.3s, transform 0.2s;
        }

        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }

        /* Forgot Password Link */
        .forgot-password {
            color: #4CAF50;
            font-size: 14px;
            text-decoration: none;
            margin-top: 10px;
            display: block;
        }

        .forgot-password:hover {
            text-decoration: underline;
        }

        /* Error Message */
        .error-message {
            color: red;
            font-size: 14px;
            margin-top: 10px;
        }

        .stMarkdown {
            font-family: 'Roboto', sans-serif;
            text-align: center;
        }

        /* Header Styling */
        .stHeader {
            text-align: center;
            padding: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Predefined login credentials (You can change these to a secure method or database)
USER_CREDENTIALS = {"username": "RASHEED", "password": "ZAMRUD149"}

# Function to check login status
def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

# Function to display login form with modern UI
def show_login_form():
    st.write("### 🤖  WELCOME BACK  🤖")
    st.write("###  📋 INPUT YOUR DATA  📋")
    username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")
    password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
    
    if st.button("Login", use_container_width=True):
        if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
            st.session_state['logged_in'] = True  # Set login status to True
            st.success("Logged in successfully!", icon="✅")
        else:
            st.error("Invalid username or password. Please try again.", icon="❌")

# Sidebar navigation with icons and improved UI
def show_sidebar():
    st.sidebar.title("🛠️ Menu")
    option = st.sidebar.radio(
        "Select Process:",
        ["📂 Data Preparation", "📊 EDA", "📈 Modeling", "🤖 Prediction", "📊 Cross-validation", "📈 Download & Export Model", "⚙️ Log Out"],
        index=0,  # Default option to be selected
        label_visibility="collapsed"
    )
    return option

# Reusable function to load datasets with progress bar and modern feedback
def load_dataset(upload_key):
    uploaded_file = st.file_uploader(f"Upload Dataset for {upload_key} (CSV)", type=["csv"], key=upload_key)
    if uploaded_file:
        try:
            st.info("Loading dataset...")
            data = pd.read_csv(uploaded_file)
            st.success("Dataset successfully loaded! 👍")
            return data
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return None
    else:
        st.info(f"Please upload a dataset for {upload_key}.")
        return None

# Function for handling missing values
def handle_missing_values(data):
    imputer = SimpleImputer(strategy="median")
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    empty_columns = [col for col in numerical_cols if data[col].isnull().all()]
    
    if empty_columns:
        st.warning(f"Columns fully empty will be removed: {empty_columns}")
        data = data.drop(columns=empty_columns)
        numerical_cols = [col for col in numerical_cols if col not in empty_columns]
    
    imputed_data = imputer.fit_transform(data[numerical_cols])
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols, index=data.index)
    data[numerical_cols] = imputed_df
    return data

# Fungsi untuk menyimpan dataset yang dibersihkan dengan nama yang diinginkan
def save_cleaned_dataset(data):
    # Minta pengguna untuk memasukkan nama file dan lokasi penyimpanan
    folder_path = "cleaned_datasets"  # Folder tempat menyimpan dataset
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Membuat folder jika belum ada
    
    filename = st.text_input("Masukkan nama file untuk dataset yang sudah dibersihkan (tanpa ekstensi):")
    
    if filename:
        # Tentukan path lengkap untuk menyimpan dataset
        file_path = os.path.join(folder_path, f"{filename}.csv")
        
        # Cek apakah file dengan nama tersebut sudah ada
        if os.path.exists(file_path):
            st.warning(f"File '{filename}.csv' sudah ada. Silakan pilih nama lain.")
        else:
            # Simpan dataset
            data.to_csv(file_path, index=False)
            st.success(f"Dataset berhasil disimpan sebagai '{filename}.csv' di folder '{folder_path}'!")
    else:
        st.warning("Harap masukkan nama file untuk dataset yang sudah dibersihkan.")

# Main app logic
check_login()

if not st.session_state['logged_in']:
    show_login_form()  # Show login form if not logged in
else:
    # Once logged in, show the main content
    option = show_sidebar()

    # Data Preparation
    if option == "📂 Data Preparation":
        st.title("📂 Data Preparation")
        st.write("### 🛠️ Clean Your Dataset")
        data = load_dataset("Data Preparation")
        if data is not None:
            st.write("### 📋 Dataset Overview")
            st.dataframe(data.head())
            st.write("### ✅ Handle Missing Values")
            data = handle_missing_values(data)
            st.dataframe(data.head())
            if st.button("💾 Save Clean Dataset"):
                save_cleaned_dataset(data)  # Memanggil fungsi untuk menyimpan dataset yang sudah dibersihkan

    # Exploratory Data Analysis (EDA)
    elif option == "📊 EDA":
        st.title("📊 Exploratory Data Analysis")
        st.write("### 🔍 Analyze Your Data")
        data = load_dataset("EDA")
        if data is not None:
            st.write("### 📋 Dataset Overview")
            st.dataframe(data.head())
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) > 0:
                selected_col = st.selectbox("Choose Column for Distribution Visualization:", numerical_cols)
                if selected_col:
                    st.write(f"### 🔢 Distribution of {selected_col}")
                    fig = px.histogram(data, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig)
                if st.checkbox("Show Correlation Heatmap"):
                    if len(numerical_cols) > 1:
                        st.write("### 🌡️ Correlation Heatmap")
                        fig_corr, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(data[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                        st.pyplot(fig_corr)
                    else:
                        st.warning("Dataset does not have enough numeric columns for a heatmap.")
            else:
                st.warning("Dataset does not have numeric columns for analysis.")
            
            if st.checkbox("Show Descriptive Statistics"):
                st.write("### 📊 Descriptive Statistics")
                st.dataframe(data.describe())
            
            if st.checkbox("Show Scatter Matrix"):
                st.write("### 🔍 Scatter Matrix")
                fig_scatter_matrix = px.scatter_matrix(
                    data, 
                    dimensions=data.select_dtypes(include=['float64', 'int64']).columns,
                    title="Scatter Matrix of Features"
                )
                st.plotly_chart(fig_scatter_matrix)

    # Model Training and Evaluation
    elif option == "📈 Modeling":
        st.title("📈 Modeling")
        st.write("### 🧠 Train Your Machine Learning Model")
        data = load_dataset("Modeling")
        if data is not None:
            st.write("### 📋 Dataset Overview")
            st.dataframe(data.head())
            target = st.selectbox("🎯 Select Target Variable:", data.columns)
            features = data.drop(columns=[target])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            st.success("Model trained successfully!")
            y_pred = model.predict(X_test)
            st.write("### 📊 Model Evaluation")
            st.metric("🎯 Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text("📋 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            if st.checkbox("Show Feature Importance"):
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                feature_importance = rf_model.feature_importances_
                feature_names = X.columns
                fig = px.bar(x=feature_names, y=feature_importance, title="Feature Importance")
                st.plotly_chart(fig)

    # Prediction
    elif option == "🤖 Prediction":
        st.title("🤖 Prediction")
        st.write("### 🔮 Make Predictions with New Dataset")
        train_data = load_dataset("Prediction (Training Data)")
        if train_data is not None:
            st.write("### 📋 Training Dataset Overview")
            st.dataframe(train_data.head())
            target = st.selectbox("🎯 Select Target Variable (Training):", train_data.columns)
            features = train_data.drop(columns=[target])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = train_data[target]
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
            st.success("Model trained successfully!")

            pred_data = load_dataset("Prediction (New Data)")
            if pred_data is not None:
                st.write("### 📋 Prediction Dataset Overview")
                st.dataframe(pred_data.head())
                pred_data_imputed = SimpleImputer(strategy="median").fit_transform(pred_data.select_dtypes(include=['float64', 'int64']))
                pred_data_imputed_df = pd.DataFrame(pred_data_imputed, columns=X.columns, index=pred_data.index)

                predictions = model.predict(pred_data_imputed_df)
                pred_data['Prediction'] = predictions
                st.write("### 🔮 Prediction Results")
                st.dataframe(pred_data)

                st.write("### 📈 Prediction Visualization")
                fig_pred = px.bar(pred_data, x=pred_data.index, y='Prediction', title="Prediction Results")
                st.plotly_chart(fig_pred)

    # Cross-validation
    elif option == "📊 Cross-validation":
        st.title("📊 Cross-validation")
        st.write("### 🧪 Evaluate Model with Cross-Validation")
        data = load_dataset("Cross-validation")
        if data is not None :
            st.write("### 📋 Dataset Overview")
            st.dataframe(data.head())
            target = st.selectbox("🎯 Select Target Variable:", data.columns)
            features = data.drop(columns=[target])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = data[target]

            # Using cross-validation to evaluate the model
            model = LogisticRegression(max_iter=1000, random_state=42)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cross_val_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

            st.write("### 📊 Cross-Validation Results")
            st.metric("🎯 Average Accuracy", f"{np.mean(cross_val_scores):.2f}")
            st.text(f"📋 Scores from each fold: {cross_val_scores}")
            
            # Displaying the distribution of cross-validation results
            st.write("### 📊 Cross-Validation Score Distribution")
            fig_cv, ax = plt.subplots()
            ax.hist(cross_val_scores, bins=5, edgecolor='black')
            ax.set_title("Cross-Validation Accuracy Distribution")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Frequency")
            st.pyplot(fig_cv)


    # Download & Export Model
    elif option == "📈 Download & Export Model":
        st.title("📈 Download & Export Model")
        st.write("### 💾 Export Your Trained Model")
        if st.button("Export Model"):
            # Save the last trained model
            model_filename = "trained_model.pkl"
            pickle.dump(model, open(model_filename, 'wb'))
            st.success(f"Model successfully exported as {model_filename}")

    # Log Out
    elif option == "⚙️ Log Out":
        st.session_state['logged_in'] = False
        st.success("Logged out successfully.", icon="✅")
