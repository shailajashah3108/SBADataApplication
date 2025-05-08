
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dmba import plotDecisionTree, regressionSummary, classificationSummary, liftChart, gainsChart
import io
from contextlib import redirect_stdout
import streamlit as st
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import math

def plot_confusion_matrix_small(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    z = cm.tolist()
    x = labels
    y = labels

    fig = ff.create_annotated_heatmap(
        z,
        x=x, y=y,
        annotation_text=[[f"{val}" for val in row] for row in z],
        colorscale="Viridis",
        showscale=False
    )
    fig.update_layout(
        title_text="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=300,
        height=250,
        font=dict(size=12),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

# App layout configuration
st.set_page_config(page_title="MSBA Capstone Project – CSU Sacramento", layout="wide")

# Custom top spacing and top credit fix
st.markdown("""
<style>
    .block-container {
        padding-top: 3rem !important;
    }
    .top-credit {
        text-align: center;
        font-size: 12px;
        color: gray;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
        word-wrap: break-word;
        white-space: normal;
        max-width: 100%;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Credit and Title
st.markdown("<div class='top-credit'>MSBA Capstone Project - California State University, Sacramento</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: navy; padding-top: 0.25rem'>📚 Small Business Administration (SBA) Loans Dataset Case Study</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>An end-to-end capstone in business analytics — using SBA loan data to uncover patterns, build predictive models, and guide decisions on potential loan defaults.</p>",
    unsafe_allow_html=True
)

# Navigation
st.markdown("---")
st.markdown("### 🔎 <u>Select Analysis Phase</u>", unsafe_allow_html=True)

choice = st.radio(
    "Navigate through the SBA loan analytics journey:",
    [
        "🧾 SBA Loan Risk Problem Statement",
        "Step 1: Clean & Prepare Loan Data",
        "Step 2: Explore Trends & Distributions",
        "Step 3: Train Models to Predict Default",
        "Step 4: Apply Best Model & Make Decisions"
    ],
    index=0,
    key="nav_radio"
)
st.markdown("---")

# Phase 1: Data Preparation
if choice == "Step 1: Clean & Prepare Loan Data":
    st.markdown("### 🧹 Step 1: Clean & Prepare Loan Data")
    raw_file = st.file_uploader("Upload raw dataset (CSV)", type="csv")
    if raw_file:
        df = pd.read_csv(raw_file)
        st.write("🔍 Raw Data Preview", df.head(3))
        df.shape

        st.write("### 🧾 Column-wise Missing Value Summary")
        st.write("_Below is the count of missing values per column to guide data cleaning decisions._")

        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing Values']
        st.dataframe(
            missing_df,
            use_container_width=False,  # Optional: set to True if full-width is okay
            height=350
        )

        # Format Dates
        date_col = ['ApprovalDate', 'ChgOffDate','DisbursementDate']
        df[date_col] = pd.to_datetime(df[date_col].stack(),format='%d-%b-%y').unstack()

        df['ApprovalFY'].replace('1976A', 1976, inplace=True)
        df['ApprovalFY']= df['ApprovalFY'].astype(int)

        # Remove $ from currency
        curr_col = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
        df[curr_col] = df[curr_col].replace(r'[\$,]', '', regex=True).astype(float)

        st.markdown("✅ Standardized date columns and cleaned financial fields (`$`, `,`) for numeric conversion.")

        # Convert Output Variable as 1 and 0
        df['MIS_Status'] = df['MIS_Status'].replace({'P I F': 0, 'CHGOFF':1})
        df = df.dropna(subset=['MIS_Status'])
        df = df.astype({'MIS_Status':'int64'})

        st.markdown("✅ Transformed `MIS_Status` into a binary target variable for predictive modeling.")

        ind_code = df['NAICS']

        def get_code(ind_code):
            if ind_code <= 0:
                return ''
            
            return (ind_code // 10 ** (int(math.log(ind_code, 10)) - 1))


        def sector_name(i):
            
            def_code = {11:'Agriculture, Forestry, Fishing & Hunting', 21:'Mining, Quarying, Oil & Gas',
                        22:'Utilities', 23:'Constuction', 31:'Manufacturing', 32:'Manufacturing', 33:'Manufacturing',
                        42:'Wholesale Trade', 44:'Retail Trade', 45:'Retail Trade', 48:'Transportation & Warehousing',
                        49:'Transportation & Warehousing', 51:'Information', 52:'Finance & Insurance', 
                        53:'Real Estate, Rental & Leasing', 54:'Professional, Scientific & Technical Service',
                        55:'Management of Companies & Enterprise', 
                        56:'Administrative, Support, Waste Management & Remediation Service',
                        61:'Educational Service', 62:'Health Care & Social Assistance',
                        71:'Arts, Entertainment & Recreation', 72:'Accomodation & Food Service',
                        81:'Other Servieces (Ex: Public Administration)', 92:'Public Administration'
                    }
            
            if i in def_code:
                return def_code[i]

        df.loc[:, 'ind_code'] = df.NAICS.apply(get_code)
        df.loc[:, 'Sector_name'] = df.ind_code.apply(sector_name)

        # Replace 1.0 as 0 and 2.0 as 1 in NewBusiness and replace others as nan
        df['NewExist'] = np.where((df['NewExist'] != 1.0) & (df['NewExist'] != 2.0), 
                                        np.nan, df.NewExist)
        df['NewExist'] = df['NewExist'].replace({1.0: 0, 2.0:1})
        df['NewExist'] = df['NewExist'].astype('Int64')

        # Replace [0,N] to 0 and [1,Y] to 1 and replace others as nan
        df['LowDoc'] = df['LowDoc'].replace(['C', 'S', 'A', 'R', 1, 0], np.nan)
        df['LowDoc'] = df['LowDoc'].replace({'N': 0, 'Y':1})
        df['LowDoc'] = np.where((df['LowDoc'] != 0) & (df['LowDoc'] != 1), 
                                    np.nan, df.LowDoc)

        # LowDoc where DisbursementGross < 150000 can be treated as 'Yes' where one page loan application is needed
        df['LowDoc'] = np.where(
            (pd.isna(df['LowDoc'])) & (df['DisbursementGross'] < 150000), 
            1, 
            np.where(
                (pd.isna(df['LowDoc'])) & (df['DisbursementGross'] >= 150000), 
                0, 
                df['LowDoc']
            )
        )
        df['LowDoc'] = df['LowDoc'].astype('Int64')

        df['RevLineCr'] = df['RevLineCr'].replace({'N': 0, 'Y':1, '1': 1, '0': 0 })
        df['RevLineCr'] = np.where((df['RevLineCr'] != 0) & (df['RevLineCr'] != 1), 
                                        np.nan, df.RevLineCr)
        df['RevLineCr'] = df['RevLineCr'].astype('Int64')

        st.markdown("✅ Handled missing values in `LowDoc`, `NewExist` and `RevLineCr` based on domain logic")

        df = df.drop(['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'BankState', 'NAICS', 'ApprovalDate', 'ApprovalFY', 'FranchiseCode', 'ChgOffDate', 'BalanceGross', 'ChgOffPrinGr', 'ind_code'], axis=1)

        # Derived Variable RealEstate
        df['RealEstate'] = df['Term'].apply(lambda x: 1 if x >= 240 else 0)

        st.markdown("✅ Created `RealEstate` feature for loan terms ≥ 240 months.")
  
        # Derived Variable Recession
        df['DisbursementDate'] = pd.to_datetime(df['DisbursementDate'], errors='coerce')
        df['DaysTerm'] =  df['Term']*30
        df['Active'] = df['DisbursementDate'] + pd.TimedeltaIndex(df['DaysTerm'], unit='D')
        df['Active'] = pd.to_datetime(df['Active'], errors='coerce').dt.date
        startdate = datetime.datetime.strptime('2007-12-1', "%Y-%m-%d").date()
        enddate = datetime.datetime.strptime('2009-06-30', "%Y-%m-%d").date()
        df['Recession'] = df['Active'].apply(lambda x: 1 if pd.notnull(x) and startdate <= x <= enddate else 0)

        st.markdown("✅ Added `Recession` indicator for loans active during 2007–2009.")

        # Derived Variable Portion
        df['Portion'] = df['SBA_Appv']/df['GrAppv']

        st.markdown("✅ Calculated `Portion` as SBA_Appv / GrAppv to measure guarantee level.")

        # Filter loans with DisbursementDate before 2010
        filtered_df = df[df['DisbursementDate'] <= '2010-12-31']

        st.markdown("✅ Filtered out loans disbursed after 2010 to ensure known outcomes only.")
        
        # Drop irrelevant columns
        filtered_df = filtered_df.drop(['DisbursementDate', 'Term', 'GrAppv', 'SBA_Appv', 'DaysTerm', 'Active'], axis=1)

        st.markdown("✅ Dropped irrelevant or redundant columns (e.g., IDs, names, unused dates).")

        # Define the subset of columns to drop missing values rows
        subset_columns = ['NewExist', 'RevLineCr']
        filtered_df = filtered_df.dropna(subset=subset_columns)


        st.markdown("✅ Identifying the correlations to avoid multicollinearity.")
        # Select only numeric columns
        numeric_columns = filtered_df[['NoEmp', 'CreateJob', 'DisbursementGross', 'RetainedJob', 'Portion']]

        correlation_matrix = numeric_columns.corr()
        # Create compact figure
        # Create a small, sharp figure
        # Round values for cleaner display
        rounded_corr = np.round(correlation_matrix.values, 1)

        # Assuming correlation_matrix is your DataFrame
        # Assuming correlation_matrix is your DataFrame
        fig = px.imshow(
            correlation_matrix,
            text_auto=".1f",
            color_continuous_scale="Blues",
            aspect="auto"
        )

        fig.update_layout(
            title="Correlation Matrix",
            title_x=0.0,
            margin=dict(l=5, r=5, t=30, b=5),
            width=550,              # Compact width
            height=350,             # Smaller height
            font=dict(size=12),     # Bigger font for labels
        )

        fig.update_xaxes(
            tickangle=45,
            tickfont=dict(size=11),
            side="bottom"
        )
        fig.update_yaxes(
            tickfont=dict(size=11)
        )

        # Render it neatly aligned
        st.plotly_chart(fig, use_container_width=False)

        # Identify strong positive correlations
        strong_corr_pos = correlation_matrix[(correlation_matrix >= 0.8) & (correlation_matrix < 1.0)]
        strong_corr_unstacked_pos = strong_corr_pos.unstack().dropna()
        strong_positive_corr = strong_corr_unstacked_pos[~strong_corr_unstacked_pos.duplicated()]

        # Convert to DataFrame for better formatting
        strong_positive_corr_df = strong_positive_corr.reset_index()
        strong_positive_corr_df.columns = ["Feature 1", "Feature 2", "Correlation"]

        # Display in Streamlit
        st.markdown("✅ **Identifying Strong Positive Correlations**")
        st.dataframe(strong_positive_corr_df, use_container_width=False)

        # Identify strong negative correlations
        strong_corr_neg = correlation_matrix[(correlation_matrix <= -0.6) & (correlation_matrix > -1.0)]
        strong_corr_unstacked_neg = strong_corr_neg.unstack().dropna()
        strong_negative_corr = strong_corr_unstacked_neg[~strong_corr_unstacked_neg.duplicated()]

        # Convert to DataFrame for better formatting
        strong_negative_corr_df = strong_negative_corr.reset_index()
        strong_negative_corr_df.columns = ["Feature 1", "Feature 2", "Correlation"]

        # Display in Streamlit
        st.markdown("✅ **Identifying Strong Negative Correlations**")
        st.dataframe(strong_negative_corr_df, use_container_width=False)

        filtered_df = filtered_df.drop(['RetainedJob'], axis=1)

        st.markdown("✅ Dropped Columns due to Multicollinearity")

        # Define target and non-predictive columns
        target_column = "MIS_Status"
        non_predictive = ["State", "Sector_name"]

        # Get predictor columns
        predictor_columns = [col for col in filtered_df.columns if col not in [target_column] + non_predictive]

        # Convert to DataFrame with one column
        predictor_df = pd.DataFrame(predictor_columns, columns=["Selected Predictors"])

        # Show in Streamlit as a table
        st.markdown("🧮 **Selected Predictors for Modeling**")
        st.dataframe(predictor_df)

        st.write("🔍 Cleaned Data Preview", filtered_df.head(3))
        filtered_df.shape

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned Dataset", csv, "cleaned_data.csv", "text/csv")

# Phase 2: EDA
elif choice == "Step 2: Explore Trends & Distributions":
    st.markdown("### 📊 Step 2: Explore Trends & Distributions")
    st.markdown("Explore Tableau dashboard insights for trends and summaries:")
    st.markdown(
    '[🔗 Dive Deeper: SBA Loan Trends & Defaults](https://public.tableau.com/views/SBATableau/Dashboard1?:embed=yes&:toolbar=no&:fullscreen=true)',
    unsafe_allow_html=True
    )

# Phase 3: Predictive Modeling
elif choice == "Step 3: Train Models to Predict Default":
    st.markdown("### 🔍 Step 3: Train Models to Predict Default")
    clean_file = st.file_uploader("Upload cleaned dataset (CSV)", type="csv", key="predict")
    if clean_file:
        data = pd.read_csv(clean_file)
        st.write("📋 Cleaned Data Preview", data.head(3))

        # Select Target and Predictors
        target = st.selectbox("🎯 Select Target Variable", data.columns)
        features = st.multiselect("🧮 Select Predictor Variables", [col for col in data.columns if col != target])

        if features:
            # Checkboxes for models
            st.subheader("📌 Select Models to Run")
            run_logit = st.checkbox("Logistic Regression")
            run_tree = st.checkbox("Classification Tree")
            run_lda = st.checkbox("Discriminant Analysis")

            if st.button("🚀 Run Selected Models"):
                model_summary = []

                if run_logit:
                    model_name = "Logistic Regression"
                    st.success("✅ Logistic Regression Trained")
                    # Prepare data
                    X = data.drop(['State', 'Sector_name', 'MIS_Status'], axis=1)
                    y = data[target]
                    X = pd.get_dummies(X, columns=["UrbanRural"], drop_first=True, dtype=int)
                    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=1)
                    logit_model = LogisticRegression(penalty="l2", C=1e42, solver='liblinear', random_state=1, tol=1e-20, max_iter=10000, class_weight={0:1, 1: 5})
                    logit_model.fit(X_train, y_train)
                    preds = logit_model.predict_proba(X_valid)
                    success_probability = logit_model.predict_proba(X_valid)[:,1]
                    predictions_nominal = [ 0 if x < 0.4 else 1 for x in success_probability]
                    full_result = pd.DataFrame({'actual': y_valid, 
                                                'p(0)': [p[0] for p in preds],
                                                'p(1)': [p[1] for p in preds],
                                                'predicted': predictions_nominal})
                    report = classification_report(full_result['actual'], full_result['predicted'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(2)
                    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]

                    # Highlight recall and F1-score for class '1'
                    def highlight_class_1(s):
                        is_target = s.name == '1'
                        return ['background-color: yellow' if is_target and col in ['recall', 'f1-score'] else '' for col in s.index]

                    st.markdown(f"### 📊 {model_name} Report")
                    st.dataframe(report_df.style.apply(highlight_class_1, axis=1).format("{:.2f}"))

                    # Save values to summary
                    recall = report['1']['recall'] if '1' in report else 0
                    f1 = report['1']['f1-score'] if '1' in report else 0

                    # Confusion matrix
                    cm = confusion_matrix(full_result.actual, full_result.predicted)
                    fig = plot_confusion_matrix_small(full_result.actual, full_result.predicted, labels=['PIF', 'Default'])
                    st.plotly_chart(fig, use_container_width=False)

                    # --- Cost/Gain Matrix Calculation ---
                    st.markdown("💰 Cost / Gain Matrix")

                    # Unpack confusion matrix values
                    tn, fp, fn, tp = confusion_matrix(full_result.actual, full_result.predicted).ravel()

                    # Pull disbursement and portion data
                    disbursement_gross = X_valid['DisbursementGross']
                    loss_portion = 1 - X_valid['Portion']

                    # Calculate cost/gain
                    gain_tn = disbursement_gross[(full_result['actual'] == 0) & (full_result['predicted'] == 0)].sum() * 0.05
                    loss_fn_temp = np.where(
                        (full_result['actual'] == 1) & (full_result['predicted'] == 0),
                        full_result['p(1)'] * loss_portion * disbursement_gross,
                        0
                    )
                    loss_fn = loss_fn_temp.sum()
                    net_profit = gain_tn - loss_fn

                    # Raw numeric values for the table
                    cost_gain_df = pd.DataFrame({
                        "Outcome": ["True Negative (PIF Correct)", "False Negative (Missed Default)"],
                        "Count": [tn, fn],
                        "Effect": ["+5% of Disbursement", "-p(1) x (1 - Portion) x Disbursement"],
                        "Value ($)": [gain_tn, -loss_fn]
                    })

                    # Function to format billions
                    def format_billions(val):
                        return f"${val / 1e9:,.2f}B"

                    # Display table with styled formatting
                    st.dataframe(
                        cost_gain_df.style.format({"Value ($)": format_billions, "Count": "{:,.0f}"})
                    )

                    # Display formatted net profit
                    net_display = f"${net_profit / 1e9:,.2f}B"
                    color = "green" if net_profit >= 0 else "red"
                    st.markdown(f"**🧾 Net Profit Estimate using Logistic Regression model :** <span style='color:{color}'>{net_display}</span>", unsafe_allow_html=True)
                    model_summary.append({'Model': model_name, 'Recall': recall, 'F1 Score': f1, 'Net Profit': net_profit})
                    st.session_state['logit_model'] = logit_model

                if run_tree:
                    model_name = "Classification Tree"
                    st.success("✅ Classification Tree Trained")
                     # Prepare data
                    X = data.drop(['State', 'Sector_name', 'MIS_Status'], axis=1)
                    y = data[target]
                    X = pd.get_dummies(X, columns=["UrbanRural"], drop_first=False, dtype=int)
                    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=1)
                    tree_model = DecisionTreeClassifier(criterion="gini", max_depth=12, min_samples_split=105, min_impurity_decrease=0)
                    tree_model.fit(X_train, y_train)
                    preds = tree_model.predict_proba(X_valid)
                    success_probability = tree_model.predict_proba(X_valid)[:,1]
                    predictions_nominal = [ 0 if x < 0.1 else 1 for x in success_probability]
                    full_result = pd.DataFrame({'actual': y_valid,
                                                'p(0)': [p[0] for p in preds],
                                                'p(1)': [p[1] for p in preds],
                                                'predicted': predictions_nominal})
                    report = classification_report(full_result['actual'], full_result['predicted'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(2)
                    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]

                    # Highlight recall and F1-score for class '1'
                    def highlight_class_1(s):
                        is_target = s.name == '1'
                        return ['background-color: yellow' if is_target and col in ['recall', 'f1-score'] else '' for col in s.index]

                    st.markdown(f"### 📊 {model_name} Report")
                    st.dataframe(report_df.style.apply(highlight_class_1, axis=1).format("{:.2f}"))

                    # Save values to summary
                    recall = report['1']['recall'] if '1' in report else 0
                    f1 = report['1']['f1-score'] if '1' in report else 0

                    # Confusion matrix
                    cm = confusion_matrix(full_result.actual, full_result.predicted)
                    fig = plot_confusion_matrix_small(full_result.actual, full_result.predicted, labels=['PIF', 'Default'])
                    st.plotly_chart(fig, use_container_width=False)

                    # --- Cost/Gain Matrix Calculation ---
                    st.markdown("💰 Cost / Gain Matrix")

                    # Unpack confusion matrix values
                    tn, fp, fn, tp = confusion_matrix(full_result.actual, full_result.predicted).ravel()

                    # Pull disbursement and portion data
                    disbursement_gross = X_valid['DisbursementGross']
                    loss_portion = 1 - X_valid['Portion']

                    # Calculate cost/gain
                    gain_tn = disbursement_gross[(full_result['actual'] == 0) & (full_result['predicted'] == 0)].sum() * 0.05
                    loss_fn_temp = np.where(
                        (full_result['actual'] == 1) & (full_result['predicted'] == 0),
                        full_result['p(1)'] * loss_portion * disbursement_gross,
                        0
                    )
                    loss_fn = loss_fn_temp.sum()
                    net_profit = gain_tn - loss_fn

                    # Raw numeric values for the table
                    cost_gain_df = pd.DataFrame({
                        "Outcome": ["True Negative (PIF Correct)", "False Negative (Missed Default)"],
                        "Count": [tn, fn],
                        "Effect": ["+5% of Disbursement", "-p(1) x (1 - Portion) x Disbursement"],
                        "Value ($)": [gain_tn, -loss_fn]
                    })

                    # Function to format billions
                    def format_billions(val):
                        return f"${val / 1e9:,.2f}B"

                    # Display table with styled formatting
                    st.dataframe(
                        cost_gain_df.style.format({"Value ($)": format_billions, "Count": "{:,.0f}"})
                    )

                    # Display formatted net profit
                    net_display = f"${net_profit / 1e9:,.2f}B"
                    color = "green" if net_profit >= 0 else "red"
                    st.markdown(f"**🧾 Net Profit Estimate using Classification Tree model :** <span style='color:{color}'>{net_display}</span>", unsafe_allow_html=True)
                    model_summary.append({'Model': model_name, 'Recall': recall, 'F1 Score': f1, 'Net Profit': net_profit})
                    st.session_state['tree_model'] = tree_model

                if run_lda:
                    model_name = "Discriminant Analysis"
                    st.success("✅ Discriminant Analysis Trained")
                    # Prepare data
                    X = data.drop(['State', 'Sector_name', 'MIS_Status'], axis=1)
                    y = data[target]
                    X = pd.get_dummies(X, columns=["UrbanRural"], drop_first=True, dtype=int)
                    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=1)
                    lda_model = LinearDiscriminantAnalysis(priors=[0.48518302716704786, 0.5148169728329521])
                    lda_model.fit(X_train, y_train)
                    preds = lda_model.predict_proba(X_valid)
                    success_probability = lda_model.predict_proba(X_valid)[:,1]
                    predictions_nominal = [ 0 if x < 0.4 else 1 for x in success_probability]
                    full_result = pd.DataFrame({'actual': y_valid,
                                                'p(0)': [p[0] for p in preds],
                                                'p(1)': [p[1] for p in preds],
                                                'predicted': predictions_nominal})
                    report = classification_report(full_result['actual'], full_result['predicted'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(2)
                    report_df = report_df[['precision', 'recall', 'f1-score', 'support']]

                    # Highlight recall and F1-score for class '1'
                    def highlight_class_1(s):
                        is_target = s.name == '1'
                        return ['background-color: yellow' if is_target and col in ['recall', 'f1-score'] else '' for col in s.index]

                    st.markdown(f"### 📊 {model_name} Report")
                    st.dataframe(report_df.style.apply(highlight_class_1, axis=1).format("{:.2f}"))

                    # Save values to summary
                    recall = report['1']['recall'] if '1' in report else 0
                    f1 = report['1']['f1-score'] if '1' in report else 0

                    # Confusion matrix
                    cm = confusion_matrix(full_result.actual, full_result.predicted)
                    fig = plot_confusion_matrix_small(full_result.actual, full_result.predicted, labels=['PIF', 'Default'])
                    st.plotly_chart(fig, use_container_width=False)

                    # --- Cost/Gain Matrix Calculation ---
                    st.markdown("💰 Cost / Gain Matrix")

                    # Unpack confusion matrix values
                    tn, fp, fn, tp = confusion_matrix(full_result.actual, full_result.predicted).ravel()

                    # Pull disbursement and portion data
                    disbursement_gross = X_valid['DisbursementGross']
                    loss_portion = 1 - X_valid['Portion']

                    # Calculate cost/gain
                    gain_tn = disbursement_gross[(full_result['actual'] == 0) & (full_result['predicted'] == 0)].sum() * 0.05
                    loss_fn_temp = np.where(
                        (full_result['actual'] == 1) & (full_result['predicted'] == 0),
                        full_result['p(1)'] * loss_portion * disbursement_gross,
                        0
                    )
                    loss_fn = loss_fn_temp.sum()
                    net_profit = gain_tn - loss_fn

                    # Raw numeric values for the table
                    cost_gain_df = pd.DataFrame({
                        "Outcome": ["True Negative (PIF Correct)", "False Negative (Missed Default)"],
                        "Count": [tn, fn],
                        "Effect": ["+5% of Disbursement", "-p(1) x (1 - Portion) x Disbursement"],
                        "Value ($)": [gain_tn, -loss_fn]
                    })

                    # Function to format billions
                    def format_billions(val):
                        return f"${val / 1e9:,.2f}B"

                    # Display table with styled formatting
                    st.dataframe(
                        cost_gain_df.style.format({"Value ($)": format_billions, "Count": "{:,.0f}"})
                    )

                    # Display formatted net profit
                    net_display = f"${net_profit / 1e9:,.2f}B"
                    color = "green" if net_profit >= 0 else "red"
                    st.markdown(f"**🧾 Net Profit Estimate using Discriminant model:** <span style='color:{color}'>{net_display}</span>", unsafe_allow_html=True)
                    model_summary.append({'Model': model_name, 'Recall': recall, 'F1 Score': f1, 'Net Profit': net_profit})
                    st.session_state['lda_model'] = lda_model

                # Display Summary Table
                if model_summary:
                    st.subheader("📊 Model Comparison Summary")
                    summary_df = pd.DataFrame(model_summary)

                    # Ensure 'Net Profit' is kept as numeric for comparison
                    summary_df['Net Profit Raw'] = summary_df['Net Profit']  # Duplicate for internal use

                    # Get index of row with max profit
                    max_profit_index = summary_df['Net Profit Raw'].idxmax()

                    # Highlight the max profit row
                    def highlight_max_profit(row):
                        return ['background-color: #DFF2BF' if row.name == max_profit_index else '' for _ in row]

                    # Format Net Profit into billions ($X.XXB)
                    def format_billions(val):
                        return f"${val / 1e9:,.2f}B" if pd.notnull(val) else val

                    # Apply formatting and highlighting
                    styled_df = summary_df.drop(columns=['Net Profit Raw']).style \
                        .format({
                            "Recall": "{:.2f}",
                            "F1 Score": "{:.2f}",
                            "Net Profit": format_billions
                        }) \
                        .apply(highlight_max_profit, axis=1)

                    # Display styled table
                    st.dataframe(styled_df)
                    st.session_state['model_summary'] = model_summary


# Phase 4: Evaluation
elif choice == "Step 4: Apply Best Model & Make Decisions":
    st.markdown("### 🧠 Step 4: Apply Best Model & Make Decisions ")
    # Determine best model from previous step
    model_summary = st.session_state.get('model_summary', [])
    logit_model = st.session_state.get('logit_model', [])
    tree_model = st.session_state.get('tree_model', [])
    lda_model = st.session_state.get('lda_model', [])

    if model_summary:
        best_model_entry = max(model_summary, key=lambda x: x.get('Net Profit', 0))
        best_model_name = best_model_entry['Model']
        st.success(f"Using most profitable model: **{best_model_name}** for prediction")

        # Upload new dataset
        new_data_file = st.file_uploader("📤 Upload New Loan Dataset for Prediction (CSV)", type=['csv'])
        if new_data_file:
            new_data = pd.read_csv(new_data_file)
            st.write("📋 New Data Preview", new_data.head(3))

            # Preprocessing: Drop unused columns and dummy encode
            try:
                X_new = new_data.drop(columns=['State', 'Sector_name'], axis=1)
        
                # Load and apply best model
                if best_model_name == "Logistic Regression":
                    prediction_probs = logit_model.predict_proba(X_new)[:, 1]
                    predictions = ['Not Default' if x < 0.4 else 'Default' for x in prediction_probs]

                elif best_model_name == "Classification Tree":
                    prediction_probs = tree_model.predict_proba(X_new)[:, 1]
                    predictions = ['Not Default' if x < 0.1 else 'Default' for x in prediction_probs]

                elif best_model_name == "Discriminant Analysis":
                    prediction_probs = lda_model.predict_proba(X_new)[:, 1]
                    predictions = ['Not Default' if x < 0.4 else 'Default' for x in prediction_probs]

                # Add predictions to new data
                new_data['Predicted_Status'] = predictions
                new_data['p(1)'] = prediction_probs

                st.success("✅ Predictions added to dataset!")
                st.dataframe(new_data.head(3))

                # Enable CSV download
                csv_output = new_data.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions CSV", data=csv_output, file_name="loan_predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    else:
        st.warning("Please run a predictive model first in section 3.")

elif choice == "🧾 SBA Loan Risk Problem Statement":
    st.markdown("### 💡 SBA Loan Default Classification Problem")

    st.markdown("""
    <br>

    ### 📄 **Objective:**  
    Decide whether a new SBA loan application should be **approved or rejected**, based on patterns in past loan data.

    ### 💥 **Why it matters:**  
    Misclassifying a risky loan as low risk (called a **False Negative**) leads to **5x more financial loss** than rejecting a safe loan (False Positive).

    ### ⚖️ **Challenge:**  
    We must find the **right trade-off** between:  
    ➤ **Profitability** (approve more loans and earn returns)  
    ➤ **Risk detection** (catch defaults with high recall)

    ### 🏁 **Your goal:**  
    Build a model that maximizes **Net Profit** while maintaining **high Recall** for defaults.
    """, unsafe_allow_html=True)
        
