import os
import streamlit as st
import pandas as pd
from pandasai import Agent
from dotenv import load_dotenv
import ssl
from scipy import stats

# Disable SSL verification to avoid SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context


# Module 2: Load and combine CSV/Excel files
@st.cache_data
def load_combined_data(files):
    combined_df = pd.DataFrame()
    for file in files:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df


# Module 3: Describe Data Function
def describe_data(dataframe):
    if dataframe.empty:
        st.write("The DataFrame is empty.")
        return
    try:
        st.write("### Dataset Summary")
        st.write(dataframe.describe())  # Basic summary stats
        st.write("### Data Types")
        st.write(dataframe.dtypes)  # Data types of columns

        st.write("### Mean, Median, Mode")
        st.write(f"Mean: \n{dataframe.mean(numeric_only=True)}")
        st.write(f"Median: \n{dataframe.median(numeric_only=True)}")
        st.write(f"Mode: \n{dataframe.mode().iloc[0]}")

        st.write("### Z-Score Calculation (for numerical columns)")
        numerical_cols = dataframe.select_dtypes(include=['number'])
        if not numerical_cols.empty:
            z_scores = stats.zscore(numerical_cols)
            st.write(f"Z-Scores: \n{pd.DataFrame(z_scores, columns=numerical_cols.columns).head()}")

        st.write("### P-Value (against the column's mean)")
        for column in numerical_cols.columns:
            _, p_value = stats.ttest_1samp(dataframe[column].dropna(), dataframe[column].mean())
            st.write(f"P-Value for {column}: {p_value}")

        st.write("P-values help assess whether the observed data deviates significantly from the mean.")

    except Exception as e:
        st.error(f"Error in describing data: {e}")


# Module 4: Handle natural language queries with Agent
def handle_nlp_query(dataframe, query, agent):
    if query.lower() == "describe your data":
        describe_data(dataframe)
        return "Data described successfully."
    else:
        return agent.chat(query)


# Main Streamlit App Interface
def main():
    st.title("AI-Powered Data Analysis Chatbot")

    uploaded_files = st.file_uploader("Upload your Excel or CSV files", type=["csv", "xlsx"],
                                      accept_multiple_files=True)

    if uploaded_files:
        combined_data = load_combined_data(uploaded_files)
        st.write("Combined Data Preview:", combined_data.head())

        agent = Agent(combined_data)

        query = st.text_input("Enter your query (e.g., 'Describe your data', 'Generate a scatter plot'):")

        if st.button("Run Query"):
            if query:
                result = handle_nlp_query(combined_data, query, agent)
                st.write(f"Query Result: {result}")


if __name__ == "__main__":
    main()
