#pip install streamlit

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sepsis Prediction Viewer", layout="wide")

# Load the CSVs
@st.cache_data
def load_data():
    df = pd.read_csv("bart_subset.csv")
    probs = pd.read_csv("bart_predictions.csv")["BART_Prob"]
    return df, probs

# Load
df, bart_probs = load_data()

# Add predictions
df = df.copy()
df["BART_Prob"] = bart_probs
df["BART_Pred"] = (df["BART_Prob"] > 0.11).astype(int)  # Tuned threshold

# UI: Title
st.title("Sepsis Prediction Viewer")

# Sidebar: Row index selector
index = st.slider("Choosed patient row", 0, len(df) - 1, 0)
row = df.iloc[index]

# Display selected features
cols_to_show = ['patient', 'time', 'HR', 'Temp', 'MAP', 'Resp', 'O2Sat',
                'WBC', 'Platelets', 'Age', 'Gender']
st.subheader(f"Patient #{int(row['patient'])} (Time: {int(row['time'])})")
st.dataframe(row[cols_to_show].to_frame().T)

# Display prediction
st.markdown(f"###BART Probability of Sepsis: `{row['BART_Prob']:.4f}`")
if row["BART_Pred"] == 1:
    st.error("Prediction for HIGH Risk of Sepsis")
else:
    st.success("Prediction for LOW Risk of Sepsis")

# Optional: Show ground truth
if st.checkbox("Show true label"):
    label = int(row["SepsisLabel"])
    if label == 1:
        st.warning("SEPSIS (Positive) TLabel")
    else:
        st.info("Non-Sepsis (Negative) TLabel")

# Optional: Download row
if st.button("Download this row as CSV"):
    row.to_frame().T.to_csv("selected_row.csv", index=False)
    st.success("Saved as selected_row.csv")


#Try to run the App