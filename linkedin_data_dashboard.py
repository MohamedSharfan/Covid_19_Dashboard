import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt



url = "LinkedIn_Post_Analytics_Data(Sheet1).csv"

df = pd.read_csv(url)

st.subheader("Data Preview")
st.write(df.head())
# print(df.columns)
print(df.isnull().sum())

df["Impression_Rolling"] = df["Impressions"].rolling(window=7).mean()
df = df.sort_values("Month", ascending=True)

st.subheader("LinkedIn Impresssion Preview")
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='Month', y='Impression_Rolling', label='7 months of linkedin impressions rolling', color='skyblue')
sns.lineplot(data=df, x='Month', y='Impressions', label='linkedin impressions', color='red')
plt.title("LinkedIn Impressions Over Time")
plt.xlabel("Month")
plt.ylabel("Impressions")
plt.legend(title='Legend', loc='upper left')
plt.grid(True)
plt.show()

st.pyplot(plt)