import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


st.title("COVID-19 Dashboard")

url = "covid_19_clean_complete.csv"
st.success("Data loaded successfully!")
df = pd.read_csv(url, parse_dates=["Date"])
print(df.head())

print(df.isnull().sum())
df["Province/State"] = df["Province/State"].fillna("unknown")
df_filtered = df[df["Date"] > "2020-01-01"]

st.subheader("Data Preview")
st.write(df.head())



country_total = df_filtered.groupby("Country/Region")["Confirmed"].max().reset_index()
country_total = country_total.sort_values(by="Confirmed", ascending=False)

st.subheader("Confirmed Cases Barchart")

plt.figure(figsize=(10,6))
sns.barplot(data = country_total.head(10), x="Country/Region", y="Confirmed")
plt.title("Top 10 Countries by Confirmed COVID-19 Cases")
plt.xlabel("Country/Region")
plt.ylabel("Confirmed")
plt.show()

st.pyplot(plt)

lanka = df[df["Country/Region"] == "Sri Lanka"]
lanka_grouped = lanka.groupby("Date")["Confirmed"].sum().reset_index()

st.subheader("COVID-19 Confirmed Cases in Sri Lanka")

plt.figure(figsize=(10,6))
sns.lineplot(data=lanka_grouped, x="Date", y="Confirmed")
plt.title("COVID-19 Confirmed Cases Trend in Sri Lanka")
plt.xlabel("Date")
plt.ylabel("Confimed Cases")
plt.show()

st.pyplot(plt)