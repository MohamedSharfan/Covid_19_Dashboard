import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = "covid_19_clean_complete.csv"
print("Data loaded successfully")
df = pd.read_csv(url, parse_dates=["Date"])
print(df.head())

print(df.isnull().sum())
df["Province/State"] = df["Province/State"].fillna("unknown")

df_filtered = df[df["Date"] > "2020-01-01"]

country_total = df_filtered.groupby("Country/Region")["Confirmed"].max().reset_index()
country_total = country_total.sort_values(by="Confirmed", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data = country_total.head(10), x="Country/Region", y="Confirmed")
plt.title("Top 10 Countries by Confirmed COVID-19 Cases")
plt.xlabel("Country/Region")
plt.ylabel("Confirmed")
plt.show()

lanka = df[df["Country/Region"] == "Sri Lanka"]
lanka_grouped = lanka.groupby("Date")["Confirmed"].sum().reset_index()

plt.figure(figsize=(10,6))
sns.lineplot(data=lanka_grouped, x="Date", y="Confirmed")
plt.title("COVID-19 Confirmed Cases Trend in Sri Lanka")
plt.xlabel("Date")
plt.ylabel("Confimed Cases")
plt.show()
