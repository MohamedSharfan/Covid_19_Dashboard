import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



url = r"E:\ML\covid_19_clean_complete.csv"
print("Data loaded successfully")

df = pd.read_csv(url, parse_dates=["Date"])
# print(df.shape) # to find rows and column
# print(df.info()) #data types and nulls
# print(df.columns) #all column names
# print(df.head()) #first 5 rows
df.isnull().sum()
df["Province/State"] = df["Province/State"].fillna("Unknown") #option 1 fill nan

# df_cleaned = df.dropna() #option 2 drop rows with missing data
df = df.drop("Active", axis=1)

df["Activve"] = df["Confirmed"] - df["Deaths"] - df["Recovered"]
# print(df["Recovered"].isnull().sum())
india = df[df["Country/Region"] == 'India']
print(india["Confirmed"].sum())

df["Mortality_rate"] = df["Deaths"] / df["Confirmed"]

# pivot = df.pivot_table(values="Confirmed", columns="Country/Region", index="Date", aggfunc="sum" )
# print(pivot.head())

# countries = df.groupby("Country/Region")["Confirmed"].max().reset_index()
# print(countries.sort_values(by="Confirmed", ascending=False).head())


# jan_data = df[df["Date"] > "2020-01-22" ]
# plt.figure(figsize=(8, 5))
# sns.histplot(data = jan_data, x="Confirmed", bins= 30, kde=True)
# plt.title("date after 2020")
# plt.xlabel("Confirmed")
# plt.ylabel("Frequency")
# plt.show() 

# deaths_afghanistan = df[df["Deaths"]]
# print(deaths_afghanistan) 

# plt.figure(figsize=(8,5))
# sns.scatterplot(data=df[df["Country/Region"] == "Sri Lanka"], x="Date",y="Confirmed", hue="Deaths")
# plt.title("CORONA Confirmed people in Sri lanka")
# plt.xlabel("Confirmed")
# plt.ylabel("Frequency")
# plt.show()