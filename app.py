import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#view first 5 rows
df.head()


sns.countplot(data=df, x='Survived')
plt.title("Survivors (1 = Yes, 0 = No)")
plt.show()

print(df.isnull().sum())

