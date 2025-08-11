import pandas as pd
import seaborn as sns
import numpy as np
# import streamlit as st
import matplotlib.pyplot as plt



url = "LinkedIn_Post_Analytics_Data.xlsx"

df = pd.read_excel(url)


print(df.head())
print(df.columns)
