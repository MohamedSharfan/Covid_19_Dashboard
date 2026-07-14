import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format




training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")



print(training_df.describe())