#data
import numpy as np
import pandas as pd

#ml
import keras
import ml_edu.experiment
import ml_edu.results

import plotly.express as px




#load the dataset 
taxi_ds = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

training_df = taxi_ds.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

print("Total number of rows: {0}\n\n".format(len(training_df)))
training_df.head(200)
print(training_df.describe(include = 'all'))

max_fare = training_df['FARE'].max()
print("maximuum fare: ${fare:.2f}".format(fare = max_fare))

mean_distance = training_df['TRIP_MILES'].mean()
print("mean distance : ${distance:.2f}".format(distance = mean_distance))

num_unique_companies = training_df['COMPANY'].nunique()
print("num_unique_companies : ${distance}".format(distance = num_unique_companies))

most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("most_freq_payment_type : ${type}".format(type = most_freq_payment_type))

missing_values = training_df.isnull().sum().sum()
print("missing value: ", "No" if missing_values == 0 else "yes")

#view correlation matrix
print(training_df.corr(numeric_only = True))

#view pairplot
#Sometimes it is helpful to visualize relationships 
#between features in a dataset; one way to do this is with a pair plot. A pair plot generates a grid of pairwise
# plots to visualize the relationship of each feature with all other features all in one place.
fig = px.scatter_matrix(training_df, dimensions = ["FARE", "TRIP_MILES", "TRIP_SECONDS"])
#fig.show()


def create_model(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric],
) -> keras.Model:
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    #linear_regression actual thing
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs,outputs=outputs)
    


    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                  loss="mean_squared_error",
                  metrics=metrics)
    
    return model



def train_model(
    experiment_name:str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    
    features = {name: dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(
        x = features,
        y=label,
        batch_size = settings.batch_size,
        epochs = settings.number_epochs
    )

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )

print("SUCCESS: defining linear regression functions complete.")


# @title Code - Experiment 1

# The following variables are the hyperparameters.
settings_1 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES']
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_1 = create_model(settings_1, metrics)

experiment_1 = train_model('one_feature', model_1, training_df, 'FARE', settings_1)

# ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
# ml_edu.results.plot_model_predictions(experiment_1, training_df, 'FARE')




setting_2 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES', 'TRIP_MINUTES']
)

training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]
model_2 = create_model(setting_2, metrics)
experiment_2 = train_model('Two_features', model_2, training_df, 'FARE', setting_2)

# ml_edu.results.plot_experiment_metrics(experiment_2, ['rmse'])
# ml_edu.results.plot_model_predictions(experiment_2, training_df, 'FARE')



#compare experiments
fig = ml_edu.results.compare_experiment([experiment_1, experiment_2], ['rmse'], training_df, training_df['FARE'].values)
print(fig)


def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return


#@title Code - Make predictions

output = predict_fare(experiment_2.model, training_df, experiment_2.settings.input_features, 'FARE')
show_predictions(output)