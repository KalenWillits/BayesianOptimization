# %% markdown
# ## Bayesian methods of hyperparameter optimization
# %% markdown
# In addition to the random search and the grid search methods for selecting optimal hyperparameters, we can use Bayesian methods of probabilities to select the optimal hyperparameters for an algorithm.
#
# In this case study, we will be using the BayesianOptimization library to perform hyperparmater tuning. This library has very good documentation which you can find here: https://github.com/fmfn/BayesianOptimization
#
# You will need to install the Bayesian optimization module. Running a cell with an exclamation point in the beginning of the command will run it as a shell command — please do this to install this module from our notebook in the cell below.
# %% codecell
# ! pip install bayesian-optimization
# %% codecell
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import lightgbm
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier, cv, Pool
# %% codecell
import os
os.listdir()
# %% markdown
# ## How does Bayesian optimization work?
# %% markdown
# Bayesian optimization works by constructing a posterior distribution of functions (Gaussian process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, and the algorithm becomes more certain of which regions in parameter space are worth exploring and which are not, as seen in the picture below.
# %% markdown
# <img src="https://github.com/fmfn/BayesianOptimization/blob/master/examples/bo_example.png?raw=true" />
# As you iterate over and over, the algorithm balances its needs of exploration and exploitation while taking into account what it knows about the target function. At each step, a Gaussian Process is fitted to the known samples (points previously explored), and the posterior distribution, combined with an exploration strategy (such as UCB — aka Upper Confidence Bound), or EI (Expected Improvement). This process is used to determine the next point that should be explored (see the gif below).
# <img src="https://github.com/fmfn/BayesianOptimization/raw/master/examples/bayesian_optimization.gif" />
# %% markdown
# ## Let's look at a simple example
# %% markdown
# The first step is to create an optimizer. It uses two items:
# * function to optimize
# * bounds of parameters
#
# The function is the procedure that counts metrics of our model quality. The important thing is that our optimization will maximize the value on function. Smaller metrics are best. Hint: don't forget to use negative metric values.
# %% markdown
# Here we define our simple function we want to optimize.
# %% codecell
def simple_func(a, b):
    return a + b
# %% markdown
# Now, we define our bounds of the parameters to optimize, within the Bayesian optimizer.
# %% codecell
optimizer = BayesianOptimization(
    simple_func,
    {'a': (1, 3),
    'b': (4, 7)})
# %% markdown
# These are the main parameters of this function:
#
# * **n_iter:** This is how many steps of Bayesian optimization you want to perform. The more steps, the more likely you are to find a good maximum.
#
# * **init_points:** This is how many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
# %% markdown
# Let's run an example where we use the optimizer to find the best values to maximize the target value for a and b given the inputs of 3 and 2.
# %% codecell
optimizer.maximize(3,2)
# %% markdown
# Great, now let's print the best parameters and the associated maximized target.
# %% codecell
print(optimizer.max['params']);optimizer.max['target']
# %% markdown
# ## Test it on real data using the Light GBM
# %% markdown
# The dataset we will be working with is the famous flight departures dataset. Our modeling goal will be to predict if a flight departure is going to be delayed by 15 minutes based on the other attributes in our dataset. As part of this modeling exercise, we will use Bayesian hyperparameter optimization to identify the best parameters for our model.
# %% markdown
# **<font color='teal'> You can load the zipped csv files just as you would regular csv files using Pandas read_csv. In the next cell load the train and test data into two seperate dataframes. </font>**
#
# %% codecell
cd_data = 'data/'
train_df = pd.read_csv(cd_data+'flight_delays_train.csv.zip')
test_df = pd.read_csv(cd_data+'flight_delays_test.csv.zip')
# %% markdown
# **<font color='teal'> Print the top five rows of the train dataframe and review the columns in the data. </font>**
# %% codecell
train_df.head()
# %% markdown
# **<font color='teal'> Use the describe function to review the numeric columns in the train dataframe. </font>**
# %% codecell
train_df.describe()
# %% markdown
# Notice, `DepTime` is the departure time in a numeric representation in 2400 hours.
# %% markdown
#  **<font color='teal'>The response variable is 'dep_delayed_15min' which is a categorical column, so we need to map the Y for yes and N for no values to 1 and 0. Run the code in the next cell to do this.</font>**
# %% codecell
#train_df = train_df[train_df.DepTime <= 2400].copy()
y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
# %% markdown
# ## Feature Engineering
# Use these defined functions to create additional features for the model. Run the cell to add the functions to your workspace.
# %% codecell
def label_enc(df_column):
    df_column = LabelEncoder().fit_transform(df_column)
    return df_column

def make_harmonic_features_sin(value, period=2400):
    value *= 2 * np.pi / period
    return np.sin(value)

def make_harmonic_features_cos(value, period=2400):
    value *= 2 * np.pi / period
    return np.cos(value)

def feature_eng(df):
    df['flight'] = df['Origin']+df['Dest']
    df['Month'] = df.Month.map(lambda x: x.split('-')[-1]).astype('int32')
    df['DayofMonth'] = df.DayofMonth.map(lambda x: x.split('-')[-1]).astype('uint8')
    df['begin_of_month'] = (df['DayofMonth'] < 10).astype('uint8')
    df['midddle_of_month'] = ((df['DayofMonth'] >= 10)&(df['DayofMonth'] < 20)).astype('uint8')
    df['end_of_month'] = (df['DayofMonth'] >= 20).astype('uint8')
    df['DayOfWeek'] = df.DayOfWeek.map(lambda x: x.split('-')[-1]).astype('uint8')
    df['hour'] = df.DepTime.map(lambda x: x/100).astype('int32')
    df['morning'] = df['hour'].map(lambda x: 1 if (x <= 11)& (x >= 7) else 0).astype('uint8')
    df['day'] = df['hour'].map(lambda x: 1 if (x >= 12) & (x <= 18) else 0).astype('uint8')
    df['evening'] = df['hour'].map(lambda x: 1 if (x >= 19) & (x <= 23) else 0).astype('uint8')
    df['night'] = df['hour'].map(lambda x: 1 if (x >= 0) & (x <= 6) else 0).astype('int32')
    df['winter'] = df['Month'].map(lambda x: x in [12, 1, 2]).astype('int32')
    df['spring'] = df['Month'].map(lambda x: x in [3, 4, 5]).astype('int32')
    df['summer'] = df['Month'].map(lambda x: x in [6, 7, 8]).astype('int32')
    df['autumn'] = df['Month'].map(lambda x: x in [9, 10, 11]).astype('int32')
    df['holiday'] = (df['DayOfWeek'] >= 5).astype(int)
    df['weekday'] = (df['DayOfWeek'] < 5).astype(int)
    df['airport_dest_per_month'] = df.groupby(['Dest', 'Month'])['Dest'].transform('count')
    df['airport_origin_per_month'] = df.groupby(['Origin', 'Month'])['Origin'].transform('count')
    df['airport_dest_count'] = df.groupby(['Dest'])['Dest'].transform('count')
    df['airport_origin_count'] = df.groupby(['Origin'])['Origin'].transform('count')
    df['carrier_count'] = df.groupby(['UniqueCarrier'])['Dest'].transform('count')
    df['carrier_count_per month'] = df.groupby(['UniqueCarrier', 'Month'])['Dest'].transform('count')
    df['deptime_cos'] = df['DepTime'].map(make_harmonic_features_cos)
    df['deptime_sin'] = df['DepTime'].map(make_harmonic_features_sin)
    df['flightUC'] = df['flight']+df['UniqueCarrier']
    df['DestUC'] = df['Dest']+df['UniqueCarrier']
    df['OriginUC'] = df['Origin']+df['UniqueCarrier']
    return df.drop('DepTime', axis=1)
# %% markdown
# Concatenate the training and testing dataframes.
#
# %% codecell
full_df = pd.concat([train_df.drop('dep_delayed_15min', axis=1), test_df])
full_df = feature_eng(full_df)
# %% markdown
# Apply the earlier defined feature engineering functions to the full dataframe.
# %% codecell
for column in ['UniqueCarrier', 'Origin', 'Dest','flight',  'flightUC', 'DestUC', 'OriginUC']:
    full_df[column] = label_enc(full_df[column])
# %% markdown
#
# Split the new full dataframe into X_train and X_test.
# %% codecell
X_train = full_df[:train_df.shape[0]]
X_test = full_df[train_df.shape[0]:]
# %% markdown
# Create a list of the categorical features.
# %% codecell
categorical_features = ['Month',  'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest','flight',  'flightUC', 'DestUC', 'OriginUC']
# %% markdown
# Let's build a light GBM model to test the bayesian optimizer.
# %% markdown
# ### [LightGBM](https://lightgbm.readthedocs.io/en/latest/) is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient with the following advantages:
#
# * Faster training speed and higher efficiency.
# * Lower memory usage.
# * Better accuracy.
# * Support of parallel and GPU learning.
# * Capable of handling large-scale data.
# %% markdown
# First, we define the function we want to maximize and that will count cross-validation metrics of lightGBM for our parameters.
#
# Some params such as num_leaves, max_depth, min_child_samples, min_data_in_leaf should be integers.
# %% codecell
def lgb_eval(num_leaves,max_depth,lambda_l2,lambda_l1,min_child_samples, min_data_in_leaf):
    params = {
        "objective" : "binary",
        "metric" : "auc",
        'is_unbalance': True,
        "num_leaves" : int(num_leaves),
        "max_depth" : int(max_depth),
        "lambda_l2" : lambda_l2,
        "lambda_l1" : lambda_l1,
        "num_threads" : 20,
        "min_child_samples" : int(min_child_samples),
        'min_data_in_leaf': int(min_data_in_leaf),
        "learning_rate" : 0.03,
        "subsample_freq" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1
    }
    lgtrain = lightgbm.Dataset(X_train, y_train,categorical_feature=categorical_features)
    cv_result = lightgbm.cv(params,
                       lgtrain,
                       1000,
                       early_stopping_rounds=100,
                       stratified=True,
                       nfold=3)
    return cv_result['auc-mean'][-1]
# %% markdown
# Apply the Bayesian optimizer to the function we created in the previous step to identify the best hyperparameters. We will run 10 iterations and set init_points = 2.
#
# %% codecell
lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (25, 4000),
                                                'max_depth': (5, 63),
                                                'lambda_l2': (0.0, 0.05),
                                                'lambda_l1': (0.0, 0.05),
                                                'min_child_samples': (50, 10000),
                                                'min_data_in_leaf': (100, 2000)
                                                })

lgbBO.maximize(n_iter=10, init_points=2)
# %% markdown
#  **<font color='teal'> Print the best result by using the '.max' function.</font>**
# %% codecell
lgbBO.max
# %% markdown
# Review the process at each step by using the '.res[0]' function.
# %% codecell
lgbBO.res[0]
