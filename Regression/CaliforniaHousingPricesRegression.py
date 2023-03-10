# ----------------- Libraries ----------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# ----------------- Data Loading ----------------- #
data = pd.read_csv('./housing.csv')

# Test
print('HEAD OF THE RAW DATA')
print(data.head())
print('----------------------------------------------------------')
print('INFORMATIONS ABOUT THE RAW DATA')
print(data.info())
print('----------------------------------------------------------')
print('CATEGORY TYPES OF OCEAN PROXIMITY COLUMN')
print(data[['ocean_proximity']].value_counts())
print('----------------------------------------------------------')



# ----------------- Creating Income Category Column ----------------- #
data['income_category'] = pd.cut(data['median_income'],
                                 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1,2,3,4,5])

# Histogram of income_category
plt.title('Histogram of Income Category Column')
plt.hist(data['income_category'], bins=10)
plt.show()



# ----------------- Separating The Data As Train And Test ----------------- #
# Stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_category']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print('INCOME CATEGORY RATIOS INSIDE THE TEST SET')
print(strat_test_set['income_category'].value_counts() / len(strat_test_set))
print('----------------------------------------------------------')
# Now we should delete the column income_category
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_category', axis=1, inplace=True)



# ----------------- Data Visualization ----------------- #
data = strat_train_set.copy()

plt.title('Distribution of data according to latitude and longitude')
plt.scatter(data['longitude'],
            data['latitude'],
            alpha=0.4,
            s=data['population']/100,   # Radius of every circles
            c=data['median_house_value'],   # Colors of the dots represent the prices (Low:blue/High:red)
            cmap=plt.get_cmap('jet'))
plt.xlabel('Latitude (Enlem)')
plt.ylabel('Longitude (Boylam)')
plt.legend('Population')
plt.show()



# ----------------- Finding Correlations ----------------- #
print('CORRELATIONS')
corr_matrix = data.corr()
# The correlation of each attribute according to the median_house_value
print(corr_matrix['median_house_value'].sort_values(ascending=False))
print('----------------------------------------------------------')
# As we can see median_income column affects really well to the median_house_value column



# ----------------- Creating New Useful Columns ----------------- #
data['rooms_per_household'] = data['total_rooms'] / data['households']
data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
data['population_per_household'] = data['population'] / data['households']
# New correlation matrix
print('CORRELATION MATRIX WITH SOME NEW COLUMNS')
corr_matrix = data.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
# As we can see from cm, bedrooms_per_room has wonderful effect to median_house_value
# Small number of bedrooms -> High median_house_value
print('----------------------------------------------------------')



# ----------------- Data Slicing ----------------- #
data = strat_train_set.drop('median_house_value', axis=1)
data_labels = strat_train_set['median_house_value'].copy()



# ----------------- Missing Value Management ----------------- #
# In this dataset, total_bedrooms column has a few missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
# median can just calculated with numeric things, so we can drop categorical column temporary
data_numeric =data.drop('ocean_proximity',axis=1)
imputer.fit(data_numeric)
X = imputer.transform(data_numeric)
# Creating new data frame (Now, there is no missing value)
data_tr = pd.DataFrame(X, columns=data_numeric.columns, index=data_numeric.index)
print("DATA FRAME'S INFORMATION THAT HAS NOT MISSING VALUE")
print(data_tr.info())
print('----------------------------------------------------------')



# ----------------- Data Encoding ----------------- #
# (Categorical -> Numeric)
data_categorical = data[['ocean_proximity']]
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
data_encoded = ordinal_encoder.fit_transform(data_categorical)
# One hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
data_categorical_1hot = ohe.fit_transform(data_categorical)



# ----------------- Special Transformators ----------------- #
from sklearn.base import BaseEstimator, TransformerMixin
rooms_column_index, bedrooms_column_index, population_column_index, households_column_index = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self     # There is nothing to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_column_index] / X[:, households_column_index]
        population_per_household = X[:, population_column_index] / X[:, households_column_index]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_column_index] / X[:, rooms_column_index]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]




# ----------------- Data Scaling / Pipelines ----------------- #
# That process is kind of automation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline takes parameters that will process step by step
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])
# This pipeline will process everything that you wrote in order
# It manages missing values, transformation process, scaling

'''
By calling the fit_transform method, the operations were performed in sequence, 
and the result of each operation became the input of the other step
'''
data_numeric_tr = numeric_pipeline.fit_transform(data_numeric)
print('DATAFRAME THAT OBTAINED FROM PIPELINE')
print(data_numeric_tr)
print('----------------------------------------------------------')
# Until here, this pipeline will just process something about numeric columns
# If you want to manage numeric and categorical columns just one process, you'll code that is under here

from sklearn.compose import ColumnTransformer
# Numeric attributes for send them to the numeric pipeline
numeric_attribs = list(data_numeric)
categorical_attribs = ['ocean_proximity']

# That code will process like a pipeline
full_pipeline = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_attribs),
    ('categorical', OneHotEncoder(), categorical_attribs)
])

# Let's process the data using the full_pipeline with one line
data_prepared = full_pipeline.fit_transform(data)
print('THE PREPARED DATA')
print(data_prepared)
print('----------------------------------------------------------')



# ----------------- Linear Regression ----------------- #
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
# Train the model
linear_reg.fit(data_prepared, data_labels)

# Get the data from somewhere
some_data = data.iloc[:5]
some_labels = data_labels.iloc[:5]

# Preparing our data using full_pipeline
some_data_prepared = full_pipeline.transform(some_data)

# Make predictions
print('Predictions: ', linear_reg.predict(some_data_prepared))
print('Labels', list(some_labels))
print('----------------------------------------------------------')

# Measure the score of the linear model
from sklearn.metrics import mean_squared_error
data_predictions = linear_reg.predict(data_prepared)
linear_mse = mean_squared_error(data_labels, data_predictions)
linear_rmse = np.sqrt(linear_mse)
print("linear_rmse = ", linear_rmse) # 68627.87 is too high prediction error
print('----------------------------------------------------------')

# K-fold validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_reg, data_prepared, data_labels, scoring='neg_mean_squared_error',cv=10)
linear_rmse_scores = np.sqrt(-scores)
def displayScores(scores):
    '''
    This func, prints some scores about the model
    :param scores:
    :return:
    '''
    print('Scores : ', scores)
    print('Mean : ', scores.mean())
    print('Standart deviation : ', scores.std())
    print('----------------------------------------------------------')
print('Linear Reg Rmse Scores')
displayScores(linear_rmse_scores)
# We may use another model



# ----------------- Decision Tree Regression ----------------- #
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
# Train the model
tree_reg.fit(data_prepared, data_labels)

# Measure the score of the decision tree regression
data_predictions = tree_reg.predict(data_prepared)
tree_mse = mean_squared_error(data_labels, data_predictions)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse = ", tree_rmse) # Result is 0, how? We need to try to get error with another algorithm
print('----------------------------------------------------------')

# K-fold validation
scores = cross_val_score(tree_reg, data_prepared, data_labels, scoring='neg_mean_squared_error',cv=10)
tree_rmse_scores = np.sqrt(-scores)

print('Decision Tree Reg Rmse Scores')
displayScores(tree_rmse_scores)
# Decision tree worser than linear regression for that data set
# We may choose another model



# ----------------- Random Forest Regression ----------------- #
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
# Train the model
forest_reg.fit(data_prepared, data_labels)

# Measure the score of the random forest regression
data_predictions = forest_reg.predict(data_prepared)
forest_mse = mean_squared_error(data_labels, data_predictions)
forest_rmse = np.sqrt(forest_mse)
print("forest_rmse = ", forest_rmse) # 19247.54
print('----------------------------------------------------------')

# K-fold validation
scores = cross_val_score(forest_reg, data_prepared, data_labels, scoring='neg_mean_squared_error',cv=10)
forest_rmse_scores = np.sqrt(-scores)
displayScores(forest_rmse_scores)
# As we can see from K-fold validation, random forest can be used for this problem



# ----------------- GridSearchCV for Random Forest ----------------- #
from sklearn.model_selection import GridSearchCV
# We write down all the situations we want to be tried
parameters = [
    {'n_estimators':[3, 10, 30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2,3,4]}
]
# Model to train
forest_reg = RandomForestRegressor()

# Process the grid search algorithm
grid_search = GridSearchCV(forest_reg,
                           parameters,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(data_prepared, data_labels)
# With that parameters, the computer calculates all the possibilities 90 times, it'll take time

# Let's check optimum parameters
print('Best parameters of random forest for this data set')
print(grid_search.best_params_) # max_features = 6 and n_estimators = 30 are optimum parameters
print('----------------------------------------------------------')



# ----------------- Predictions Using Test Set ----------------- #
# Get the model with optimum parameters
final_model = grid_search.best_estimator_
x_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()
# As you can see we can directly transform the test set using the pipeline that we wrote
x_test_prepared = full_pipeline.transform(x_test)

final_predictions = final_model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("final_rmse = ", final_rmse) # 47957.47