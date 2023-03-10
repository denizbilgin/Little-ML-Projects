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



# ----------------- Data slicing ----------------- #
data = strat_train_set.drop('median_house_value', axis=1)
data_labels = strat_train_set['median_house_value'].copy()



# *********************************** PIPELINE BASED AUTOMIZE SYSTEM *********************************** #
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



# ----------------- Pipelines ----------------- #
# That process is kind of automation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Pipeline takes parameters that will process step by step
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Missing value management
    ('attribs_adder', CombinedAttributesAdder()),   # Special transformation (adding new useful columns)
    ('std_scaler', StandardScaler())                # Scaling
])
# This pipeline will process everything that you wrote in order
# It manages missing values, transformation process, scaling
'''
By calling the fit_transform method, the operations were performed in sequence, 
and the result of each operation became the input of the other step
'''
data_numeric = data.drop('ocean_proximity',axis=1)
data_numeric_tr = numeric_pipeline.fit_transform(data_numeric)
print('DATAFRAME THAT OBTAINED FROM PIPELINE')
print(data_numeric_tr)
print('----------------------------------------------------------')
# Until here, this pipeline will just process something about numeric columns
# If you want to manage numeric and categorical columns just one process, you'll code that is under here

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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


''' NOW YOU CAN STUDY ON ML ALGORITHMS, BECAUSE YOUR DATA IS READY TO TRAINING '''
