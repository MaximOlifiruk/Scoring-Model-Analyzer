from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import FeatureUnion


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


def create_pipelines(numeric_columns, label_columns, ordinal_columns, onehot_columns, data):
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(numeric_columns)),
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    label_encoder = LabelEncoder()
    for column in label_columns:
        data[column] = label_encoder.fit_transform(data[column])
        imputer = SimpleImputer(strategy='most_frequent')
        data[column] = imputer.fit_transform(data[column].values.reshape(-1, 1))
    label_data = data[label_columns]

    ordinal_pipeline = Pipeline([
        ('selector', DataFrameSelector(ordinal_columns)),
        ('ordinal_encoder', OrdinalEncoder()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(onehot_columns)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("ordinal_pipeline", ordinal_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    return full_pipeline, label_data


def create_columns(data):
    numeric_columns = []
    for column in data.columns:
        if data[column].dtype != 'object':
            numeric_columns.append(column) 
    label_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
    ordinal_columns = ['Dependents']
    onehot_columns = ['Property_Area']

    return numeric_columns, label_columns, ordinal_columns, onehot_columns