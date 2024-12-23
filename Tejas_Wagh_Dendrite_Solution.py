#Tejas_Wagh
#Dendrite.ai Internship Assignment

# Load Required Libraries
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.feature_selection import SelectFromModel

# Step 1 - Here, we load the json file(parameters) and the csv file(iris dataset)
with open('dendrite.json') as f:
    jsfile = json.load(f)

dataset_path = jsfile['design_state_data']['session_info']['dataset']
data = pd.read_csv(r"iris2.csv")

# Extract features and target from the dataset
target_column = jsfile['design_state_data']['target']['target']
X = data.drop(columns=[target_column])
y = data[target_column]


# Step 2 - Data Preprocessing 
# Separate numerical and categorical columns
feature_handling = jsfile['design_state_data']['feature_handling']

#Filtering the reduction method as specified in the json document
reduction_method = jsfile['design_state_data']['feature_reduction']['feature_reduction_method']
if reduction_method == 'PCA':
    n_components = int(jsfile['design_state_data']['feature_reduction']['num_of_features_to_keep'])
    reducer = ('reducer', PCA(n_components=n_components))
elif reduction_method == 'Tree-based':
    n_trees = int(jsfile['design_state_data']['feature_reduction']['num_of_trees'])
    base_model = RandomForestRegressor(n_estimators=n_trees)
    reducer = ('reducer', SelectFromModel(base_model))
else:
    reducer = ('reducer', 'passthrough')

# We use SimpleImputer to impute the missing numerical values with mean and replace the missing categorical values with "missing".
# Further, we use standard scaling and one hot encoding on numerical and categorical columns respectively.
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # Drop any columns not specified
)

# Models are initialised 
models = {
    "RandomForestRegressor": RandomForestRegressor(),
    "GBTRegressor": GradientBoostingRegressor(),
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(),
    "LassoRegression": Lasso(),
    "ElasticNetRegression": ElasticNet(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
}

# Now the parameters in the json document cannot be taken as-is. The actual hyperparameters used in GridSearchCV are a combination of the ones in the json document.
# So, the necessary combinations are listed in the dictionary defined below.
# The dictionary is dynamically defined, meaning it can extract parameters even if they are changed in the original json document.

valid_params = {
    "RandomForestRegressor": {
        "n_estimators": lambda x: list(range(x["min_trees"], x["max_trees"] + 1)),
        "max_depth": lambda x: [x["max_depth"]], 
        "min_samples_leaf": lambda x: list(range(x["min_samples_per_leaf_min_value"], 
                                               x["min_samples_per_leaf_max_value"] + 1)),
        "max_features": lambda x: ["sqrt"],
        "n_jobs": lambda x: [-1]
    },
    "GBTRegressor": {
        "n_estimators": lambda x: list(range(x["num_of_BoostingStages"][0], x["num_of_BoostingStages"][1] + 1)),
        "learning_rate": lambda x: [x["min_stepsize"], x["max_stepsize"]],
        "max_depth": lambda x: list(range(x["min_depth"], x["max_depth"] + 1)),
        "subsample": lambda x: [float(rate) for rate in np.linspace(
            max(0.1, min(1.0, x["min_subsample"])), 
            max(0.1, min(1.0, x["max_subsample"])), 
            num=3)]
    },
    "LinearRegression": {},
    "RidgeRegression": {
        "alpha": lambda x: [x["min_regparam"], x["max_regparam"]],
        "max_iter": lambda x: list(range(x["min_iter"], x["max_iter"] + 1)),
    },
    "LassoRegression": {
        "alpha": lambda x: [x["min_regparam"], x["max_regparam"]],
        "max_iter": lambda x: list(range(x["min_iter"], x["max_iter"] + 1)),
    },
    "ElasticNetRegression": {
        "alpha": lambda x: [x["min_regparam"], x["max_regparam"]],
        "l1_ratio": lambda x: [x["min_elasticnet"], x["max_elasticnet"]],
        "max_iter": lambda x: list(range(x["min_iter"], x["max_iter"] + 1)),
    },
    "DecisionTreeRegressor": {
        "max_depth": lambda x: list(range(x["min_depth"], x["max_depth"] + 1)),
        "min_samples_split": lambda x: [2, 5, 10],
        "min_samples_leaf": lambda x: [x["min_samples_per_leaf"]],
    }
}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate over models and perform training and evaluation
for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        try:
            # Build pipeline with preprocessing
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                reducer,
                ('model', model)
            ])

            # Hyperparameter tuning
            param_grid = filtered_params[model_name]
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)

            # Evaluate the model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test,y_pred)
            
# Here we log two of the most common metrics for - mean squared error and r2 score.

            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Mean Squared Error for {model_name}: {mse}")
            print(f"r2 score for {model_name}: {r2}")  
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue