# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths


os.environ["HF_TOKEN"] = "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"  # please use your token
api = HfApi(token=os.getenv("HF_TOKEN"))

# please create your dataset as you create your space
DATASET_PATH = "hf://datasets/DhanashreeP/Tourism-mlops-prediction-FE/tourism.csv"
# hf://datasets/DhanashreeP/Tourism-mlops-prediction-FE/tourism.csv

df_tourism = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("The Dataset shape is :\n", df_tourism.shape)
print("The first 5 rows are :\n", df_tourism.head())
print("The datatypes are :\n", df_tourism.dtypes)


# Handle missing values
# For simplicity, we'll fill missing numerical values with the mean and categorical with the mode.
for col in df_tourism.columns:
    if df_tourism[col].dtype in ['int64', 'float64']:
        df_tourism[col] = df_tourism[col].fillna(df_tourism[col].mean())
    else:
        df_tourism[col] = df_tourism[col].fillna(df_tourism[col].mode()[0])

# Encode categorical features
categorical_cols = df_tourism.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df_tourism[col] = le.fit_transform(df_tourism[col])

# Define features (X) and target (y)
X = df_tourism.drop('ProdTaken', axis=1)
y = df_tourism['ProdTaken']


# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename

        repo_id="DhanashreeP/Tourism-mlops-prediction-FE",

        repo_type="dataset",
    )
