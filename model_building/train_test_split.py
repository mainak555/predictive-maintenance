
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import pandas as pd
import sys
import os

# loading dataset from Hugging Face data space
hfApi = HfApi(token=os.getenv("HF_TOKEN"))
target = os.getenv('TARGET_VARIABLE', default='Engine Condition')
DATASET_PATH = f"hf://datasets/{os.getenv("HF_REPO")}/{os.getenv("CSV_DATA_FILE")}"

try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
        print(f"{os.getenv("CSV_DATA_FILE")}.csv missing @HF Dataset")
        sys.exit(1)
except Exception as e:
    print(f"Error Checking Path: {DATASET_PATH} | Err: {e}")
    sys.exit(1)

print("\033[1mRows: {}\033[0m & \033[1mColumns: {}\033[0m".format(
            df.shape[0], df.shape[1]
    ))

# removing unnecessary column(s)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# train-test split
train, test = train_test_split(
    df, test_size=0.2, random_state=40, stratify=df[target]
)

# saving train/test split locally
train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)

# uploading train and test datasets back to the Hugging Face data space
for file in ["train.csv","test.csv"]:
    hfApi.upload_file(
        repo_id=os.getenv('HF_REPO'),
        path_or_fileobj=file,
        repo_type="dataset",
        path_in_repo=file,
    )
