# %% [markdown]
### /utils/download_spam_dataset.py
# ## Used to download the dataset 

# %%
import urllib.request
import zipfile
import os
from pathlib import Path

# %%
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, zip_destination_dir):
    """Downloads the dataset using the url provided. If the dataset is already downloaded, the function returns. Extracts the contents of the zip file and makes it a tab-separated value file.
    """
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # Downloading the file
    with urllib.request.urlopen(url) as response:

        if (not zip_destination_dir.exists()):  # Create the directory where the zip file will go to 
            os.makedirs(zip_destination_dir, exist_ok=True) # Handles creation of any non-existing parent directories
        with open(zip_path, "wb") as out_file:  # Download the file to the specified location
            out_file.write(response.read())

    # Unzipping the file to the allocated location
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# %%
def download_dataset_wrapper(extracted_path = "sms_spam_collection"):
    """Code inspired from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb
    Args:
        extracted_path (str): The directory that will contain the training data"""
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

    extracted_path = extracted_path 
    zip_destination_dir=Path(extracted_path)
    zip_path = zip_destination_dir / "sms_spam_collection.zip"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, zip_destination_dir)
        data_split_dir= extracted_path + "/" + "data_splits"
        os.makedirs(name=data_split_dir, exist_ok=True)    # create parent directory for the train, validation, and test datasets for later
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path) 

# %%
if __name__ == "__main__":
    download_dataset_wrapper()
