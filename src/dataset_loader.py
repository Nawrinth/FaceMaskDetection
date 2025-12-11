# dataset_loader.py

import os
import gdown
import zipfile


def load_dataset():
    print("\n=== DATASET LOADER ===")
    choice = input("Do you have the dataset locally? (y/n): ").strip().lower()
    # ------------------------------------
    # Case 1: Dataset on local system
    # ------------------------------------
    if choice == "y":
        dataset_path = input("Enter the local dataset path: ").strip()

        if os.path.exists(dataset_path):
            print("✔ Dataset found at:", dataset_path)
            return dataset_path
        else:
            print("❌ Path does not exist!")
            return None

    # ------------------------------------
    # Case 2: Download from link
    # ------------------------------------
    else:
        url = input("Enter dataset download link: ").strip()

        output = "dataset.zip"

        print("\n⬇ Downloading dataset...")
        try:
            gdown.download(url, output, quiet=False)
        except Exception as e:
            print("❌ Error downloading dataset:", e)
            return None

        print("\n📦 Extracting...")
        try:
            with zipfile.ZipFile(output, "r") as zip_ref:
                zip_ref.extractall("dataset")
        except Exception as e:
            print("❌ Error extracting dataset:", e)
            return None

        dataset_path = "dataset"
        print("✔ Dataset extracted to:", dataset_path)

        return dataset_path


# Allow standalone execution
if __name__ == "__main__":
    path = load_dataset()
    print("\nFinal dataset path:", path)
