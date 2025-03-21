import gdown
import os

# Google Drive file IDs (Replace with correct ones)
model_file_id = "1cWzgwHBLYn_RdWBziellO_UlKC87ZrHj"
tokenizer_file_id = "1eWOZbuKOE90vBdLG4rvvL9Clzt9tCrt1"

# File paths
model_path = "news_model.keras"
tokenizer_path = "tokenizer.pkl"

# Download if not exists
if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_path, quiet=False)

if not os.path.exists(tokenizer_path):
    print("Downloading tokenizer...")
    gdown.download(f"https://drive.google.com/uc?id={tokenizer_file_id}", tokenizer_path, quiet=False)

print("All files downloaded successfully!")
