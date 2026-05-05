# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Download latest version
path = kagglehub.dataset_download("bhavikjikadara/fake-news-detection", output_dir="/home/renku/work/aml2026-group-13/datasets")

print("Path to dataset files:", path)

  
