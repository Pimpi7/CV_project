import os
import random
import shutil
from collections import defaultdict
import numpy as np

# Source and destination directories
source_dir = "DFFD_dataset"
destination_dir = "DFFD_dataset_downsampled_050"

splits = [
        'train',
        'validation',
        'test'
    ]

# 6. Generate statistics
print("\nDataset Statistics:")
total_copied = 0
for split_name in splits:
    real_count = len(os.listdir(os.path.join(destination_dir, split_name, 'real')))
    fake_count = len(os.listdir(os.path.join(destination_dir, split_name, 'fake')))
    split_total = real_count + fake_count
    total_copied += split_total
    
    print(f"{split_name.capitalize()}: {split_total} images ({real_count} real, {fake_count} fake)")

print(f"Total: {total_copied} images")
print(f"Real/Fake ratio: {sum([len(os.listdir(os.path.join(destination_dir, s, 'real'))) for s in splits])/total_copied:.2f} real, "
        f"{sum([len(os.listdir(os.path.join(destination_dir, s, 'fake'))) for s in splits])/total_copied:.2f} fake")

