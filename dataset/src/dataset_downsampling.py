import os
import random
import shutil
from collections import defaultdict
import numpy as np

# Desired proportions
train_fraction = 0.70
val_fraction = 0.15
test_fraction = 0.15  # Should sum to 1.0
real_fake_balance = 0.25  # (real/tot) %

# Target total dataset size (adjust based on your resources)
target_total_images = 10000  # Example: 10K images total

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Source and destination directories
source_dir = "DFFD_dataset"
destination_dir = "DFFD_dataset_downsampled_025"


def sample_files_from_directory(directory, sample_size):
    """Sample a specific number of files from a directory."""
    files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # If we need more files than available, take all of them
    if sample_size >= len(files):
        return files
    
    # Otherwise, randomly sample
    return random.sample(files, sample_size)


def create_downsampled_dataset():
    """Create a downsampled, balanced version of the DFFD dataset."""
    
    # 1. Create destination directories
    os.makedirs(destination_dir, exist_ok=True)
    
    # 2. Identify manipulation types and real image sources
    manipulation_types = []
    for item in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, item)):
            if item not in ['scripts', 'all_bounding_box_files', 'bounding_boxes', 'data_lists']:
                manipulation_types.append(item)
    
    real_dirs = ['ffhq']  # Assuming FFHQ is used for real images
    fake_dirs = [d for d in manipulation_types if d not in real_dirs]
    
    print(f"Found real dirs: {real_dirs}")
    print(f"Found fake dirs: {fake_dirs}")
    
    # 3. Calculate how many images to sample from each category
    total_real_images = int(target_total_images * real_fake_balance)
    total_fake_images = target_total_images - total_real_images
    
    # Divide fake images equally among fake types
    fake_images_per_type = total_fake_images // len(fake_dirs)
    
    # 4. Calculate split sizes
    splits = {
        'train': train_fraction,
        'validation': val_fraction,
        'test': test_fraction
    }
    
    # 5. Sample and copy files
    for split_name, split_fraction in splits.items():
        print(f"Processing {split_name} split...")
        
        # Create split directories
        os.makedirs(os.path.join(destination_dir, split_name), exist_ok=True)
        os.makedirs(os.path.join(destination_dir, split_name, 'real'), exist_ok=True)
        os.makedirs(os.path.join(destination_dir, split_name, 'fake'), exist_ok=True)
        
        
        # Sample real images
        real_images_for_split = int(total_real_images * split_fraction)
        real_images_per_dir = real_images_for_split // len(real_dirs)
        
        for real_dir in real_dirs:
            src_split_dir = os.path.join(source_dir, real_dir, split_name)
            if not os.path.exists(src_split_dir):
                print(f"Warning: Source directory {src_split_dir} does not exist!")
                continue
                
            sampled_files = sample_files_from_directory(src_split_dir, real_images_per_dir)
            for file in sampled_files:
                src_path = os.path.join(src_split_dir, file)
                dst_path = os.path.join(destination_dir, split_name, 'real', f"{real_dir}_{file}")
                shutil.copy2(src_path, dst_path)
                
                
        # Sample fake images
        for fake_dir in fake_dirs:
            fake_images_for_split = int(fake_images_per_type * split_fraction)
            src_split_dir = os.path.join(source_dir, fake_dir, split_name)
            if not os.path.exists(src_split_dir):
                print(f"Warning: Source directory {src_split_dir} does not exist!")
                continue
                
            sampled_files = sample_files_from_directory(src_split_dir, fake_images_for_split)
            for file in sampled_files:
                src_path = os.path.join(src_split_dir, file)
                dst_path = os.path.join(destination_dir, split_name, 'fake', f"{fake_dir}_{file}")
                shutil.copy2(src_path, dst_path)
    
    
    # 6. Generate statistics
    print("\nDataset Statistics:")
    total_copied = 0
    for split_name in splits.keys():
        real_count = len(os.listdir(os.path.join(destination_dir, split_name, 'real')))
        fake_count = len(os.listdir(os.path.join(destination_dir, split_name, 'fake')))
        split_total = real_count + fake_count
        total_copied += split_total
        
        print(f"{split_name.capitalize()}: {split_total} images ({real_count} real, {fake_count} fake)")
    
    print(f"Total: {total_copied} images")
    print(f"Real/Fake ratio: {sum([len(os.listdir(os.path.join(destination_dir, s, 'real'))) for s in splits.keys()])/total_copied:.2f} real, "
          f"{sum([len(os.listdir(os.path.join(destination_dir, s, 'fake'))) for s in splits.keys()])/total_copied:.2f} fake")
    
    # 7. Create annotation files
    for split_name in splits.keys():
        with open(os.path.join(destination_dir, f"{split_name}_annotations.txt"), 'w') as f:
            # Real images
            real_dir = os.path.join(destination_dir, split_name, 'real')
            for img in os.listdir(real_dir):
                f.write(f"{img},0\n")  # 0 for real
                
            # Fake images
            fake_dir = os.path.join(destination_dir, split_name, 'fake')
            for img in os.listdir(fake_dir):
                f.write(f"{img},1\n")  # 1 for fake
    
    print("Annotation files created!")

# Run the downsampling
create_downsampled_dataset()