import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import seaborn as sns

def analyze_dffd_dataset(dataset_path):
    """
    Analyze the DFFD dataset to count real/fake images and the number of images per technique.
    
    Args:
        dataset_path: Path to the DFFD dataset
    """
    # Dictionary to store counts
    stats = {
        'overall': {'real': 0, 'fake': 0},
        'by_split': defaultdict(lambda: {'real': 0, 'fake': 0}),
        'by_technique': defaultdict(int),
        'by_split_and_technique': defaultdict(lambda: defaultdict(int))
    }
    
    # Get all technique folders (excluding utility folders)
    excluded_dirs = ['scripts', 'all_bounding_box_files', 'bounding_boxes', 'data_lists']
    technique_dirs = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d)) 
                     and d not in excluded_dirs]
    
    # Identify real vs. fake techniques
    # Assuming 'ffhq' is the only real image source - adjust if needed
    real_dirs = ['ffhq']
    fake_dirs = [d for d in technique_dirs if d not in real_dirs]
    
    print(f"Found real directories: {real_dirs}")
    print(f"Found fake directories: {fake_dirs}")
    
    # Process each technique directory
    for technique in technique_dirs:
        technique_path = os.path.join(dataset_path, technique)
        is_real = technique in real_dirs
        category = 'real' if is_real else 'fake'
        
        # Process each split (train, validation, test)
        for split in ['train', 'validation', 'test']:
            split_path = os.path.join(technique_path, split)
            
            if not os.path.exists(split_path):
                print(f"Warning: Split path does not exist: {split_path}")
                continue
                
            # Count images in this split
            image_count = len([f for f in os.listdir(split_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Update statistics
            stats['overall'][category] += image_count
            stats['by_split'][split][category] += image_count
            
            if not is_real:
                stats['by_technique'][technique] += image_count
                stats['by_split_and_technique'][split][technique] += image_count
    
    return stats

def print_stats(stats):
    """Print dataset statistics in a readable format"""
    print("\n=== DFFD Dataset Statistics ===\n")
    
    # Overall stats
    total = stats['overall']['real'] + stats['overall']['fake']
    print(f"Total images: {total}")
    print(f"Real images: {stats['overall']['real']} ({stats['overall']['real']/total:.2%})")
    print(f"Fake images: {stats['overall']['fake']} ({stats['overall']['fake']/total:.2%})")
    
    # Stats by split
    print("\n--- Images by Split ---")
    for split, counts in stats['by_split'].items():
        split_total = counts['real'] + counts['fake']
        print(f"\n{split.capitalize()} split:")
        print(f"  Total: {split_total}")
        print(f"  Real: {counts['real']} ({counts['real']/split_total:.2%})")
        print(f"  Fake: {counts['fake']} ({counts['fake']/split_total:.2%})")
    
    # Stats by technique
    print("\n--- Fake Images by Technique ---")
    for technique, count in sorted(stats['by_technique'].items(), key=lambda x: x[1], reverse=True):
        print(f"{technique}: {count} ({count/stats['overall']['fake']:.2%})")
    
    # Stats by split and technique
    print("\n--- Fake Images by Split and Technique ---")
    for split, techniques in stats['by_split_and_technique'].items():
        print(f"\n{split.capitalize()} split:")
        for technique, count in sorted(techniques.items(), key=lambda x: x[1], reverse=True):
            print(f"  {technique}: {count} ({count/stats['by_split'][split]['fake']:.2%})")

def visualize_stats(stats, output_dir="dataset_stats"):
    """Create visualizations of the dataset statistics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Real vs Fake pie chart
    plt.figure(figsize=(10, 6))
    plt.pie([stats['overall']['real'], stats['overall']['fake']], 
            labels=['Real', 'Fake'], 
            autopct='%1.1f%%',
            colors=['#66b3ff', '#ff9999'],
            startangle=90)
    plt.title('Distribution of Real vs. Fake Images')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_vs_fake_pie.png"), dpi=300)
    plt.close()
    
    # 2. Images by split bar chart
    plt.figure(figsize=(12, 6))
    splits = list(stats['by_split'].keys())
    real_counts = [stats['by_split'][s]['real'] for s in splits]
    fake_counts = [stats['by_split'][s]['fake'] for s in splits]
    
    x = range(len(splits))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], real_counts, width, label='Real', color='#66b3ff')
    plt.bar([i + width/2 for i in x], fake_counts, width, label='Fake', color='#ff9999')
    
    plt.xlabel('Split')
    plt.ylabel('Number of Images')
    plt.title('Image Distribution by Split')
    plt.xticks(x, [s.capitalize() for s in splits])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images_by_split.png"), dpi=300)
    plt.close()
    
    # 3. Techniques distribution bar chart
    plt.figure(figsize=(14, 8))
    techniques = list(stats['by_technique'].keys())
    counts = [stats['by_technique'][t] for t in techniques]
    
    # Sort techniques by count
    sorted_data = sorted(zip(techniques, counts), key=lambda x: x[1], reverse=True)
    techniques, counts = zip(*sorted_data)
    
    plt.bar(techniques, counts, color=sns.color_palette("husl", len(techniques)))
    plt.xlabel('Technique')
    plt.ylabel('Number of Images')
    plt.title('Fake Images by Technique')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images_by_technique.png"), dpi=300)
    plt.close()
    
    # 4. Heatmap for split and technique
    techniques = sorted(stats['by_technique'].keys())
    splits = ['train', 'validation', 'test']
    
    data = []
    for split in splits:
        row = []
        for technique in techniques:
            count = stats['by_split_and_technique'].get(split, {}).get(technique, 0)
            row.append(count)
        data.append(row)
    
    plt.figure(figsize=(16, 10))
    df = pd.DataFrame(data, index=[s.capitalize() for s in splits], columns=techniques)
    sns.heatmap(df, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Number of Images by Split and Technique')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_split_technique.png"), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def export_csv(stats, output_dir="dataset_stats"):
    """Export dataset statistics to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall stats
    overall_df = pd.DataFrame({
        'Category': ['Real', 'Fake', 'Total'],
        'Count': [
            stats['overall']['real'], 
            stats['overall']['fake'],
            stats['overall']['real'] + stats['overall']['fake']
        ],
        'Percentage': [
            stats['overall']['real']/(stats['overall']['real'] + stats['overall']['fake']),
            stats['overall']['fake']/(stats['overall']['real'] + stats['overall']['fake']),
            1.0
        ]
    })
    overall_df.to_csv(os.path.join(output_dir, "overall_stats.csv"), index=False)
    
    # 2. Stats by split
    split_data = []
    for split, counts in stats['by_split'].items():
        split_total = counts['real'] + counts['fake']
        split_data.extend([
            {'Split': split, 'Category': 'Real', 'Count': counts['real'], 'Percentage': counts['real']/split_total},
            {'Split': split, 'Category': 'Fake', 'Count': counts['fake'], 'Percentage': counts['fake']/split_total},
            {'Split': split, 'Category': 'Total', 'Count': split_total, 'Percentage': 1.0}
        ])
    
    split_df = pd.DataFrame(split_data)
    split_df.to_csv(os.path.join(output_dir, "split_stats.csv"), index=False)
    
    # 3. Stats by technique
    technique_data = []
    for technique, count in stats['by_technique'].items():
        technique_data.append({
            'Technique': technique,
            'Count': count,
            'Percentage': count/stats['overall']['fake']
        })
    
    technique_df = pd.DataFrame(technique_data)
    technique_df.to_csv(os.path.join(output_dir, "technique_stats.csv"), index=False)
    
    # 4. Stats by split and technique
    split_technique_data = []
    for split, techniques in stats['by_split_and_technique'].items():
        for technique, count in techniques.items():
            split_technique_data.append({
                'Split': split,
                'Technique': technique,
                'Count': count,
                'Percentage of Split': count/stats['by_split'][split]['fake']
            })
    
    split_technique_df = pd.DataFrame(split_technique_data)
    split_technique_df.to_csv(os.path.join(output_dir, "split_technique_stats.csv"), index=False)
    
    print(f"CSV files exported to {output_dir}")



if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path = "DFFD_dataset"
    
    # Analyze dataset
    stats = analyze_dffd_dataset(dataset_path)
    
    # Print statistics
    print_stats(stats)
    
    # Create visualizations
    visualize_stats(stats)
    
    # Export to CSV
    export_csv(stats)