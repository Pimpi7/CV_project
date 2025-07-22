import os
import sys
from collections import defaultdict
import magic  # For detecting file types - install with: pip install python-magic

def analyze_directory_structure(root_dir, max_depth=None, exclude_dirs=None, max_files_per_dir=5):
    """
    Generate and print a directory tree structure with file type information.
    
    Args:
        root_dir: Path to the root directory to analyze
        max_depth: Maximum depth to traverse (None for unlimited)
        exclude_dirs: List of directory names to exclude
        max_files_per_dir: Maximum number of files to list per directory
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    # Counters for statistics
    stats = {
        'directories': 0,
        'files': 0,
        'total_size': 0,
        'file_types': defaultdict(int),
        'file_extensions': defaultdict(int)
    }
    
    # Detect file MIME type and extension
    def get_file_info(file_path):
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            file_size = os.path.getsize(file_path)
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            stats['files'] += 1
            stats['total_size'] += file_size
            stats['file_types'][file_type] += 1
            stats['file_extensions'][ext] += 1
            
            return file_type, file_size, ext
        except Exception as e:
            return f"Error: {str(e)}", 0, ""
    
    # Generate the tree structure
    def generate_tree(dir_path, prefix="", depth=0):
        if max_depth is not None and depth > max_depth:
            return
        
        dir_name = os.path.basename(dir_path)
        if dir_name in exclude_dirs:
            return
        
        print(f"{prefix}├── {dir_name}/")
        
        # List contents of the directory
        try:
            items = os.listdir(dir_path)
        except PermissionError:
            print(f"{prefix}│   └── [Permission Denied]")
            return
        
        # Separate directories and files
        dirs = []
        files = []
        
        for item in items:
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                dirs.append(item)
                stats['directories'] += 1
            else:
                files.append(item)
        
        # Sort directories and files alphabetically
        dirs.sort()
        files.sort()
        
        # Process files
        if files:
            file_count = len(files)
            files_to_show = files[:max_files_per_dir]
            
            for i, file in enumerate(files_to_show):
                file_path = os.path.join(dir_path, file)
                file_type, file_size, ext = get_file_info(file_path)
                size_str = format_size(file_size)
                
                # Use different prefix for the last item
                if i == len(files_to_show) - 1 and not dirs:
                    # If this is the last file and there are no directories
                    file_prefix = f"{prefix}└── "
                    indent_prefix = f"{prefix}    "
                else:
                    file_prefix = f"{prefix}│   ├── "
                    indent_prefix = f"{prefix}│   │   "
                
                print(f"{file_prefix}{file} ({size_str}, {file_type})")
            
            # If there are more files than we're showing
            if file_count > max_files_per_dir:
                if not dirs:
                    print(f"{prefix}└── ... and {file_count - max_files_per_dir} more files")
                else:
                    print(f"{prefix}│   └── ... and {file_count - max_files_per_dir} more files")
        
        # Process directories
        for i, dir_name in enumerate(dirs):
            dir_path_next = os.path.join(dir_path, dir_name)
            
            # Use different prefix for the last directory
            if i == len(dirs) - 1:
                # If this is the last item
                generate_tree(dir_path_next, prefix + "    ", depth + 1)
            else:
                generate_tree(dir_path_next, prefix + "│   ", depth + 1)
    
    # Start the tree generation
    print(f"Directory Tree for: {os.path.abspath(root_dir)}")
    print("─" * 80)
    
    if os.path.isdir(root_dir):
        root_name = os.path.basename(root_dir) or root_dir
        print(f"{root_name}/")
        generate_tree(root_dir, "", 0)
    else:
        print(f"Error: {root_dir} is not a directory.")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"Total directories: {stats['directories']}")
    print(f"Total files: {stats['files']}")
    print(f"Total size: {format_size(stats['total_size'])}")
    
    print("\nFile Types:")
    for file_type, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {file_type}: {count} files ({count/stats['files']:.2%})")
    
    print("\nFile Extensions:")
    for ext, count in sorted(stats['file_extensions'].items(), key=lambda x: x[1], reverse=True):
        if ext:
            print(f"  - {ext}: {count} files ({count/stats['files']:.2%})")
        else:
            print(f"  - [no extension]: {count} files ({count/stats['files']:.2%})")
    
    return stats

def format_size(size_bytes):
    """Format file size in a human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            break
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} {unit}"

def save_tree_to_file(root_dir, output_file, max_depth=None, exclude_dirs=None, max_files_per_dir=5):
    """Save the directory tree to a text file"""
    # Redirect stdout to file
    original_stdout = sys.stdout
    with open(output_file, 'w', encoding='utf-8') as f:
        sys.stdout = f
        analyze_directory_structure(root_dir, max_depth, exclude_dirs, max_files_per_dir)
        sys.stdout = original_stdout
    
    print(f"Directory tree saved to {output_file}")


if __name__ == "__main__":
    # Set parameters
    dataset_path = "DFFD_dataset"  # Replace with your dataset path
    output_file = "dffd_directory_structure.txt"
    
    # Directories to exclude (optional)
    exclude_dirs = []
    
    # Maximum depth to traverse (None for unlimited)
    max_depth = 4
    
    # Maximum files to list per directory
    max_files_per_dir = 10
    
    # Analyze and print to console
    """ analyze_directory_structure(
        dataset_path, 
        max_depth=max_depth,
        exclude_dirs=exclude_dirs,
        max_files_per_dir=max_files_per_dir
    ) """
    
    # Save to file
    save_tree_to_file(
        dataset_path, 
        output_file,
        max_depth=max_depth,
        exclude_dirs=exclude_dirs,
        max_files_per_dir=max_files_per_dir
    )