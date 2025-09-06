#!/usr/bin/env python3
"""
Utility script to create a batch file for predict_interactive.py
Scans a directory for image files and creates a list file.
"""

import os
import argparse
from pathlib import Path

# Common image file extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'
}

def create_batch_file(image_dir, output_file, recursive=False):
    """
    Create a batch file containing paths to image files.
    
    Args:
        image_dir (str): Directory to scan for images
        output_file (str): Path to output batch file
        recursive (bool): Whether to scan subdirectories recursively
    """
    image_dir = Path(image_dir)
    output_path = Path(output_file)
    
    if not image_dir.exists():
        print(f"Error: Directory '{image_dir}' does not exist.")
        return False
        
    # Find all image files
    image_files = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(image_dir.rglob(f"*{ext}"))
            image_files.extend(image_dir.rglob(f"*{ext.upper()}"))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(str(f) for f in image_files)))
    
    if not image_files:
        print(f"No image files found in '{image_dir}'")
        return False
    
    # Write to output file
    try:
        with open(output_path, 'w') as f:
            for image_file in image_files:
                # Quote paths that contain spaces
                if ' ' in str(image_file):
                    f.write(f'"{image_file}"\n')
                else:
                    f.write(f"{image_file}\n")
        print(f"Successfully created batch file with {len(image_files)} images: {output_path}")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create a batch file for FastVLM interactive prediction")
    parser.add_argument("image_dir", help="Directory containing image files")
    parser.add_argument("-o", "--output", default="image_list.txt", help="Output batch file path (default: image_list.txt)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Scan subdirectories recursively")
    
    args = parser.parse_args()
    
    create_batch_file(args.image_dir, args.output, args.recursive)

if __name__ == "__main__":
    main()