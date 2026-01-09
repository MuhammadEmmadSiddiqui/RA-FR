"""
STEP 2: Generate IMFDB Metadata Files

This script generates the required metadata files for IMFDB dataset:
- images.txt: Maps image IDs to relative paths
- image_class_labels.txt: Maps image IDs to class labels (person IDs)

Input structure (from Step 1):
    IMFDB/
    └── images/
        ├── PersonName1/
        │   ├── movie1_frame001.jpg
        │   └── movie1_frame002.jpg
        └── PersonName2/
            └── movie2_frame001.jpg

Output files:
    - images.txt format:        <image_id> <relative_path>
      Example: 1 PersonName1/movie1_frame001.jpg
               2 PersonName1/movie1_frame002.jpg
               3 PersonName2/movie2_frame001.jpg
    
    - image_class_labels.txt format: <image_id> <class_label>
      Example: 1 0
               2 0
               3 1

Note: This implementation references the dataset loading logic in datasets/imfdb.py
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict


def generate_metadata_files(data_dir):
    """
    Generate images.txt and image_class_labels.txt from organized IMFDB structure.
    
    Args:
        data_dir: Path to IMFDB directory (contains images/ folder)
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    
    if not images_dir.exists():
        print(f"Error: images/ directory not found in {data_dir}")
        print("Please run step1_organize_folders.py first!")
        return
    
    # Collect all person folders and sort them
    person_folders = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    
    if not person_folders:
        print(f"Error: No person folders found in {images_dir}")
        return
    
    print(f"Found {len(person_folders)} person folders")
    print("-" * 60)
    
    # Map person names to class labels (sorted order)
    person_to_label = {person.name: idx for idx, person in enumerate(person_folders)}
    
    # Collect all images with their labels
    image_data = []  # List of tuples: (relative_path, class_label)
    
    for person_folder in person_folders:
        person_name = person_folder.name
        class_label = person_to_label[person_name]
        
        # Get all image files in this person's folder
        image_files = sorted([
            f for f in person_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        for image_file in image_files:
            # Relative path: PersonName/image.jpg
            relative_path = f"{person_name}/{image_file.name}"
            image_data.append((relative_path, class_label))
        
        print(f"  {person_name}: {len(image_files)} images (label={class_label})")
    
    print("-" * 60)
    print(f"Total images: {len(image_data)}")
    
    # Write images.txt
    images_txt_path = data_path / "images.txt"
    with open(images_txt_path, 'w') as f:
        for img_id, (relative_path, _) in enumerate(image_data, start=1):
            f.write(f"{img_id} {relative_path}\n")
    
    print(f"✓ Created: {images_txt_path}")
    
    # Write image_class_labels.txt
    labels_txt_path = data_path / "image_class_labels.txt"
    with open(labels_txt_path, 'w') as f:
        for img_id, (_, class_label) in enumerate(image_data, start=1):
            f.write(f"{img_id} {class_label}\n")
    
    print(f"✓ Created: {labels_txt_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("METADATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total persons:  {len(person_folders)}")
    print(f"Total images:   {len(image_data)}")
    print(f"Class labels:   0 to {len(person_folders) - 1}")
    print("\nOutput files:")
    print(f"  - {images_txt_path}")
    print(f"  - {labels_txt_path}")
    print("=" * 60)
    
    # Print sample entries
    print("\nSample entries (first 5):")
    print("\nimages.txt:")
    with open(images_txt_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.strip()}")
    
    print("\nimage_class_labels.txt:")
    with open(labels_txt_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"  {line.strip()}")
    
    # Verify consistency with imfdb.py loading logic
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print("Checking compatibility with datasets/imfdb.py...")
    
    # Read first few entries to verify format
    with open(images_txt_path, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split(' ')
        if len(parts) == 2 and parts[0].isdigit():
            print("✓ images.txt format is correct")
        else:
            print("✗ Warning: images.txt format may be incorrect")
    
    with open(labels_txt_path, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split(' ')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            print("✓ image_class_labels.txt format is correct")
        else:
            print("✗ Warning: image_class_labels.txt format may be incorrect")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="STEP 2: Generate images.txt and image_class_labels.txt for IMFDB"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dbs/IMFDB',
        help='Path to IMFDB directory containing images/ folder (default: dbs/IMFDB)'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        print("Please run step1_organize_folders.py first!")
        return
    
    print("=" * 60)
    print("IMFDB PREPARATION - STEP 2: GENERATE METADATA")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print("=" * 60)
    print()
    
    # Run metadata generation
    generate_metadata_files(args.data_dir)


if __name__ == "__main__":
    main()
