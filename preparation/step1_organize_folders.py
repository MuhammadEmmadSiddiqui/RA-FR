"""
STEP 1: Organize IMFDB Folder Structure

This script organizes raw IMFDB data into a unified structure:
- Extracts images from movie-specific folders
- Organizes by PersonName subdirectories
- Creates a flat images/ folder with person-based organization

Input structure (raw IMFDB):
    raw_imfdb/
    ├── Movie1/
    │   ├── PersonName1/
    │   │   ├── frame001.jpg
    │   │   └── frame002.jpg
    │   └── PersonName2/
    │       └── frame001.jpg
    └── Movie2/
        └── PersonName1/
            └── frame001.jpg

Output structure:
    IMFDB/
    └── images/
        ├── PersonName1/
        │   ├── Movie1_frame001.jpg
        │   ├── Movie1_frame002.jpg
        │   └── Movie2_frame001.jpg
        └── PersonName2/
            └── Movie1_frame001.jpg
"""

import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict


def organize_imfdb_folders(input_dir, output_dir):
    """
    Organize IMFDB raw data into unified person-based folder structure.
    
    Args:
        input_dir: Path to raw IMFDB data (movie folders)
        output_dir: Path to output organized IMFDB data
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    
    # Create output directory
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        'movies_processed': 0,
        'persons_found': set(),
        'images_copied': 0,
        'duplicates_skipped': 0
    }
    
    print(f"Scanning input directory: {input_path}")
    print("-" * 60)
    
    # Iterate through movie folders
    for movie_folder in sorted(input_path.iterdir()):
        if not movie_folder.is_dir():
            continue
            
        movie_name = movie_folder.name
        stats['movies_processed'] += 1
        print(f"\nProcessing movie: {movie_name}")
        
        # Iterate through person folders within each movie
        for person_folder in sorted(movie_folder.iterdir()):
            if not person_folder.is_dir():
                continue
                
            person_name = person_folder.name
            stats['persons_found'].add(person_name)
            
            # Create person directory in output
            output_person_dir = images_dir / person_name
            output_person_dir.mkdir(exist_ok=True)
            
            # Copy all images with movie prefix
            image_count = 0
            for image_file in person_folder.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Create new filename with movie prefix
                    new_filename = f"{movie_name}_{image_file.name}"
                    output_path_img = output_person_dir / new_filename
                    
                    # Check for duplicates
                    if output_path_img.exists():
                        stats['duplicates_skipped'] += 1
                        print(f"  ⚠ Skipping duplicate: {new_filename}")
                        continue
                    
                    # Copy image
                    shutil.copy2(image_file, output_path_img)
                    stats['images_copied'] += 1
                    image_count += 1
            
            if image_count > 0:
                print(f"  ✓ {person_name}: {image_count} images")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"Movies processed:     {stats['movies_processed']}")
    print(f"Unique persons found: {len(stats['persons_found'])}")
    print(f"Images copied:        {stats['images_copied']}")
    print(f"Duplicates skipped:   {stats['duplicates_skipped']}")
    print(f"\nOutput directory:     {images_dir}")
    print("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="STEP 1: Organize IMFDB raw data into person-based folder structure"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to raw IMFDB data directory (contains movie folders)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='dbs/IMFDB',
        help='Path to output IMFDB directory (default: dbs/IMFDB)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    print("=" * 60)
    print("IMFDB PREPARATION - STEP 1: ORGANIZE FOLDERS")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Run organization
    organize_imfdb_folders(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
