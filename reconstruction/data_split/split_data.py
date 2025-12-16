import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse

def get_file_list(source_dir: Path) -> List[Path]:
    """Get list of .h5 files from source directory."""
    return sorted(list(source_dir.glob("*.h5")))

def create_directories(base_dir: Path) -> Tuple[Path, Path, Path]:
    """Create train, val, and test directories."""
    train_dir = base_dir / "multicoil_train"
    val_dir = base_dir / "multicoil_val" 
    test_dir = base_dir / "multicoil_test"
    
    # Create directories if they don't exist
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return train_dir, val_dir, test_dir

def split_files(files: List[Path], train_ratio: float = 0.7, val_ratio: float = 0.15, 
                test_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split files into train, validation, and test sets."""
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle files
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)
    
    total_files = len(files_shuffled)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    # Split files
    train_files = files_shuffled[:train_count]
    val_files = files_shuffled[train_count:train_count + val_count]
    test_files = files_shuffled[train_count + val_count:]
    
    print(f"Total files: {total_files}")
    print(f"Train files: {len(train_files)} ({len(train_files)/total_files:.1%})")
    print(f"Val files: {len(val_files)} ({len(val_files)/total_files:.1%})")
    print(f"Test files: {len(test_files)} ({len(test_files)/total_files:.1%})")
    
    return train_files, val_files, test_files

def copy_files(files: List[Path], destination_dir: Path, split_name: str, 
               copy_mode: str = "copy") -> None:
    """Copy or move files to destination directory."""
    
    print(f"\n{copy_mode.capitalize()}ing {len(files)} files to {split_name} set...")
    
    for i, file_path in enumerate(files, 1):
        dest_path = destination_dir / file_path.name
        
        try:
            if copy_mode == "copy":
                shutil.copy2(file_path, dest_path)
            elif copy_mode == "move":
                shutil.move(str(file_path), str(dest_path))
            else:
                raise ValueError(f"Invalid copy_mode: {copy_mode}")
                
            if i % 10 == 0 or i == len(files):
                print(f"  Progress: {i}/{len(files)} files processed")
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

def get_file_sizes(files: List[Path]) -> Tuple[float, float, float]:
    """Calculate total size of files in GB."""
    total_size = sum(f.stat().st_size for f in files)
    return total_size / (1024**3)  # Convert to GB

def print_summary(train_files: List[Path], val_files: List[Path], 
                  test_files: List[Path]) -> None:
    """Print summary of the data split."""
    
    train_size = get_file_sizes(train_files)
    val_size = get_file_sizes(val_files)
    test_size = get_file_sizes(test_files)
    total_size = train_size + val_size + test_size
    
    print("\n" + "="*60)
    print("DATA SPLIT SUMMARY")
    print("="*60)
    print(f"{'Split':<10} {'Files':<8} {'Size (GB)':<12} {'Percentage':<12}")
    print("-" * 60)
    print(f"{'Train':<10} {len(train_files):<8} {train_size:<12.2f} {train_size/total_size:<12.1%}")
    print(f"{'Val':<10} {len(val_files):<8} {val_size:<12.2f} {val_size/total_size:<12.1%}")
    print(f"{'Test':<10} {len(test_files):<8} {test_size:<12.2f} {test_size/total_size:<12.1%}")
    print("-" * 60)
    print(f"{'Total':<10} {len(train_files + val_files + test_files):<8} {total_size:<12.2f} {'100.0%':<12}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Split FastMRI knee data into train/val/test sets")
    
    parser.add_argument(
        "--source_dir", 
        type=Path,
        default="/scratch/ml8347/MRI/Dataset/FastMRI/knee/multicoil_train",
        help="Source directory containing .h5 files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path, 
        default="/scratch/ml8347/MRI/train/train_dataset",
        help="Output directory for train/val/test splits"
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of files for training set (default: 0.7)"
    )
    
    parser.add_argument(
        "--val_ratio", 
        type=float,
        default=0.15,
        help="Ratio of files for validation set (default: 0.15)"
    )
    
    parser.add_argument(
        "--test_ratio",
        type=float, 
        default=0.15,
        help="Ratio of files for test set (default: 0.15)"
    )
    
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "move"],
        default="copy",
        help="Whether to copy or move files (default: copy)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show split without actually copying/moving files"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")
    
    # Get list of files
    print(f"Scanning source directory: {args.source_dir}")
    files = get_file_list(args.source_dir)
    
    if not files:
        raise ValueError(f"No .h5 files found in {args.source_dir}")
    
    print(f"Found {len(files)} .h5 files")
    
    # Split files
    train_files, val_files, test_files = split_files(
        files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    # Print summary
    print_summary(train_files, val_files, test_files)
    
    if args.dry_run:
        print("\nDRY RUN - No files were actually copied/moved.")
        return
    
    # Create output directories
    train_dir, val_dir, test_dir = create_directories(args.output_dir)
    
    # Copy/move files
    copy_files(train_files, train_dir, "train", args.copy_mode)
    copy_files(val_files, val_dir, "validation", args.copy_mode)  
    copy_files(test_files, test_dir, "test", args.copy_mode)
    
    print(f"\n✅ Data split completed successfully!")
    print(f"Output directory: {args.output_dir}")
    
    # Create a summary file
    summary_file = args.output_dir / "split_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("FastMRI Knee Data Split Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Source directory: {args.source_dir}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Copy mode: {args.copy_mode}\n\n")
        
        f.write(f"Train files ({len(train_files)}):\n")
        for file in train_files:
            f.write(f"  {file.name}\n")
        
        f.write(f"\nValidation files ({len(val_files)}):\n")
        for file in val_files:
            f.write(f"  {file.name}\n")
            
        f.write(f"\nTest files ({len(test_files)}):\n") 
        for file in test_files:
            f.write(f"  {file.name}\n")
    
    print(f"📄 Split summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
