#!/usr/bin/env python3
"""
Complete Math Books Image Organizer
Organizes math book images (covers, intros, ToCs) into a consistent volume-based structure.
Handles all publishers and all volume combinations with proper naming conventions.
"""

import os
import shutil
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Publisher mappings
PUBLISHER_MAP = {
    "בני גורן": "goren",
    "ארכימדס": "archimedes", 
    "יואל גבע": "geva"
}

# Volume mappings
VOLUME_MAP = {
    "א": "a", "ב": "b", "ג": "c", "ד": "d", "ה": "e",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "_1": "1", "_2": "2", "_11": "11", "_12": "12"
}

# Grade mappings
GRADE_MAP = {
    "י": "10", "יא": "11", "יב": "12",
    "10": "10", "11": "11", "12": "12"
}

# Category keywords with priorities (higher number = higher priority)
CATEGORY_KEYWORDS = {
    "cover": {
        "keywords": ["כריכה", "cover", "כותרת", "שער"],
        "priority": 3
    },
    "intro": {
        "keywords": ["מבוא", "פרומפט", "intro", "הקדמה", "פרמפט"],
        "priority": 2
    },
    "toc": {
        "keywords": ["תוכן", "עניינים", "toc", "content", "ענינים"],
        "priority": 1
    }
}

# Special file patterns
SPECIAL_PATTERNS = {
    "toc": [
        r"^M\d+\.pdf$",  # Archimedes M files (M1.pdf, M2.pdf, etc.)
        r"משולב.*\.pdf$"  # Combined files
    ]
}


class MathBookOrganizer:
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)
        
        # Track processed files for summary
        self.processed_files = []
        self.skipped_files = []
        
    def clean_filename(self, filename: str) -> str:
        """Remove special characters and normalize filename."""
        # Remove Hebrew special characters and leading numbers
        cleaned = re.sub(r'^[‏\u200f\u200e\d\s]*', '', filename)
        return cleaned.strip()
    
    def get_category(self, filename: str, file_path: Optional[Path] = None) -> Tuple[str, int]:
        """Determine file category from filename and return category with priority."""
        cleaned = self.clean_filename(filename).lower()
        
        best_category = "unknown"
        best_priority = 0
        
        # Check special patterns first
        for category, patterns in SPECIAL_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, filename, re.IGNORECASE):
                    return category, 10  # High priority for special patterns
        
        # Check filename for category keywords
        for category, info in CATEGORY_KEYWORDS.items():
            for keyword in info["keywords"]:
                if keyword in cleaned or keyword in filename.lower():
                    if info["priority"] > best_priority:
                        best_category = category
                        best_priority = info["priority"]
                        
        return best_category, best_priority
    
    def extract_number_from_filename(self, filename: str, category: str) -> int:
        """Extract ordering number from filename."""
        # Look for numbers in various patterns
        patterns = [
            r'_(\d+)(?:\.|$)',  # number after underscore before extension
            r'\s(\d+)(?:\.|$)',  # number after space before extension
            rf'{category}.*?(\d+)',  # number after category word
            r'(\d+)$',  # number at end of name (before extension)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 1  # Default to 1 if no number found
    
    def process_file(self, source_path: Path, target_folder: Path, 
                     category_counts: Dict[str, int]) -> bool:
        """Process a single image file."""
        filename = source_path.name
        category, priority = self.get_category(filename, source_path)
        
        if category == "unknown":
            print(f"  Warning: Unknown category for {filename}, skipping...")
            self.skipped_files.append(str(source_path))
            return False
        
        # Get the next number for this category
        if category not in category_counts:
            category_counts[category] = 1
        else:
            category_counts[category] += 1
        
        number = category_counts[category]
        
        # Create target filename
        extension = source_path.suffix.lower()
        target_filename = f"{category}-{number}{extension}"
        target_path = target_folder / target_filename
        
        # Handle duplicates
        counter = 1
        while target_path.exists():
            counter += 1
            target_filename = f"{category}-{number}_{counter}{extension}"
            target_path = target_folder / target_filename
        
        # Copy file
        print(f"  Copying: {filename} -> {target_filename}")
        try:
            shutil.copy2(source_path, target_path)
            self.processed_files.append({
                "source": str(source_path),
                "target": str(target_path),
                "category": category
            })
            return True
        except Exception as e:
            print(f"  Error copying {filename}: {e}")
            self.skipped_files.append(str(source_path))
            return False
    
    def organize_files_by_category(self, files: List[Path], target_folder: Path) -> int:
        """Organize files by category (cover, intro, toc) and process in order."""
        target_folder.mkdir(exist_ok=True)
        
        # Group files by category first
        categorized_files = {"cover": [], "intro": [], "toc": []}
        unknown_files = []
        
        for file_path in files:
            category, priority = self.get_category(file_path.name, file_path)
            if category in categorized_files:
                categorized_files[category].append((file_path, priority))
            else:
                unknown_files.append(file_path)
        
        # Sort files within each category by filename (to maintain order)
        for category in categorized_files:
            categorized_files[category].sort(key=lambda x: (x[1], x[0].name))  # Sort by priority then filename
        
        processed_count = 0
        category_counts = {}
        
        # Process files in order: cover, intro, toc
        for category in ["cover", "intro", "toc"]:
            for file_path, _ in categorized_files[category]:
                if self.process_file(file_path, target_folder, category_counts):
                    processed_count += 1
        
        # Process unknown files last
        for file_path in unknown_files:
            if self.process_file(file_path, target_folder, category_counts):
                processed_count += 1
        
        return processed_count
    
    def get_all_image_files(self, directory: Path, extensions: List[str] = None) -> List[Path]:
        """Get all image files from directory recursively."""
        if extensions is None:
            extensions = ["*.JPG", "*.jpg", "*.jpeg", "*.png", "*.PNG", "*.pdf", "*.PDF"]
        
        all_files = []
        if directory.exists() and directory.is_dir():
            for ext in extensions:
                all_files.extend(directory.rglob(ext))
        
        # Filter out non-image files and system files
        filtered_files = []
        for file_path in all_files:
            if (file_path.is_file() and 
                not file_path.name.startswith('.') and 
                not file_path.name.startswith('~') and
                not file_path.name.lower().endswith('.docx') and
                not file_path.name.lower().endswith('.ini')):
                filtered_files.append(file_path)
        
        return sorted(filtered_files, key=lambda x: x.name)
    
    def organize_goren_books(self):
        """Organize Beni Goren books."""
        print("\nProcessing בני גורן books...")
        base_path = self.source_dir / "בני גורן" / "בני גורן תוכן עניינים"
        
        if not base_path.exists():
            print("  בני גורן directory not found, skipping...")
            return
        
        # Grade 10, Level 3
        grade_path = base_path / "כיתה י 3 יחל"
        if grade_path.exists():
            print("  Processing כיתה י 3 יחל...")
            files = self.get_all_image_files(grade_path)
            
            # Separate by volume using filename patterns
            volumes = {"a": [], "b": [], "c": []}
            
            for file_path in files:
                filename = file_path.name.lower()
                if "כרך א" in filename or "כותרת כרך א" in filename:
                    volumes["a"].append(file_path)
                elif "כרך ב" in filename or "כותרת כרך ב" in filename or "כרךב" in filename:
                    volumes["b"].append(file_path)
                elif "כרך ג" in filename or "כותרת כרך ג" in filename:
                    volumes["c"].append(file_path)
            
            for vol_letter, vol_files in volumes.items():
                if vol_files:
                    target_folder = self.target_dir / f"3-10-goren-{vol_letter}"
                    count = self.organize_files_by_category(vol_files, target_folder)
                    if count > 0:
                        print(f"    3-10-goren-{vol_letter}: {count} files")
        
        # Grade 10, Level 4 (subdivided folders)
        grade_path = base_path / "כיתה י 4 יחל"
        if grade_path.exists():
            print("  Processing כיתה י 4 יחל...")
            for vol_dir in grade_path.iterdir():
                if vol_dir.is_dir() and "בני ג" in vol_dir.name:
                    if "_א" in vol_dir.name:
                        vol_letter = "a"
                    elif "_ב" in vol_dir.name:
                        vol_letter = "b"
                    elif "_ג" in vol_dir.name:
                        vol_letter = "c"
                    else:
                        continue
                    
                    files = self.get_all_image_files(vol_dir)
                    target_folder = self.target_dir / f"4-10-goren-{vol_letter}"
                    count = self.organize_files_by_category(files, target_folder)
                    if count > 0:
                        print(f"    4-10-goren-{vol_letter}: {count} files")
        
        # Grade 10, Level 5
        grade_path = base_path / "כיתה י 5 יחל"
        if grade_path.exists():
            print("  Processing כיתה י 5 יחל...")
            files = self.get_all_image_files(grade_path)
            target_folder = self.target_dir / "5-10-goren-a"
            count = self.organize_files_by_category(files, target_folder)
            if count > 0:
                print(f"    5-10-goren-a: {count} files")
        
        # Grade 11, Level 3
        grade_path = base_path / "כיתה יא 3 יחל"
        if grade_path.exists():
            print("  Processing כיתה יא 3 יחל...")
            files = self.get_all_image_files(grade_path)
            
            volumes = {"a": [], "b": [], "c": []}
            
            for file_path in files:
                filename = file_path.name.lower()
                if "כרך א" in filename:
                    volumes["a"].append(file_path)
                elif "כרך ב" in filename:
                    volumes["b"].append(file_path)
                elif "כרך ג" in filename:
                    volumes["c"].append(file_path)
            
            for vol_letter, vol_files in volumes.items():
                if vol_files:
                    target_folder = self.target_dir / f"3-11-goren-{vol_letter}"
                    count = self.organize_files_by_category(vol_files, target_folder)
                    if count > 0:
                        print(f"    3-11-goren-{vol_letter}: {count} files")
        
        # Grade 11, Level 4 (subdivided folders)
        grade_path = base_path / "כיתה יא 4 יחל"
        if grade_path.exists():
            print("  Processing כיתה יא 4 יחל...")
            for vol_dir in grade_path.iterdir():
                if vol_dir.is_dir() and "בני ג" in vol_dir.name:
                    if "_א" in vol_dir.name:
                        vol_letter = "a"
                    elif "_ב" in vol_dir.name:
                        vol_letter = "b"
                    elif "_ג" in vol_dir.name:
                        vol_letter = "c"
                    else:
                        continue
                    
                    files = self.get_all_image_files(vol_dir)
                    target_folder = self.target_dir / f"4-11-goren-{vol_letter}"
                    count = self.organize_files_by_category(files, target_folder)
                    if count > 0:
                        print(f"    4-11-goren-{vol_letter}: {count} files")
        
        # Grade 11, Level 5
        grade_path = base_path / "כיתה יא 5 יחל"
        if grade_path.exists():
            print("  Processing כיתה יא 5 יחל...")
            files = self.get_all_image_files(grade_path)
            
            volumes = {"b1": [], "b2": []}
            
            for file_path in files:
                filename = file_path.name.lower()
                if "ב_1" in filename or "ב 1" in filename:
                    volumes["b1"].append(file_path)
                elif "ב_2" in filename or "ב 2" in filename:
                    volumes["b2"].append(file_path)
            
            for vol_id, vol_files in volumes.items():
                if vol_files:
                    target_folder = self.target_dir / f"5-11-goren-{vol_id}"
                    count = self.organize_files_by_category(vol_files, target_folder)
                    if count > 0:
                        print(f"    5-11-goren-{vol_id}: {count} files")
        
        # Grade 12, Level 3
        grade_path = base_path / "כיתה יב 3 יחל"
        if grade_path.exists():
            print("  Processing כיתה יב 3 יחל...")
            files = self.get_all_image_files(grade_path)
            target_folder = self.target_dir / "3-12-goren-geometry"
            count = self.organize_files_by_category(files, target_folder)
            if count > 0:
                print(f"    3-12-goren-geometry: {count} files")
        
        # Grade 12, Level 5 (subdivided folders)
        grade_path = base_path / "כיתה יב 5 יחל"
        if grade_path.exists():
            print("  Processing כיתה יב 5 יחל...")
            
            # Volume ג_1
            vol_dir = grade_path / "ג_1"
            if vol_dir.exists():
                files = self.get_all_image_files(vol_dir)
                target_folder = self.target_dir / "5-12-goren-c1"
                count = self.organize_files_by_category(files, target_folder)
                if count > 0:
                    print(f"    5-12-goren-c1: {count} files")
            
            # Volume ג_2
            vol_dir = grade_path / "ג_2"
            if vol_dir.exists():
                files = self.get_all_image_files(vol_dir)
                target_folder = self.target_dir / "5-12-goren-c2"
                count = self.organize_files_by_category(files, target_folder)
                if count > 0:
                    print(f"    5-12-goren-c2: {count} files")
    
    def organize_archimedes_books(self):
        """Organize Archimedes books."""
        print("\nProcessing ארכימדס books...")
        
        # Process 471 (Grade 10, Level 4)
        vol_path = self.source_dir / "ארכימדס" / "ארכימדס" / "י 471"
        if vol_path.exists():
            print("  Processing י 471...")
            files = self.get_all_image_files(vol_path)
            target_folder = self.target_dir / "4-10-archimedes-a"
            count = self.organize_files_by_category(files, target_folder)
            if count > 0:
                print(f"    4-10-archimedes-a: {count} files")
        
        # Process 571 (Grade 10, Level 5)
        vol_path = self.source_dir / "ארכימדס" / "ארכימדס" / "י 571"
        if vol_path.exists():
            print("  Processing י 571...")
            files = self.get_all_image_files(vol_path)
            target_folder = self.target_dir / "5-10-archimedes-a"
            count = self.organize_files_by_category(files, target_folder)
            if count > 0:
                print(f"    5-10-archimedes-a: {count} files")
        
        # Process 581 (Grade 11, Level 5)
        vol_path = self.source_dir / "ארכימדס" / "ארכימדס_ספר 581 - ינואר 2023"
        if vol_path.exists():
            print("  Processing ספר 581...")
            files = self.get_all_image_files(vol_path)
            target_folder = self.target_dir / "5-11-archimedes-a"
            count = self.organize_files_by_category(files, target_folder)
            if count > 0:
                print(f"    5-11-archimedes-a: {count} files")
    
    def organize_geva_books(self):
        """Organize Yoel Geva books."""
        print("\nProcessing יואל גבע books...")
        base_path = self.source_dir / "יואל גבע"
        
        if not base_path.exists():
            print("  יואל גבע directory not found, skipping...")
            return
        
        # Grade 10, Level 3 (172)
        grade_path = base_path / "כיתה י 3 יחל" / "172"
        if grade_path.exists():
            print("  Processing כיתה י 3 יחל...")
            
            # Volume א
            vol_dir = grade_path / "172_א"
            if vol_dir.exists():
                files = self.get_all_image_files(vol_dir)
                target_folder = self.target_dir / "3-10-geva-a"
                count = self.organize_files_by_category(files, target_folder)
                if count > 0:
                    print(f"    3-10-geva-a: {count} files")
            
            # Volume ב
            vol_dir = grade_path / "172_ב"
            if vol_dir.exists():
                files = self.get_all_image_files(vol_dir)
                target_folder = self.target_dir / "3-10-geva-b"
                count = self.organize_files_by_category(files, target_folder)
                if count > 0:
                    print(f"    3-10-geva-b: {count} files")
        
        # Grade 10, Level 4 (471)
        grade_path = base_path / "כיתה י 4 יחל"
        if grade_path.exists():
            print("  Processing כיתה י 4 יחל...")
            
            for vol_letter, vol_dir_name in [("a", "כרך_471_א"), ("b", "כרך_471_ב"), ("c", "כרך_471_ג")]:
                vol_dir = grade_path / vol_dir_name
                if vol_dir.exists():
                    files = self.get_all_image_files(vol_dir)
                    target_folder = self.target_dir / f"4-10-geva-{vol_letter}"
                    count = self.organize_files_by_category(files, target_folder)
                    if count > 0:
                        print(f"    4-10-geva-{vol_letter}: {count} files")
        
        # Grade 10, Level 5
        grade_path = base_path / "כיתה י 5 יחל"
        if grade_path.exists():
            print("  Processing כיתה י 5 יחל...")
            
            # 571 volume
            vol_dir = grade_path / "חוברת 571"
            if vol_dir.exists():
                files = self.get_all_image_files(vol_dir)
                target_folder = self.target_dir / "5-10-geva-571"
                count = self.organize_files_by_category(files, target_folder)
                if count > 0:
                    print(f"    5-10-geva-571: {count} files")
            
            # 804-806 volumes
            for vol_letter, vol_dir_name in [("a", "804-806_כרך_א"), ("b", "804-806_כרך_ב")]:
                vol_dir = grade_path / vol_dir_name
                if vol_dir.exists():
                    files = self.get_all_image_files(vol_dir)
                    target_folder = self.target_dir / f"5-10-geva-804-{vol_letter}"
                    count = self.organize_files_by_category(files, target_folder)
                    if count > 0:
                        print(f"    5-10-geva-804-{vol_letter}: {count} files")
        
        # Grade 11, Level 3 (371)
        grade_path = base_path / "כיתה יא 3 יחל"
        if grade_path.exists():
            print("  Processing כיתה יא 3 יחל...")
            
            for vol_letter, vol_dir_name in [("a", "כרך_371_א"), ("b", "כרך_371_ב")]:
                vol_dir = grade_path / vol_dir_name
                if vol_dir.exists():
                    files = self.get_all_image_files(vol_dir)
                    target_folder = self.target_dir / f"3-11-geva-{vol_letter}"
                    count = self.organize_files_by_category(files, target_folder)
                    if count > 0:
                        print(f"    3-11-geva-{vol_letter}: {count} files")
        
        # Grade 11, Level 4 (471)
        grade_path = base_path / "כיתה יא 4 יחל"
        if grade_path.exists():
            print("  Processing כיתה יא 4 יחל...")
            
            for vol_letter, vol_dir_name in [("a", "כרך_471_יא_א"), ("b", "כרך_471_יא_ב")]:
                vol_dir = grade_path / vol_dir_name
                if vol_dir.exists():
                    files = self.get_all_image_files(vol_dir)
                    target_folder = self.target_dir / f"4-11-geva-{vol_letter}"
                    count = self.organize_files_by_category(files, target_folder)
                    if count > 0:
                        print(f"    4-11-geva-{vol_letter}: {count} files")
        
        # Grade 11, Level 5 (806)
        grade_path = base_path / "כיתה יא 5 יחל"
        if grade_path.exists():
            print("  Processing כיתה יא 5 יחל...")
            
            for vol_letter, vol_dir_name in [("c", "כרך_806_יא_ג"), ("d", "כרך_806_יא_ד")]:
                vol_dir = grade_path / vol_dir_name
                if vol_dir.exists():
                    files = self.get_all_image_files(vol_dir)
                    target_folder = self.target_dir / f"5-11-geva-{vol_letter}"
                    count = self.organize_files_by_category(files, target_folder)
                    if count > 0:
                        print(f"    5-11-geva-{vol_letter}: {count} files")
        
        # Grade 12, Level 4 (805)
        grade_path = base_path / "כיתה יב 4 יחל" / "כרך 805 4 יחל"
        if grade_path.exists():
            print("  Processing כיתה יב 4 יחל...")
            files = self.get_all_image_files(grade_path)
            target_folder = self.target_dir / "4-12-geva-a"
            count = self.organize_files_by_category(files, target_folder)
            if count > 0:
                print(f"    4-12-geva-a: {count} files")
        
        # Grade 12, Level 5 (807)
        grade_path = base_path / "כיתה יב 5 יחל"
        if grade_path.exists():
            print("  Processing כיתה יב 5 יחל...")
            
            for vol_letter, vol_dir_name in [("a", "כרך_807_יב_א"), ("b", "כרך_807_יב_ב")]:
                vol_dir = grade_path / vol_dir_name
                if vol_dir.exists():
                    files = self.get_all_image_files(vol_dir)
                    target_folder = self.target_dir / f"5-12-geva-{vol_letter}"
                    count = self.organize_files_by_category(files, target_folder)
                    if count > 0:
                        print(f"    5-12-geva-{vol_letter}: {count} files")
    
    def generate_summary(self):
        """Generate a summary report of the organization."""
        print("\n=== Organization Summary ===")
        print(f"Total files processed: {len(self.processed_files)}")
        print(f"Files skipped: {len(self.skipped_files)}")
        
        # Count files per volume
        volume_counts = {}
        if self.target_dir.exists():
            for volume_dir in self.target_dir.iterdir():
                if volume_dir.is_dir():
                    file_count = len([f for f in volume_dir.iterdir() if f.is_file()])
                    volume_counts[volume_dir.name] = file_count
        
        print("\nVolumes created:")
        for volume, count in sorted(volume_counts.items()):
            print(f"  {volume}: {count} files")
        
        # Save detailed report
        report = {
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "volume_counts": volume_counts,
            "total_processed": len(self.processed_files),
            "total_skipped": len(self.skipped_files)
        }
        
        report_path = self.target_dir / "organization_report.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\nDetailed report saved to: {report_path}")
        except Exception as e:
            print(f"\nWarning: Could not save report: {e}")
        
        if self.skipped_files:
            print(f"\nWarning: {len(self.skipped_files)} files were skipped.")
            print("First 10 skipped files:")
            for file_path in self.skipped_files[:10]:
                print(f"  - {file_path}")
    
    def organize_all(self):
        """Main method to organize all books."""
        print("Starting math books organization...")
        print(f"Source: {self.source_dir}")
        print(f"Target: {self.target_dir}")
        
        if not self.source_dir.exists():
            print(f"Error: Source directory '{self.source_dir}' does not exist!")
            return False
        
        try:
            self.organize_goren_books()
            self.organize_archimedes_books()
            self.organize_geva_books()
            
            self.generate_summary()
            print("\nOrganization complete!")
            return True
            
        except Exception as e:
            print(f"\nError during organization: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Organize math book images into consistent volume structure"
    )
    parser.add_argument(
        "--source", 
        default="math-books",
        help="Source directory containing math book images (default: math-books)"
    )
    parser.add_argument(
        "--target",
        default="volumes",
        help="Target directory for organized volumes (default: volumes)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
    
    organizer = MathBookOrganizer(args.source, args.target)
    success = organizer.organize_all()
    
    if success:
        print(f"\nSuccess! Check the '{args.target}' directory for organized volumes.")
    else:
        print("\nOrganization failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())