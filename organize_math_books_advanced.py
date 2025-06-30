#!/usr/bin/env python3
"""
Advanced Math Books Image Organizer
Organizes math book images (covers, intros, ToCs) into a consistent volume-based structure.
Uses OCR or image analysis to better categorize ambiguous files.
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
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5"
}

# Grade mappings
GRADE_MAP = {
    "י": "10", "יא": "11", "יב": "12",
    "10": "10", "11": "11", "12": "12"
}

# Category keywords
CATEGORY_KEYWORDS = {
    "cover": ["כריכה", "cover", "כותרת", "שער"],
    "intro": ["מבוא", "פרומפט", "intro", "הקדמה"],
    "toc": ["תוכן", "עניינים", "toc", "content", "ענינים"]
}


class MathBookOrganizer:
    def __init__(self, source_dir: str, target_dir: str, use_ocr: bool = False):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.use_ocr = use_ocr
        self.target_dir.mkdir(exist_ok=True)
        
        # Track processed files for summary
        self.processed_files = []
        self.skipped_files = []
        
    def clean_filename(self, filename: str) -> str:
        """Remove special characters and normalize filename."""
        # Remove Hebrew special characters
        cleaned = re.sub(r'^[‏\u200f\u200e]+', '', filename)
        # Remove leading numbers
        cleaned = re.sub(r'^\d+', '', cleaned)
        return cleaned.strip()
    
    def get_category(self, filename: str, file_path: Optional[Path] = None) -> str:
        """Determine file category from filename or content."""
        cleaned = self.clean_filename(filename)
        
        # Check filename for category keywords
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in cleaned.lower():
                    return category
        
        # If use_ocr is enabled and file_path provided, try reading the image
        if self.use_ocr and file_path and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Here you would implement OCR logic
            # For now, we'll just return unknown
            pass
            
        return "unknown"
    
    def extract_number(self, filename: str, category: str) -> int:
        """Extract ordering number from filename."""
        # Look for numbers after underscore or space
        patterns = [
            r'_(\d+)',  # number after underscore
            r'\s(\d+)\.',  # number before extension
            rf'{category}.*?(\d+)',  # number after category word
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
        category = self.get_category(filename, source_path)
        
        if category == "unknown":
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
        
        # Copy file
        print(f"  Copying: {filename} -> {target_filename}")
        shutil.copy2(source_path, target_path)
        
        self.processed_files.append({
            "source": str(source_path),
            "target": str(target_path),
            "category": category
        })
        
        return True
    
    def process_volume_directory(self, source_dir: Path, volume_id: str,
                                file_patterns: List[str]) -> int:
        """Process files from a directory into a volume folder."""
        target_folder = self.target_dir / volume_id
        target_folder.mkdir(exist_ok=True)
        
        processed_count = 0
        category_counts = {}
        
        # Collect all matching files
        all_files = []
        for pattern in file_patterns:
            for file_path in source_dir.glob(pattern):
                if file_path.is_file():
                    all_files.append(file_path)
        
        # Sort files to maintain order
        all_files.sort(key=lambda x: self.clean_filename(x.name))
        
        # Group files by category first
        categorized_files = {"cover": [], "intro": [], "toc": []}
        
        for file_path in all_files:
            category = self.get_category(file_path.name, file_path)
            if category in categorized_files:
                categorized_files[category].append(file_path)
        
        # Process files in order: cover, intro, toc
        for category in ["cover", "intro", "toc"]:
            for file_path in categorized_files[category]:
                if self.process_file(file_path, target_folder, category_counts):
                    processed_count += 1
        
        return processed_count
    
    def organize_goren_books(self):
        """Organize Beni Goren books."""
        print("\nProcessing בני גורן books...")
        base_path = self.source_dir / "בני גורן" / "בני גורן תוכן עניינים"
        
        # Define volume configurations
        volume_configs = [
            # Grade 10, Level 3
            ("כיתה י 3 יחל", [
                ("3-10-goren-a", ["*כרך א*", "*כותרת כרך א*"]),
                ("3-10-goren-b", ["*כרך*ב*", "*כותרת כרך*ב*"]),
                ("3-10-goren-c", ["*כרך ג*", "*כותרת כרך ג*"])
            ]),
            # Grade 11, Level 3
            ("כיתה יא 3 יחל", [
                ("3-11-goren-a", ["*כרך א*"]),
                ("3-11-goren-b", ["*כרך*ב*", "*כרך ב*"]),
                ("3-11-goren-c", ["*כרך*ג*", "*כרך ג*"])
            ]),
            # Grade 10, Level 5
            ("כיתה י 5 יחל", [
                ("5-10-goren-804", ["*804*", "*806*"])
            ]),
            # Grade 11, Level 5
            ("כיתה יא 5 יחל", [
                ("5-11-goren-b1", ["*ב_1*", "*ב 1*"]),
                ("5-11-goren-b2", ["*ב_2*", "*ב 2*"])
            ])
        ]
        
        for grade_dir, volumes in volume_configs:
            grade_path = base_path / grade_dir
            if grade_path.exists():
                print(f"  Processing {grade_dir}...")
                for volume_id, patterns in volumes:
                    count = self.process_volume_directory(grade_path, volume_id, patterns)
                    if count > 0:
                        print(f"    {volume_id}: {count} files")
        
        # Process subdivided volumes (Grade 10, Level 4)
        grade_10_4_path = base_path / "כיתה י 4 יחל"
        if grade_10_4_path.exists():
            print("  Processing כיתה י 4 יחל...")
            for vol_dir in grade_10_4_path.iterdir():
                if vol_dir.is_dir() and "בני ג" in vol_dir.name:
                    if "_א" in vol_dir.name or vol_dir.name.endswith("_א"):
                        volume_id = "4-10-goren-a"
                    elif "_ב" in vol_dir.name or vol_dir.name.endswith("_ב"):
                        volume_id = "4-10-goren-b"
                    elif "_ג" in vol_dir.name or vol_dir.name.endswith("_ג"):
                        volume_id = "4-10-goren-c"
                    else:
                        continue
                    
                    count = self.process_volume_directory(vol_dir, volume_id, ["*.JPG", "*.jpg", "*.png"])
                    if count > 0:
                        print(f"    {volume_id}: {count} files")
        
        # Similar processing for Grade 11, Level 4
        grade_11_4_path = base_path / "כיתה יא 4 יחל"
        if grade_11_4_path.exists():
            print("  Processing כיתה יא 4 יחל...")
            for vol_dir in grade_11_4_path.iterdir():
                if vol_dir.is_dir() and "בני ג" in vol_dir.name:
                    if "_א" in vol_dir.name:
                        volume_id = "4-11-goren-a"
                    elif "_ב" in vol_dir.name:
                        volume_id = "4-11-goren-b"
                    elif "_ג" in vol_dir.name:
                        volume_id = "4-11-goren-c"
                    else:
                        continue
                    
                    count = self.process_volume_directory(vol_dir, volume_id, ["*.JPG", "*.jpg", "*.png"])
                    if count > 0:
                        print(f"    {volume_id}: {count} files")
        
        # Process Grade 12 volumes
        grade_12_5_path = base_path / "כיתה יב 5 יחל"
        if grade_12_5_path.exists():
            print("  Processing כיתה יב 5 יחל...")
            if (grade_12_5_path / "ג_1").exists():
                count = self.process_volume_directory(
                    grade_12_5_path / "ג_1", "5-12-goren-c1", ["*.JPG", "*.jpg", "*.png"]
                )
                if count > 0:
                    print(f"    5-12-goren-c1: {count} files")
            
            if (grade_12_5_path / "ג_2").exists():
                count = self.process_volume_directory(
                    grade_12_5_path / "ג_2", "5-12-goren-c2", ["*.JPG", "*.jpg", "*.png"]
                )
                if count > 0:
                    print(f"    5-12-goren-c2: {count} files")
    
    def organize_archimedes_books(self):
        """Organize Archimedes books."""
        print("\nProcessing ארכימדס books...")
        base_path = self.source_dir / "ארכימדס" / "ארכימדס"
        
        # Process 471 (Grade 10, Level 4)
        if (base_path / "י 471").exists():
            count = self.process_volume_directory(
                base_path / "י 471", "4-10-archimedes-a", ["*.pdf", "*.PDF"]
            )
            if count > 0:
                print(f"  4-10-archimedes-a: {count} files")
        
        # Process 571 (Grade 10, Level 5)
        if (base_path / "י 571").exists():
            count = self.process_volume_directory(
                base_path / "י 571", "5-10-archimedes-a", 
                ["*.pdf", "*.PDF", "*.jpg", "*.png", "*.JPG", "*.PNG"]
            )
            if count > 0:
                print(f"  5-10-archimedes-a: {count} files")
        
        # Process 581 (Grade 11, Level 5)
        archimedes_581_path = self.source_dir / "ארכימדס" / "ארכימדס_ספר 581 - ינואר 2023"
        if archimedes_581_path.exists():
            target_folder = self.target_dir / "5-11-archimedes-a"
            target_folder.mkdir(exist_ok=True)
            
            # Process cover
            cover_file = archimedes_581_path / "Archimedes_Book_cover_581_new-1.png"
            if cover_file.exists():
                shutil.copy2(cover_file, target_folder / "cover-1.png")
            
            # Process M files as ToC
            toc_count = 0
            for i in range(1, 11):
                m_file = archimedes_581_path / f"M{i}.pdf"
                if m_file.exists():
                    toc_count += 1
                    shutil.copy2(m_file, target_folder / f"toc-{toc_count}.pdf")
            
            if toc_count > 0:
                print(f"  5-11-archimedes-a: {toc_count + 1} files")
    
    def organize_geva_books(self):
        """Organize Yoel Geva books."""
        print("\nProcessing יואל גבע books...")
        base_path = self.source_dir / "יואל גבע"
        
        # Grade 10, Level 3
        if (base_path / "כיתה י 3 יחל" / "172").exists():
            print("  Processing כיתה י 3 יחל...")
            
            # Volume א
            if (base_path / "כיתה י 3 יחל" / "172" / "172_א").exists():
                count = self.process_volume_directory(
                    base_path / "כיתה י 3 יחל" / "172" / "172_א",
                    "3-10-geva-a", ["*.JPG", "*.jpg", "*.png"]
                )
                if count > 0:
                    print(f"    3-10-geva-a: {count} files")
            
            # Volume ב
            if (base_path / "כיתה י 3 יחל" / "172" / "172_ב").exists():
                count = self.process_volume_directory(
                    base_path / "כיתה י 3 יחל" / "172" / "172_ב",
                    "3-10-geva-b", ["*.JPG", "*.jpg", "*.png"]
                )
                if count > 0:
                    print(f"    3-10-geva-b: {count} files")
        
        # Grade 10, Level 4
        if (base_path / "כיתה י 4 יחל").exists():
            print("  Processing כיתה י 4 יחל...")
            
            for vol_letter, vol_dir in [("a", "כרך_471_א"), ("b", "כרך_471_ב"), ("c", "כרך_471_ג")]:
                vol_path = base_path / "כיתה י 4 יחל" / vol_dir
                if vol_path.exists():
                    count = self.process_volume_directory(
                        vol_path, f"4-10-geva-{vol_letter}", ["*.JPG", "*.jpg", "*.png"]
                    )
                    if count > 0:
                        print(f"    4-10-geva-{vol_letter}: {count} files")
        
        # Grade 10, Level 5
        if (base_path / "כיתה י 5 יחל").exists():
            print("  Processing כיתה י 5 יחל...")
            
            # 571 volume
            if (base_path / "כיתה י 5 יחל" / "חוברת 571").exists():
                count = self.process_volume_directory(
                    base_path / "כיתה י 5 יחל" / "חוברת 571",
                    "5-10-geva-571", ["*.JPG", "*.jpg", "*.png"]
                )
                if count > 0:
                    print(f"    5-10-geva-571: {count} files")
            
            # 804-806 volumes
            for vol_letter, vol_dir in [("a", "804-806_כרך_א"), ("b", "804-806_כרך_ב")]:
                vol_path = base_path / "כיתה י 5 יחל" / vol_dir
                if vol_path.exists():
                    count = self.process_volume_directory(
                        vol_path, f"5-10-geva-804-{vol_letter}", ["*.JPG", "*.jpg", "*.png"]
                    )
                    if count > 0:
                        print(f"    5-10-geva-804-{vol_letter}: {count} files")
        
        # Process other grades similarly...
        # Grade 11 and 12 processing would follow the same pattern
        
    def generate_summary(self):
        """Generate a summary report of the organization."""
        print("\n=== Organization Summary ===")
        print(f"Total files processed: {len(self.processed_files)}")
        print(f"Files skipped: {len(self.skipped_files)}")
        
        # Count files per volume
        volume_counts = {}
        for volume_dir in self.target_dir.iterdir():
            if volume_dir.is_dir():
                file_count = len(list(volume_dir.glob("*")))
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
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
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
        
        self.organize_goren_books()
        self.organize_archimedes_books()
        self.organize_geva_books()
        
        self.generate_summary()
        print("\nOrganization complete!")


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
        "--use-ocr",
        action="store_true",
        help="Use OCR to better categorize ambiguous files (requires additional setup)"
    )
    
    args = parser.parse_args()
    
    organizer = MathBookOrganizer(args.source, args.target, args.use_ocr)
    organizer.organize_all()


if __name__ == "__main__":
    main()