#!/usr/bin/env python3
"""
Label Studio OCR to VietOCR Format Converter

This script converts exported Label Studio OCR annotations from JSON format
to VietOCR training format (tab-separated annotation files).

Usage:
    python label_studio_to_vietocr_converter.py input.json output_dir [options]

Requirements:
    - Label Studio JSON export file
    - Images referenced in the JSON file
"""

import json
import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LabelStudioToVietOCRConverter:
    """Converter class for Label Studio OCR annotations to VietOCR format."""
    
    def __init__(self, input_json_path: str, output_dir: str, image_dir: Optional[str] = None):
        """
        Initialize the converter.
        
        Args:
            input_json_path: Path to Label Studio JSON export file
            output_dir: Directory to save VietOCR annotation files
            image_dir: Directory containing images (if different from JSON references)
        """
        self.input_json_path = Path(input_json_path)
        self.output_dir = Path(output_dir)
        self.image_dir = Path(image_dir) if image_dir else None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'processed_tasks': 0,
            'skipped_tasks': 0,
            'total_annotations': 0,
            'valid_annotations': 0
        }
    
    def load_label_studio_data(self) -> List[Dict[str, Any]]:
        """Load and parse Label Studio JSON export."""
        try:
            with open(self.input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'data' in data:
                return [data]
            else:
                logger.error("Unexpected JSON structure. Expected list of tasks or single task object.")
                return []
                
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_json_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            return []
    
    def extract_image_path(self, task: Dict[str, Any]) -> Optional[str]:
        """
        Extract image path from Label Studio task.
        
        Args:
            task: Label Studio task object
            
        Returns:
            Image path or None if not found
        """
        # Common patterns for image paths in Label Studio
        data = task.get('data', {})
        
        # Try different possible keys for image path
        possible_keys = ['image', 'ocr', 'img', 'image_url', 'image_path', 'data']
        
        for key in possible_keys:
            if key in data:
                image_path = data[key]
                if isinstance(image_path, str):
                    return image_path
                elif isinstance(image_path, dict) and 'url' in image_path:
                    return image_path['url']
        
        # If no direct path found, look for nested structures
        for key, value in data.items():
            if isinstance(value, str) and any(ext in value.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
                return value
        
        return None
    
    def extract_annotations(self, task: Dict[str, Any]) -> List[str]:
        """
        Extract text annotations from Label Studio task.
        
        Args:
            task: Label Studio task object
            
        Returns:
            List of text annotations
        """
        annotations = []
        
        # Look for annotations in the task
        task_annotations = task.get('annotations', [])
        
        for annotation in task_annotations:
            results = annotation.get('result', [])
            
            for result in results:
                # Extract text from different possible structures
                value = result.get('value', {})
                
                if 'text' in value:
                    text = value['text']
                    if isinstance(text, list):
                        # Join list of text segments
                        text = ' '.join(str(t) for t in text if t)
                    elif isinstance(text, str):
                        text = text.strip()
                    else:
                        continue
                    
                    if text:
                        annotations.append(text)
                
                # Alternative: look for 'transcription' or 'label' keys
                elif 'transcription' in value:
                    text = value['transcription']
                    if isinstance(text, str) and text.strip():
                        annotations.append(text.strip())
                
                elif 'label' in value:
                    text = value['label']
                    if isinstance(text, str) and text.strip():
                        annotations.append(text.strip())
        
        return annotations
    
    def resolve_image_path(self, image_path: str) -> Optional[str]:
        """
        Resolve the actual image path, considering the image directory.
        
        Args:
            image_path: Original image path from JSON
            
        Returns:
            Resolved image path or None if not found
        """
        # Handle S3 URLs by extracting filename
        if image_path.startswith('s3://'):
            # Extract filename from S3 URL
            # s3://bucket/path/filename.ext -> filename.ext
            filename = Path(image_path).name
            logger.debug(f"Extracted filename from S3 URL: {filename}")
        else:
            filename = Path(image_path).name
        
        # If image_dir is specified, use it as base
        if self.image_dir:
            resolved_path = self.image_dir / filename
            
            if resolved_path.exists():
                logger.debug(f"Found image at: {resolved_path}")
                return str(resolved_path)
            else:
                logger.debug(f"Image not found at: {resolved_path}")
        
        # Try original path (for non-S3 URLs)
        if not image_path.startswith('s3://') and Path(image_path).exists():
            return image_path
        
        # Try relative to input JSON directory
        json_dir = self.input_json_path.parent
        relative_path = json_dir / filename
        if relative_path.exists():
            return str(relative_path)
        
        return None
    
    def convert(self) -> bool:
        """
        Convert Label Studio JSON to VietOCR format.
        
        Returns:
            True if conversion was successful, False otherwise
        """
        logger.info(f"Loading Label Studio data from: {self.input_json_path}")
        tasks = self.load_label_studio_data()
        
        if not tasks:
            logger.error("No valid tasks found in the input file.")
            return False
        
        self.stats['total_tasks'] = len(tasks)
        
        # Prepare annotation data
        annotation_data = []
        
        for i, task in enumerate(tasks):
            logger.info(f"Processing task {i+1}/{len(tasks)}")
            
            # Extract image path
            image_path = self.extract_image_path(task)
            if not image_path:
                logger.warning(f"Task {i+1}: No image path found, skipping")
                self.stats['skipped_tasks'] += 1
                continue
            
            # Resolve actual image path
            resolved_path = self.resolve_image_path(image_path)
            if not resolved_path:
                logger.warning(f"Task {i+1}: Image not found: {image_path}, skipping")
                self.stats['skipped_tasks'] += 1
                continue
            
            # Extract annotations
            annotations = self.extract_annotations(task)
            if not annotations:
                logger.warning(f"Task {i+1}: No annotations found, skipping")
                self.stats['skipped_tasks'] += 1
                continue
            
            self.stats['total_annotations'] += len(annotations)
            
            # For each annotation, create an entry
            for annotation in annotations:
                # Clean the annotation text
                clean_text = annotation.strip()
                if clean_text:
                    annotation_data.append((resolved_path, clean_text))
                    self.stats['valid_annotations'] += 1
            
            self.stats['processed_tasks'] += 1
        
        # Write annotation files
        self.write_annotation_files(annotation_data)
        
        # Print statistics
        self.print_statistics()
        
        return True
    
    def write_annotation_files(self, annotation_data: List[tuple]):
        """
        Write annotation data to VietOCR format files.
        
        Args:
            annotation_data: List of (image_path, text) tuples
        """
        if not annotation_data:
            logger.warning("No annotation data to write.")
            return
        
        # Create train/val split (80/20 by default)
        total_samples = len(annotation_data)
        train_size = int(total_samples * 0.8)
        
        train_data = annotation_data[:train_size]
        val_data = annotation_data[train_size:]
        
        # Write training annotation file
        train_file = self.output_dir / "annotation_train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for image_path, text in train_data:
                # Use relative path from output directory
                rel_path = os.path.relpath(image_path, self.output_dir)
                f.write(f"{rel_path}\t{text}\n")
        
        logger.info(f"Written {len(train_data)} training samples to: {train_file}")
        
        # Write validation annotation file
        if val_data:
            val_file = self.output_dir / "annotation_val.txt"
            with open(val_file, 'w', encoding='utf-8') as f:
                for image_path, text in val_data:
                    rel_path = os.path.relpath(image_path, self.output_dir)
                    f.write(f"{rel_path}\t{text}\n")
            
            logger.info(f"Written {len(val_data)} validation samples to: {val_file}")
        
        # Write combined annotation file
        combined_file = self.output_dir / "annotation_all.txt"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for image_path, text in annotation_data:
                rel_path = os.path.relpath(image_path, self.output_dir)
                f.write(f"{rel_path}\t{text}\n")
        
        logger.info(f"Written {len(annotation_data)} total samples to: {combined_file}")
    
    def print_statistics(self):
        """Print conversion statistics."""
        logger.info("=" * 50)
        logger.info("CONVERSION STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total tasks processed: {self.stats['total_tasks']}")
        logger.info(f"Successfully processed: {self.stats['processed_tasks']}")
        logger.info(f"Skipped tasks: {self.stats['skipped_tasks']}")
        logger.info(f"Total annotations found: {self.stats['total_annotations']}")
        logger.info(f"Valid annotations: {self.stats['valid_annotations']}")
        
        if self.stats['total_tasks'] > 0:
            success_rate = (self.stats['processed_tasks'] / self.stats['total_tasks']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("=" * 50)


def main():
    """Main function to run the converter."""
    parser = argparse.ArgumentParser(
        description="Convert Label Studio OCR annotations to VietOCR format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python label_studio_to_vietocr_converter.py export.json ./vietocr_data/
    
    # With custom image directory
    python label_studio_to_vietocr_converter.py export.json ./vietocr_data/ --image-dir ./images/
    
    # Handle S3 URLs with local images
    python label_studio_to_vietocr_converter.py export.json ./vietocr_data/ --image-dir "G:\\datasets\\images\\"
    
    # Verbose output
    python label_studio_to_vietocr_converter.py export.json ./vietocr_data/ --verbose
        """
    )
    
    parser.add_argument('input_json', help='Path to Label Studio JSON export file')
    parser.add_argument('output_dir', help='Output directory for VietOCR annotation files')
    parser.add_argument('--image-dir', help='Directory containing images (if different from JSON references). Handles S3 URLs by extracting filename.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input_json):
        logger.error(f"Input file does not exist: {args.input_json}")
        sys.exit(1)
    
    # Create converter and run conversion
    converter = LabelStudioToVietOCRConverter(
        input_json_path=args.input_json,
        output_dir=args.output_dir,
        image_dir=args.image_dir
    )
    
    success = converter.convert()
    
    if success:
        logger.info("Conversion completed successfully!")
        logger.info(f"Output files are in: {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Copy your images to the output directory")
        logger.info("2. Update VietOCR config file to point to the annotation files")
        logger.info("3. Run VietOCR training with the converted data")
    else:
        logger.error("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
