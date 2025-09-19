#!/usr/bin/env python3
"""
Text Region Cropping Script for Label Studio OCR Data

This script reads Label Studio JSON exports with coordinate information,
crops individual text regions from images, and creates a new dataset
suitable for OCR training.

Usage:
    python crop_text_regions.py input.json output_dir [options]
"""

import json
import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextRegionCropper:
    """Class for cropping text regions from images based on Label Studio annotations."""
    
    def __init__(self, input_json_path: str, output_dir: str, image_dir: Optional[str] = None):
        """
        Initialize the text region cropper.
        
        Args:
            input_json_path: Path to Label Studio JSON export file
            output_dir: Directory to save cropped images and annotations
            image_dir: Directory containing images (if different from JSON references)
        """
        self.input_json_path = Path(input_json_path)
        self.output_dir = Path(output_dir)
        self.image_dir = Path(image_dir) if image_dir else None
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cropped_images_dir = self.output_dir / "cropped_images"
        self.cropped_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'processed_tasks': 0,
            'skipped_tasks': 0,
            'total_regions': 0,
            'cropped_regions': 0,
            'failed_crops': 0
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
        """Extract image path from Label Studio task."""
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
    
    def resolve_image_path(self, image_path: str) -> Optional[str]:
        """Resolve the actual image path, considering the image directory."""
        # Handle S3 URLs by extracting filename
        if image_path.startswith('s3://'):
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
    
    def extract_text_regions(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract text regions with coordinates from Label Studio task.
        
        Args:
            task: Label Studio task object
            
        Returns:
            List of text regions with coordinates and text
        """
        regions = []
        
        # Look for annotations in the task
        task_annotations = task.get('annotations', [])
        
        for annotation in task_annotations:
            results = annotation.get('result', [])
            
            for result in results:
                # Extract text and coordinates
                value = result.get('value', {})
                
                # Get text content
                text = None
                if 'text' in value:
                    text = value['text']
                    if isinstance(text, list):
                        text = ' '.join(str(t) for t in text if t)
                    elif isinstance(text, str):
                        text = text.strip()
                elif 'transcription' in value:
                    text = value['transcription']
                    if isinstance(text, str):
                        text = text.strip()
                elif 'label' in value:
                    text = value['label']
                    if isinstance(text, str):
                        text = text.strip()
                
                if not text:
                    continue
                
                # Get coordinates
                coordinates = None
                if 'x' in value and 'y' in value and 'width' in value and 'height' in value:
                    # Rectangle coordinates
                    coordinates = {
                        'type': 'rectangle',
                        'x': float(value['x']),
                        'y': float(value['y']),
                        'width': float(value['width']),
                        'height': float(value['height'])
                    }
                elif 'points' in value:
                    # Polygon coordinates
                    points = value['points']
                    if isinstance(points, list) and len(points) >= 3:
                        coordinates = {
                            'type': 'polygon',
                            'points': points
                        }
                elif 'bbox' in value:
                    # Bounding box coordinates
                    bbox = value['bbox']
                    if isinstance(bbox, dict):
                        coordinates = {
                            'type': 'bbox',
                            'x': float(bbox.get('x', 0)),
                            'y': float(bbox.get('y', 0)),
                            'width': float(bbox.get('width', 0)),
                            'height': float(bbox.get('height', 0))
                        }
                
                if coordinates and text:
                    regions.append({
                        'text': text,
                        'coordinates': coordinates,
                        'original_width': value.get('original_width', 0),
                        'original_height': value.get('original_height', 0)
                    })
        
        return regions
    
    def crop_region(self, image: Image.Image, coordinates: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Crop a region from an image based on coordinates.
        
        Args:
            image: PIL Image object
            coordinates: Coordinate information
            
        Returns:
            Cropped PIL Image or None if cropping fails
        """
        try:
            img_width, img_height = image.size
            
            if coordinates['type'] == 'rectangle':
                # Convert percentage coordinates to pixel coordinates
                x = int(coordinates['x'] * img_width / 100)
                y = int(coordinates['y'] * img_height / 100)
                width = int(coordinates['width'] * img_width / 100)
                height = int(coordinates['height'] * img_height / 100)
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, img_width))
                y = max(0, min(y, img_height))
                width = max(1, min(width, img_width - x))
                height = max(1, min(height, img_height - y))
                
                return image.crop((x, y, x + width, y + height))
            
            elif coordinates['type'] == 'polygon':
                # Convert polygon to bounding box
                points = coordinates['points']
                if len(points) >= 3:
                    # Convert percentage coordinates to pixel coordinates
                    pixel_points = []
                    for point in points:
                        if isinstance(point, dict):
                            x = int(point.get('x', 0) * img_width / 100)
                            y = int(point.get('y', 0) * img_height / 100)
                        else:
                            x = int(point[0] * img_width / 100)
                            y = int(point[1] * img_height / 100)
                        pixel_points.append((x, y))
                    
                    # Get bounding box
                    min_x = min(p[0] for p in pixel_points)
                    max_x = max(p[0] for p in pixel_points)
                    min_y = min(p[1] for p in pixel_points)
                    max_y = max(p[1] for p in pixel_points)
                    
                    # Ensure coordinates are within image bounds
                    min_x = max(0, min(min_x, img_width))
                    min_y = max(0, min(min_y, img_height))
                    max_x = max(1, min(max_x, img_width))
                    max_y = max(1, min(max_y, img_height))
                    
                    return image.crop((min_x, min_y, max_x, max_y))
            
            elif coordinates['type'] == 'bbox':
                # Direct pixel coordinates
                x = int(coordinates['x'])
                y = int(coordinates['y'])
                width = int(coordinates['width'])
                height = int(coordinates['height'])
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, img_width))
                y = max(0, min(y, img_height))
                width = max(1, min(width, img_width - x))
                height = max(1, min(height, img_height - y))
                
                return image.crop((x, y, x + width, y + height))
            
        except Exception as e:
            logger.error(f"Error cropping region: {e}")
            return None
        
        return None
    
    def process_task(self, task: Dict[str, Any], task_index: int) -> List[Tuple[str, str]]:
        """
        Process a single task and crop all text regions.
        
        Args:
            task: Label Studio task object
            task_index: Index of the task for naming
            
        Returns:
            List of (cropped_image_path, text) tuples
        """
        # Extract image path
        image_path = self.extract_image_path(task)
        if not image_path:
            logger.warning(f"Task {task_index}: No image path found, skipping")
            self.stats['skipped_tasks'] += 1
            return []
        
        # Resolve actual image path
        resolved_path = self.resolve_image_path(image_path)
        if not resolved_path:
            logger.warning(f"Task {task_index}: Image not found: {image_path}, skipping")
            self.stats['skipped_tasks'] += 1
            return []
        
        # Load image
        try:
            image = Image.open(resolved_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            logger.error(f"Task {task_index}: Error loading image {resolved_path}: {e}")
            self.stats['skipped_tasks'] += 1
            return []
        
        # Extract text regions
        regions = self.extract_text_regions(task)
        if not regions:
            logger.warning(f"Task {task_index}: No text regions found, skipping")
            self.stats['skipped_tasks'] += 1
            return []
        
        self.stats['total_regions'] += len(regions)
        
        # Crop each region
        cropped_data = []
        base_filename = Path(resolved_path).stem
        
        for region_index, region in enumerate(regions):
            # Crop the region
            cropped_image = self.crop_region(image, region['coordinates'])
            if cropped_image is None:
                logger.warning(f"Task {task_index}, Region {region_index}: Failed to crop region")
                self.stats['failed_crops'] += 1
                continue
            
            # Generate filename for cropped image
            cropped_filename = f"{base_filename}_region_{region_index:03d}.png"
            cropped_path = self.cropped_images_dir / cropped_filename
            
            # Save cropped image
            try:
                cropped_image.save(cropped_path, 'PNG')
                cropped_data.append((str(cropped_path), region['text']))
                self.stats['cropped_regions'] += 1
                logger.debug(f"Cropped region {region_index} to: {cropped_path}")
            except Exception as e:
                logger.error(f"Task {task_index}, Region {region_index}: Error saving cropped image: {e}")
                self.stats['failed_crops'] += 1
        
        return cropped_data
    
    def crop_all_regions(self) -> bool:
        """
        Crop all text regions from all tasks.
        
        Returns:
            True if processing was successful, False otherwise
        """
        logger.info(f"Loading Label Studio data from: {self.input_json_path}")
        tasks = self.load_label_studio_data()
        
        if not tasks:
            logger.error("No valid tasks found in the input file.")
            return False
        
        self.stats['total_tasks'] = len(tasks)
        
        # Process all tasks
        all_cropped_data = []
        
        for i, task in enumerate(tasks):
            logger.info(f"Processing task {i+1}/{len(tasks)}")
            cropped_data = self.process_task(task, i+1)
            all_cropped_data.extend(cropped_data)
            self.stats['processed_tasks'] += 1
        
        # Write annotation files
        self.write_annotation_files(all_cropped_data)
        
        # Print statistics
        self.print_statistics()
        
        return True
    
    def write_annotation_files(self, cropped_data: List[Tuple[str, str]]):
        """
        Write annotation data to VietOCR format files.
        
        Args:
            cropped_data: List of (image_path, text) tuples
        """
        if not cropped_data:
            logger.warning("No cropped data to write.")
            return
        
        # Create train/val split (80/20 by default)
        total_samples = len(cropped_data)
        train_size = int(total_samples * 0.8)
        
        train_data = cropped_data[:train_size]
        val_data = cropped_data[train_size:]
        
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
            for image_path, text in cropped_data:
                rel_path = os.path.relpath(image_path, self.output_dir)
                f.write(f"{rel_path}\t{text}\n")
        
        logger.info(f"Written {len(cropped_data)} total samples to: {combined_file}")
    
    def print_statistics(self):
        """Print processing statistics."""
        logger.info("=" * 50)
        logger.info("CROPPING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total tasks processed: {self.stats['total_tasks']}")
        logger.info(f"Successfully processed: {self.stats['processed_tasks']}")
        logger.info(f"Skipped tasks: {self.stats['skipped_tasks']}")
        logger.info(f"Total text regions found: {self.stats['total_regions']}")
        logger.info(f"Successfully cropped: {self.stats['cropped_regions']}")
        logger.info(f"Failed crops: {self.stats['failed_crops']}")
        
        if self.stats['total_tasks'] > 0:
            success_rate = (self.stats['processed_tasks'] / self.stats['total_tasks']) * 100
            logger.info(f"Task success rate: {success_rate:.1f}%")
        
        if self.stats['total_regions'] > 0:
            crop_success_rate = (self.stats['cropped_regions'] / self.stats['total_regions']) * 100
            logger.info(f"Crop success rate: {crop_success_rate:.1f}%")
        
        logger.info("=" * 50)


def main():
    """Main function to run the text region cropper."""
    parser = argparse.ArgumentParser(
        description="Crop text regions from Label Studio OCR annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic cropping
    python crop_text_regions.py export.json ./cropped_data/
    
    # With custom image directory
    python crop_text_regions.py export.json ./cropped_data/ --image-dir ./images/
    
    # Verbose output
    python crop_text_regions.py export.json ./cropped_data/ --verbose
        """
    )
    
    parser.add_argument('input_json', help='Path to Label Studio JSON export file')
    parser.add_argument('output_dir', help='Output directory for cropped images and annotations')
    parser.add_argument('--image-dir', help='Directory containing images (if different from JSON references)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input_json):
        logger.error(f"Input file does not exist: {args.input_json}")
        sys.exit(1)
    
    # Create cropper and run processing
    cropper = TextRegionCropper(
        input_json_path=args.input_json,
        output_dir=args.output_dir,
        image_dir=args.image_dir
    )
    
    success = cropper.crop_all_regions()
    
    if success:
        logger.info("Text region cropping completed successfully!")
        logger.info(f"Output files are in: {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Review the cropped images in the cropped_images/ directory")
        logger.info("2. Update VietOCR config file to point to the new annotation files")
        logger.info("3. Run VietOCR training with the cropped data")
    else:
        logger.error("Text region cropping failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
