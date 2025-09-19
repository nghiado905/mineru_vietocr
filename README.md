# MinerU VietOCR Integration

A comprehensive document processing system that combines **MinerU** (a powerful PDF/document analysis framework) with **VietOCR** (Vietnamese OCR engine) to provide advanced document parsing, text extraction, and OCR capabilities with a modern web interface.

## 🚀 Features

### Core Functionality
- **PDF Document Processing**: Advanced PDF parsing with layout analysis, text extraction, and image processing
- **Multi-format Support**: Handles PDF, PNG, JPEG, JPG, WebP, and GIF files
- **Vietnamese OCR**: Specialized OCR engine optimized for Vietnamese text recognition
- **Layout Analysis**: Intelligent document structure detection including:
  - Text regions and paragraphs
  - Tables and formulas
  - Images and figures
  - Reading order detection
- **Web Interface**: Modern, responsive web UI with dark theme
- **API Endpoints**: RESTful API for programmatic access

### Advanced Features
- **Formula Recognition**: Mathematical formula detection and parsing
- **Table Extraction**: Structured table data extraction
- **Bounding Box Visualization**: Visual representation of detected regions
- **Multiple Output Formats**: Markdown, JSON, content lists, and annotated PDFs
- **Batch Processing**: Support for processing multiple documents
- **Label Studio Integration**: Tools for converting Label Studio annotations to VietOCR training format

## 📁 Project Structure

```
mineru_vietocr/
├── app.py                          # Main Flask web application
├── API.py                          # Alternative API implementation
├── test.py                         # API testing script
├── crop_text_regions.py            # Text region cropping utility
├── label_studio_to_vietocr_converter.py  # Label Studio format converter
├── templates/
│   └── index.html                  # Web interface template
├── mineru/                         # MinerU document processing framework
│   ├── backend/pipeline/           # Core processing pipeline
│   ├── model/                      # AI models (layout, OCR, table, formula)
│   ├── utils/                      # Utility functions
│   └── cli/                        # Command-line interface
├── vietocr/                        # Vietnamese OCR engine
│   ├── vietocr/                    # Core OCR implementation
│   │   ├── model/                  # OCR models (Transformer, CNN)
│   │   ├── tool/                   # Prediction and training tools
│   │   └── loader/                 # Data loading utilities
│   └── setup.py                    # Package configuration
├── output/                         # Processed document outputs
├── uploads/                        # Uploaded files storage
├── pdfs/                          # Sample PDF files
└── cropped_data/                  # Cropped text regions for training
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for better performance)
- 8GB+ RAM (16GB+ recommended)

### Dependencies

The project requires several key dependencies:

```bash
pip install -r requirements.txt
```

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nghiado905/mineru_vietocr
   cd mineru_vietocr
   ```

2. **Download required models** (models will be downloaded automatically on first use):
   - Layout detection models
   - OCR models
   - Formula recognition models
   - Table extraction models

## 🚀 Usage

### Web Interface

1. **Start the web server**:
   ```bash
   python app.py
   ```

2. **Access the interface**:
   Open your browser and navigate to `http://localhost:8000`

3. **Upload and process documents**:
   - Click "Chọn File" to select a PDF or image file
   - Click "Bắt đầu Xử lý" to start processing
   - View results in multiple formats (Markdown, JSON, etc.)

### API Usage

#### Process Document
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/process_pdf/
```

#### Download Processed Files
```bash
curl -O http://localhost:8000/download/path/to/processed/file.pdf
```

## 🔧 Configuration

### MinerU Configuration
The system automatically configures models based on available hardware:
- **Device**: Auto-detects CUDA, CPU, or MPS
- **Language**: Defaults to Vietnamese ('vi') with Chinese ('ch') support
- **Models**: Automatically downloads and caches required models

### VietOCR Configuration
Create a configuration file for VietOCR training:

```yaml
# config.yml
model:
  name: vgg_transformer
  backbone: vgg19_bn
  cnn_args:
    hidden_size: 256
  transformer_args:
    hidden_size: 256
    num_heads: 8
    num_layers: 6
    dropout: 0.1

dataset:
  train_annotation: cropped_data/annotation_train.txt
  val_annotation: cropped_data/annotation_val.txt
  image_height: 32
  image_min_width: 32
  image_max_width: 512

trainer:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.001
```

## 📊 Data Processing Pipeline

### 1. Document Analysis
- **Input**: PDF or image files
- **Layout Detection**: Identifies text blocks, tables, figures
- **Text Extraction**: Extracts text content with positioning
- **Formula Recognition**: Detects and parses mathematical formulas
- **Table Processing**: Extracts structured table data

### 2. OCR Processing
- **Text Recognition**: Uses VietOCR for Vietnamese text
- **Language Support**: Optimized for Vietnamese with multi-language capability
- **Post-processing**: Text cleaning and formatting

### 3. Output Generation
- **Markdown**: Clean, formatted text output
- **JSON**: Structured data with metadata
- **Annotated PDFs**: Visual representation of detected regions
- **Content Lists**: Organized content structure

## 🎯 Use Cases

### Document Digitization
- Convert scanned documents to searchable text
- Extract structured data from forms and reports
- Process historical documents and archives

### Content Management
- Automated document classification
- Text extraction for search indexing
- Content migration and conversion

### Research and Analysis
- Academic paper processing
- Data extraction from research documents
- Literature review automation

### Business Applications
- Invoice and receipt processing
- Contract analysis and extraction
- Report generation and analysis

## 🛠️ Development Tools

### Label Studio Integration
Convert Label Studio OCR annotations to VietOCR training format:

```bash
python label_studio_to_vietocr_converter.py export.json ./vietocr_data/
```

### Text Region Cropping
Extract individual text regions for training:

```bash
python crop_text_regions.py input.json output_dir --image-dir ./images/
```

### Testing
Test the API endpoints:

```bash
python test.py
```

## 📈 Performance

### Model Performance
- **Layout Detection**: High accuracy on complex document layouts
- **OCR Accuracy**: 88%+ precision on Vietnamese text
- **Processing Speed**: ~86ms per page on GPU (1080Ti)
- **Memory Usage**: Optimized for 8GB+ systems

### Optimization Tips
- Use GPU acceleration for faster processing
- Batch process multiple documents
- Adjust image resolution for speed vs. accuracy trade-offs
- Use appropriate model configurations for your use case

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project combines multiple open-source components:
- **MinerU**: Copyright (c) Opendatalab
- **VietOCR**: Apache 2.0 License
- **Integration Code**: See individual file headers

## 🆘 Support

### Common Issues

1. **Model Download Failures**: Ensure stable internet connection
2. **Memory Issues**: Reduce batch size or use CPU mode
3. **CUDA Errors**: Check GPU compatibility and drivers
4. **Vietnamese Text Issues**: Verify language configuration

### Getting Help
- Check the [MinerU documentation](https://github.com/opendatalab/MinerU)
- Review [VietOCR documentation](https://github.com/pbcquoc/vietocr)
- Open an issue for bugs or feature requests

## 🔄 Updates

### Recent Changes
- Integrated MinerU v2.2.2 with VietOCR v0.3.13
- Added web interface with modern UI
- Implemented Label Studio integration tools
- Enhanced Vietnamese language support
- Added batch processing capabilities

### Roadmap
- [ ] Multi-language OCR support
- [ ] Real-time processing capabilities
- [ ] Advanced table structure recognition
- [ ] Cloud deployment options
- [ ] Mobile app integration

---

**Note**: This system is designed for Vietnamese text processing but supports multiple languages. For best results with Vietnamese documents, ensure proper language configuration and use the provided Vietnamese OCR models.
