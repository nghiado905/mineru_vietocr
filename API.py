import os
import json
from flask import Flask, request, jsonify, abort, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import io
import sys
from loguru import logger
import pypdfium2 as pdfium

# Thêm đường dẫn thư viện mineru
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox, draw_line_sort_bbox
from mineru.utils.enum_class import MakeMode
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import doc_analyze

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpeg", ".jpg", ".webp", ".gif"}

def convert_to_pdf_bytes(file_bytes, suffix):
    if suffix in {".png", ".jpeg", ".jpg", ".webp", ".gif"}:
        return images_bytes_to_pdf_bytes(file_bytes)
    elif suffix == ".pdf":
        return file_bytes
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def prepare_output_dirs(base_dir, filename, method="auto"):
    stem = Path(filename).stem
    output_dir = os.path.join(base_dir, stem, method)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return image_dir, output_dir

def process_pdf_document(pdf_bytes, filename):
    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    output_files = {}

    image_dir, output_dir = prepare_output_dirs(app.config['UPLOAD_FOLDER'], filename)
    image_writer = FileBasedDataWriter(image_dir)
    md_writer = FileBasedDataWriter(output_dir)

    infer_results, images_list, pdf_doc_obj, lang, ocr_enabled = doc_analyze(
        [pdf_bytes], ["ch"], parse_method="auto", formula_enable=True, table_enable=True
    )

    model_json = infer_results[0]
    middle_json = result_to_middle_json(
        model_json, images_list[0], pdf_doc_obj[0], image_writer, lang[0], ocr_enabled[0], True
    )
    pdf_info = middle_json["pdf_info"]

    stem = Path(filename).stem
    if draw_layout_bbox(pdf_info, pdf_bytes, output_dir, f"{stem}_layout.pdf"):
        output_files["layout_pdf"] = os.path.relpath(os.path.join(output_dir, f"{stem}_layout.pdf"), app.config['UPLOAD_FOLDER'])
    if draw_span_bbox(pdf_info, pdf_bytes, output_dir, f"{stem}_span.pdf"):
        output_files["span_pdf"] = os.path.relpath(os.path.join(output_dir, f"{stem}_span.pdf"), app.config['UPLOAD_FOLDER'])
    if draw_line_sort_bbox(pdf_info, pdf_bytes, output_dir, f"{stem}_line_sort.pdf"):
        output_files["line_sort_pdf"] = os.path.relpath(os.path.join(output_dir, f"{stem}_line_sort.pdf"), app.config['UPLOAD_FOLDER'])
    md_writer.write(os.path.join(output_dir, f"{stem}_origin.pdf"), pdf_bytes)
    output_files["original_pdf"] = os.path.relpath(os.path.join(output_dir, f"{stem}_origin.pdf"), app.config['UPLOAD_FOLDER'])

    image_base = os.path.basename(image_dir)
    markdown = union_make(pdf_info, MakeMode.MM_MD, image_base)
    content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, image_base)
    with open(os.path.join(output_dir, f"{stem}.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(os.path.join(output_dir, f"{stem}_content_list.json"), "w", encoding="utf-8") as f:
        json.dump(content_list, f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, f"{stem}_middle.json"), "w", encoding="utf-8") as f:
        json.dump(middle_json, f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, f"{stem}_model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, indent=4)

    try:
        output_files["markdown"] = json.loads(markdown) if markdown and markdown.strip() else {}
    except json.JSONDecodeError:
        output_files["markdown"] = {"error": "Markdown content is not valid JSON"}
    output_files["content_list"] = content_list
    output_files["middle_json"] = middle_json
    output_files["model_output"] = model_json

    pdf_doc.close()
    return output_files

@app.route('/process_pdf/', methods=['POST'])
def process_pdf():
    """Xử lý file PDF/image và trả về JSON."""
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            abort(400, "No file part")

        file = request.files['file']
        if not file or file.filename == '':
            logger.error("No file selected")
            abort(400, "No selected file")

        filename = secure_filename(file.filename)
        suffix = os.path.splitext(filename.lower())[1]
        if suffix not in SUPPORTED_SUFFIXES:
            logger.error(f"Unsupported file type: {suffix}")
            abort(400, "Unsupported file type. Use PDF or images.")

        file_bytes = file.read()
        pdf_bytes = convert_to_pdf_bytes(file_bytes, suffix)
        logger.info(f"Processing file: {filename}")

        output_files = process_pdf_document(pdf_bytes, filename)
        response = {
            "status": "success",
            "files": {Path(filename).stem: output_files},
            "message": "File processed successfully."
        }
        logger.info(f"Response: {response}")
        return jsonify(response)

    except ValueError as ve:
        logger.error(f"Value error: {str(ve)}")
        abort(400, str(ve))
    except Exception as e:
        logger.exception(f"Error processing file: {str(e)}")
        abort(500, str(e))

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Tải file từ thư mục output."""
    try:
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(full_path):
            logger.error(f"File not found: {full_path}")
            abort(404, "File not found")
        if not os.path.abspath(full_path).startswith(os.path.abspath(app.config['UPLOAD_FOLDER'])):
            logger.error(f"Access denied to: {full_path}")
            abort(403, "Access denied")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.exception(f"Download error: {str(e)}")
        abort(500, str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)