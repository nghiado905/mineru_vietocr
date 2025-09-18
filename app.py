import os
from pathlib import Path
from flask import Flask, request, send_from_directory, jsonify, abort, render_template
from werkzeug.utils import secure_filename

import sys
import pypdfium2 as pdfium
from mineru.cli.common import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def serve_index():
    return render_template('index.html')

@app.route('/process_pdf/', methods=['POST'])
def process_pdf():
    try:
        if 'file' not in request.files:
            abort(400, description="No file part")

        file = request.files['file']
        if file.filename == '':
            abort(400, description="No selected file")

        filename = secure_filename(file.filename)
        if not filename.lower().endswith(tuple(pdf_suffixes + image_suffixes)):
            abort(400, description="Invalid file format. Only PDF or images are allowed.")

        file_bytes = file.read()
        file_suffix = f".{filename.rsplit('.', 1)[-1].lower()}"
        pdf_bytes = read_fn(file_bytes, file_suffix)
        pdf_file_name = Path(filename).stem

        output_dir = app.config['UPLOAD_FOLDER']

        output_files = process_pipeline(
            output_dir=output_dir,
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            parse_method="auto",
            p_formula_enable=True,
            p_table_enable=True,
            f_draw_layout_bbox=True,
            f_draw_span_bbox=True,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=True,
            f_dump_orig_pdf=True,
            f_dump_content_list=True,
            f_make_md_mode=MakeMode.MM_MD
        )

        response_data = {}
        for pdf_name, files in output_files.items():
            response_data[pdf_name] = {}
            for file_type, file_path in files.items():
                if file_type in ["markdown", "content_list", "middle_json", "model_output"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        response_data[pdf_name][file_type] = f.read()
                else:
                    rel_path = os.path.relpath(file_path, output_dir)
                    response_data[pdf_name][file_type] = rel_path

        return jsonify({
            "status": "success",
            "files": response_data,
            "message": "PDF processed successfully. Use the provided paths to download files.",
            "download_base": "/download/"
        })

    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

@app.route('/download/<path:rel_path>')
def download_file(rel_path):
    try:
        output_dir = app.config['UPLOAD_FOLDER']
        full_path = os.path.join(output_dir, rel_path)
        
        if not os.path.exists(full_path):
            abort(404, description="File not found")
        
        if not os.path.abspath(full_path).startswith(os.path.abspath(output_dir)):
            abort(403, description="Access denied")
        
        return send_from_directory(output_dir, rel_path)
    
    except Exception as e:
        logger.exception(e)
        abort(500, description=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)