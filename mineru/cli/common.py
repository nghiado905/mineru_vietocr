# api.py
import io
import json
import os
import copy
from pathlib import Path
from flask import Flask, request, send_from_directory, jsonify, abort, render_template
from werkzeug.utils import secure_filename
import shutil
from loguru import logger
import sys
import pypdfium2 as pdfium

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox, draw_line_sort_bbox
from mineru.utils.enum_class import MakeMode
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze



pdf_suffixes = [".pdf"]
image_suffixes = [".png", ".jpeg", ".jpg", ".webp", ".gif"]

def read_fn(file_bytes, suffix):
    if suffix in image_suffixes:
        return images_bytes_to_pdf_bytes(file_bytes)
    elif suffix in pdf_suffixes:
        return file_bytes
    else:
        raise Exception(f"Unknown file suffix: {suffix}")

def prepare_env(output_dir, pdf_file_name, parse_method):
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir

def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None):
    pdf = pdfium.PdfDocument(pdf_bytes)
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(pdf) - 1
    if end_page_id > len(pdf) - 1:
        logger.warning("end_page_id is out of range, use pdf_docs length")
        end_page_id = len(pdf) - 1
    output_pdf = pdfium.PdfDocument.new()
    page_indices = list(range(start_page_id, end_page_id + 1))
    output_pdf.import_pages(pdf, page_indices)
    output_buffer = io.BytesIO()
    output_pdf.save(output_buffer)
    output_bytes = output_buffer.getvalue()
    pdf.close()
    output_pdf.close()
    return output_bytes

def prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id):
    result = []
    for pdf_bytes in pdf_bytes_list:
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        result.append(new_pdf_bytes)
    return result

def process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        is_pipeline=True
):
    f_draw_line_sort_bbox = True
    output_files = {}

    if f_draw_layout_bbox:
        layout_pdf_path = os.path.join(local_md_dir, f"{pdf_file_name}_layout.pdf")
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
        output_files["layout_pdf"] = layout_pdf_path

    if f_draw_span_bbox:
        span_pdf_path = os.path.join(local_md_dir, f"{pdf_file_name}_span.pdf")
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")
        output_files["span_pdf"] = span_pdf_path

    if f_dump_orig_pdf:
        orig_pdf_path = os.path.join(local_md_dir, f"{pdf_file_name}_origin.pdf")
        md_writer.write(f"{pdf_file_name}_origin.pdf", pdf_bytes)
        output_files["original_pdf"] = orig_pdf_path

    if f_draw_line_sort_bbox:
        line_sort_pdf_path = os.path.join(local_md_dir, f"{pdf_file_name}_line_sort.pdf")
        draw_line_sort_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_line_sort.pdf")
        output_files["line_sort_pdf"] = line_sort_pdf_path

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
        md_path = os.path.join(local_md_dir, f"{pdf_file_name}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content_str)
        output_files["markdown"] = md_path

    if f_dump_content_list:
        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        content_list_path = os.path.join(local_md_dir, f"{pdf_file_name}_content_list.json")
        with open(content_list_path, 'w', encoding='utf-8') as f:
            json.dump(content_list, f, ensure_ascii=False, indent=4)
        output_files["content_list"] = content_list_path

    if f_dump_middle_json:
        middle_json_path = os.path.join(local_md_dir, f"{pdf_file_name}_middle.json")
        with open(middle_json_path, 'w', encoding='utf-8') as f:
            json.dump(middle_json, f, ensure_ascii=False, indent=4)
        output_files["middle_json"] = middle_json_path

    if f_dump_model_output:
        if isinstance(model_output, list):
            output_text = ("\n" + "-" * 50 + "\n").join([json.dumps(m, ensure_ascii=False, indent=4) for m in model_output])
            model_output_path = os.path.join(local_md_dir, f"{pdf_file_name}_model_output.txt")
            with open(model_output_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
            output_files["model_output"] = model_output_path
        else:
            model_output_path = os.path.join(local_md_dir, f"{pdf_file_name}_model.json")
            with open(model_output_path, 'w', encoding='utf-8') as f:
                json.dump(model_output, f, ensure_ascii=False, indent=4)
            output_files["model_output"] = model_output_path

    logger.info(f"local output dir is {local_md_dir}")
    return output_files

def convert_to_pdf_bytes(file_bytes, suffix):
    if suffix in {".png", ".jpeg", ".jpg", ".webp", ".gif"}:
        return images_bytes_to_pdf_bytes(file_bytes)
    elif suffix == ".pdf":
        return file_bytes
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def process_pipeline(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
):
    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
        pipeline_doc_analyze(
            pdf_bytes_list, p_lang_list, parse_method=parse_method,
            formula_enable=p_formula_enable, table_enable=p_table_enable
        )
    )

    output_files = {}
    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_doc = all_pdf_docs[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]

        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer,
            _lang, _ocr_enable, p_formula_enable
        )

        pdf_info = middle_json["pdf_info"]
        pdf_bytes = pdf_bytes_list[idx]

        files = process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, model_json, is_pipeline=True
        )
        output_files[pdf_file_name] = files
    return output_files


# if __name__ == "__main__":
#     pdf_path = r"D:\LabAI\OCR\MinerUN\MinerU\pdfs\g.pdf"

#     try:
#        do_parse("./output", [Path(pdf_path).stem], [read_fn(Path(pdf_path))], ["ch"],backend='pipeline'
# )
#     except Exception as e:
#         logger.exception(e)


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

