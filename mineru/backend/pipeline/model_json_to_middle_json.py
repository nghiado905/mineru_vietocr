# Copyright (c) Opendatalab. All rights reserved.
import os
import time

from loguru import logger
from tqdm import tqdm

from mineru.utils.config_reader import get_device, get_llm_aided_config, get_formula_enable
from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.backend.pipeline.para_split import para_split
from mineru.utils.block_pre_proc import prepare_block_bboxes, process_groups
from mineru.utils.block_sort import sort_blocks_by_bbox
from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType
from mineru.utils.llm_aided import llm_aided_title
from mineru.utils.model_utils import clean_memory
from mineru.backend.pipeline.pipeline_magic_model import MagicModel
from mineru.utils.ocr_utils import OcrConfidence
from mineru.utils.span_block_fix import fill_spans_in_blocks, fix_discarded_block, fix_block_spans
from mineru.utils.span_pre_proc import remove_outside_spans, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans, txt_spans_extract
from mineru.utils.table_merge import merge_table
from mineru.version import __version__
from mineru.utils.hash_utils import bytes_md5


def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=False, formula_enabled=True):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    page_w, page_h = map(int, page.get_size())
    magic_model = MagicModel(page_model_info, scale)

    """从magic_model对象中获取后面会用到的区块信息"""
    discarded_blocks = magic_model.get_discarded()
    text_blocks = magic_model.get_text_blocks()
    title_blocks = magic_model.get_title_blocks()
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations()

    img_groups = magic_model.get_imgs()
    table_groups = magic_model.get_tables()

    img_body_blocks, img_caption_blocks, img_footnote_blocks, maybe_text_image_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )

    table_body_blocks, table_caption_blocks, table_footnote_blocks, _ = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    spans = magic_model.get_all_spans()

    if len(maybe_text_image_blocks) > 0:
        for block in maybe_text_image_blocks:
            should_add_to_text_blocks = False

            if ocr_enable:
                span_in_block_list = [
                    span for span in spans
                    if span['type'] == 'text' and
                       calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block['bbox']) > 0.7
                ]

                if len(span_in_block_list) > 0:
                    spans_area = sum(
                        (span['bbox'][2] - span['bbox'][0]) * (span['bbox'][3] - span['bbox'][1])
                        for span in span_in_block_list
                    )
                    block_area = (block['bbox'][2] - block['bbox'][0]) * (block['bbox'][3] - block['bbox'][1])

                    if block_area > 0 and spans_area / block_area > 0.25:
                        should_add_to_text_blocks = True

            if should_add_to_text_blocks:
                block.pop('group_id', None) 
                text_blocks.append(block)
            else:
                img_body_blocks.append(block)


    if formula_enabled:
        interline_equation_blocks = []

    if len(interline_equation_blocks) > 0:

        for block in interline_equation_blocks:
            spans.append({
                "type": ContentType.INTERLINE_EQUATION,
                'score': block['score'],
                "bbox": block['bbox'],
                "content": "",
            })

        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks,
            page_w,
            page_h,
        )
    else:
        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equations,
            page_w,
            page_h,
        )

    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)

    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    if ocr_enable:
        pass
    else:
        spans = txt_spans_extract(page, spans, page_pil_img, scale, all_bboxes, all_discarded_blocks)

    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4
    )
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    if len(all_bboxes) == 0:
        return None

    for span in spans:
        if span['type'] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(
                span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale
            )

    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)

    fix_blocks = fix_block_spans(block_with_spans)

    sorted_blocks = sort_blocks_by_bbox(fix_blocks, page_w, page_h, footnote_blocks)

    page_info = make_page_info_dict(sorted_blocks, page_index, page_w, page_h, fix_discarded_blocks)

    return page_info


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False, formula_enabled=True):
    middle_json = {"pdf_info": [], "_backend":"pipeline", "_version_name": __version__}
    formula_enabled = get_formula_enable(formula_enabled)
    for page_index, page_model_info in tqdm(enumerate(model_list), total=len(model_list), desc="Processing pages"):
        page = pdf_doc[page_index]
        image_dict = images_list[page_index]
        page_info = page_model_info_to_page_info(
            page_model_info, image_dict, page, image_writer, page_index, ocr_enable=ocr_enable, formula_enabled=formula_enabled
        )
        if page_info is None:
            page_w, page_h = map(int, page.get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)

    """后置ocr处理"""
    need_ocr_list = []
    img_crop_list = []
    text_block_list = []
    for page_info in middle_json["pdf_info"]:
        for block in page_info['preproc_blocks']:
            if block['type'] in ['table', 'image']:
                for sub_block in block['blocks']:
                    if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']:
                        text_block_list.append(sub_block)
            elif block['type'] in ['text', 'title']:
                text_block_list.append(block)
        for block in page_info['discarded_blocks']:
            text_block_list.append(block)
    for block in text_block_list:
        for line in block['lines']:
            for span in line['spans']:
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    img_crop_list.append(span['np_img'])
                    span.pop('np_img')
    if len(img_crop_list) > 0:
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            det_db_box_thresh=0.3,
            lang=lang
        )
        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
        assert len(ocr_res_list) == len(
            need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
        for index, span in enumerate(need_ocr_list):
            ocr_text, ocr_score = ocr_res_list[index]
            if ocr_score > OcrConfidence.min_confidence:
                span['content'] = ocr_text
                span['score'] = float(f"{ocr_score:.3f}")
            else:
                span['content'] = ''
                span['score'] = 0.0

    para_split(middle_json["pdf_info"])

    merge_table(middle_json["pdf_info"])

    llm_aided_config = get_llm_aided_config()

    if llm_aided_config is not None:
        title_aided_config = llm_aided_config.get('title_aided', None)
        if title_aided_config is not None:
            if title_aided_config.get('enable', False):
                llm_aided_title_start_time = time.time()
                llm_aided_title(middle_json["pdf_info"], title_aided_config)
                logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    pdf_doc.close()
    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(model_list) >= 10:
        clean_memory(get_device())

    return middle_json

def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        'discarded_blocks': discarded_blocks,
    }
    return return_dict