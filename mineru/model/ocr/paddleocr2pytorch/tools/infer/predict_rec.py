import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import math
import time
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import math
import time
from loguru import logger

class TextRecognizer:
    def __init__(self, args, **kwargs):
        # Load VietOCR configuration
        self.device = args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        vietocr_config = Cfg.load_config_from_name('vgg_transformer')
        vietocr_config['device'] = self.device
        vietocr_config['cnn']['pretrained'] = True
        vietocr_config['predictor']['beamsearch'] = True
        self.text_recognizer = Predictor(vietocr_config)
        
        # Batch processing parameters
        self.rec_batch_num = args.rec_batch_num if hasattr(args, 'rec_batch_num') else 6
        self.rec_image_shape = [3, 32, 320]  # Default VietOCR input shape: [channels, height, width]
        self.limited_max_width = args.limited_max_width if hasattr(args, 'limited_max_width') else 1280
        self.limited_min_width = args.limited_min_width if hasattr(args, 'limited_min_width') else 16

    def resize_norm_img(self, img):
        """Resize and normalize image for VietOCR input."""
        imgC, imgH, imgW = self.rec_image_shape
        h, w = img.shape[:2]
        
        # Skip invalid images
        if h == 0 or w == 0:
            logger.warning(f"Invalid image dimensions: height={h}, width={w}")
            return None
        
        # Calculate aspect ratio and resize
        ratio = w / float(h)
        resized_w = int(imgH * ratio)
        resized_w = max(min(resized_w, self.limited_max_width), self.limited_min_width)
        
        # Resize image
        resized_image = cv2.resize(img, (resized_w, imgH), interpolation=cv2.INTER_CUBIC)
        resized_image = resized_image.astype('float32')
        
        # Convert to RGB and normalize
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image / 255.0  # VietOCR expects [0, 1] range
        
        # Pad image if necessary
        padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
        if resized_w < imgW:
            padding_im[:, :resized_w, :] = resized_image
        else:
            padding_im = resized_image[:, :imgW, :]
        
        return padding_im

    def __call__(self, img_list, tqdm_enable=False, tqdm_desc="OCR-rec Predict"):
        """Process a list of images with VietOCR."""
        img_num = len(img_list)
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        elapse = 0

        # Sort images by aspect ratio to optimize batch processing
        width_list = [img.shape[1] / float(img.shape[0]) if img.shape[0] > 0 else 0 for img in img_list]
        indices = np.argsort(np.array(width_list))

        with tqdm(total=img_num, desc=tqdm_desc, disable=not tqdm_enable) as pbar:
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)
                norm_img_batch = []
                
                # Preprocess images for the batch
                for ino in range(beg_img_no, end_img_no):
                    norm_img = self.resize_norm_img(img_list[indices[ino]])
                    if norm_img is None:
                        norm_img_batch.append(np.zeros((32, 320, 3), dtype=np.float32))
                        continue
                    norm_img_batch.append(norm_img)
                
                norm_img_batch = np.array(norm_img_batch)  # Shape: [batch, H, W, C]
                
                starttime = time.time()
                # Convert to PIL Images for VietOCR
                pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in norm_img_batch]
                
                # Predict with VietOCR
                batch_results = []
                for pil_img in pil_images:
                    try:
                        text, prob = self.text_recognizer.predict(pil_img, return_prob=True)
                        if prob is None:
                            logger.success(f"VietOCR: {text}")
                            prob = 0.9
                    except Exception as e:
                        logger.error(f"VietOCR prediction failed: {str(e)}")
                        text, prob = '', 0.9
                    batch_results.append([text, prob])
                
                # Assign results back to original indices
                for rno, result in enumerate(batch_results):
                    rec_res[indices[beg_img_no + rno]] = result
                
                elapse += time.time() - starttime
                pbar.update(end_img_no - beg_img_no)

        # Fix NaN values in recognition results
        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if isinstance(score, float) and math.isnan(score):
                rec_res[i] = (text, 0.8)
            elif score is None:
                logger.warning(f"Score is None at index {i}, setting to 0.0")
                rec_res[i] = (text, 0.8)

        return rec_res, elapse