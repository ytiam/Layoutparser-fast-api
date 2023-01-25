import layoutparser as lp
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import pytesseract
import re
import numpy as np
import ast


conf = {'model_path': 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', \
        'extra_config':["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],\
        'label_map': {0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}}


def resize_image(path):
    img = Image.open(path)
    width, height = img.size

    new_height = height*4
    new_width  = int(new_height * width / height)

    img = img.resize((new_width,new_height), Image.ANTIALIAS)
    sub_fold_path = '/'.join(path.split('/')[:-1])+'/resized/'
    if not os.path.exists(sub_fold_path):
        os.mkdir(sub_fold_path)
    new_file_path = sub_fold_path+'packaging_info_resized.png'
    img.save(new_file_path)
    return new_file_path


class LayoutParser_process():
    
    def __init__(self, conf):
        model_path = conf['model_path']
        extra_config = conf['extra_config']
        label_map = conf['label_map']
        self.model = lp.Detectron2LayoutModel(model_path, 
                                 extra_config = extra_config,
                                 label_map = label_map)
    
    def detect_header(self, image_path, crop = False, crop_h_range = None, crop_w_range = None, object_block = 'Title'):
        image = cv2.imread(image_path)
        image = image[..., ::-1] 
        if crop:
            if crop_h_range != None:
                image = image[crop_h_range[0]:crop_h_range[1],:,:]
            else:
                if crop_w_range != None:
                    image = image[:,crop_w_range[0]:crop_w_range[1],:]
                else:
                    raise ValueError("Please provide a valid crop height/width or both as crop_h_range or crop_w_range")
            if crop_w_range != None:
                image = image[:,crop_w_range[0]:crop_w_range[1],:]
            else:
                if crop_h_range != None:
                    image = image[crop_h_range[0]:crop_h_range[1],:,:]
                else:
                    raise ValueError("Please provide a valid crop height/width or both as crop_h_range or crop_w_range")

            if ((crop_h_range != None)&(crop_w_range != None)):
                image = image[crop_h_range[0]:crop_h_range[1],crop_w_range[0]:crop_w_range[1],:]

        layout = self.model.detect(image)
        #text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
        title_blocks = lp.Layout([b for b in layout if b.type==object_block])

        return lp.draw_box(image, title_blocks, box_width=3), self.detect_text_from_image(image, title_blocks)
    
    def detect_header_v2(self, img, crop = False, crop_h_range = None, crop_w_range = None, object_block = 'Title'):
        nparr = np.fromstring(img, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = image[..., ::-1] 
        if crop:
            if crop_h_range != None:
                image = image[crop_h_range[0]:crop_h_range[1],:,:]
            else:
                if crop_w_range != None:
                    image = image[:,crop_w_range[0]:crop_w_range[1],:]
                else:
                    raise ValueError("Please provide a valid crop height/width or both as crop_h_range or crop_w_range")
            if crop_w_range != None:
                image = image[:,crop_w_range[0]:crop_w_range[1],:]
            else:
                if crop_h_range != None:
                    image = image[crop_h_range[0]:crop_h_range[1],:,:]
                else:
                    raise ValueError("Please provide a valid crop height/width or both as crop_h_range or crop_w_range")

            if ((crop_h_range != None)&(crop_w_range != None)):
                image = image[crop_h_range[0]:crop_h_range[1],crop_w_range[0]:crop_w_range[1],:]

        layout = self.model.detect(image)
        #text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
        title_blocks = lp.Layout([b for b in layout if b.type==object_block])

        return lp.draw_box(image, title_blocks, box_width=3), self.detect_text_from_image(image, title_blocks)

    def detect_text_from_image(self, image, blocks):
        ocr_agent = lp.TesseractAgent(languages='eng')
        for block in blocks:
            segment_image = (block
                               .pad(left=5, right=5, top=5, bottom=5)
                               .crop_image(image))

            text = ocr_agent.detect(segment_image)
            block.set(text=text, inplace=True)

        titles = []
        for txt in blocks.get_texts():
            print(txt, end='\n---\n')
            txt_ = txt.split("\n")[0]
            txt_ = re.sub(r"[^a-zA-Z0-9]+"," ",txt_)
            txt_ = txt_.strip()
            titles.append(txt_)
        return titles