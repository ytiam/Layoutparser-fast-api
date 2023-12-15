import layoutparser as lp
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import pytesseract
import re
import numpy as np
import ast
import easyocr
reader = easyocr.Reader(['en'])


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
        self.read_func_flag = False
        
    def check_if_file_read(func):
        def wrapper(self, *arg, **kw):
            if not self.read_func_flag:
                raise ValueError("You First Need to run the read_process_image function. Please run it before "\
                "proceed ahead")
            res = func(self, *arg, **kw)
            return res
        return wrapper
        
        
    def read_process_image(self, img = None, img_path = None, crop = False, crop_h_range = None, crop_w_range = None):
        self.read_func_flag = True
        
        if img is not None:
            self.img = img
            nparr = np.fromstring(self.img, np.uint8)
            self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            try:
                assert img_path != None
                self.image = cv2.imread(img_path)
            except AssertionError:
                raise Exception("Please provide a valid image file path")
        
        self.image = self.image[..., ::-1]
        
        if crop:
            if crop_h_range != None:
                self.image = self.image[crop_h_range[0]:crop_h_range[1],:,:]
            else:
                if crop_w_range != None:
                    self.image = self.image[:,crop_w_range[0]:crop_w_range[1],:]
                else:
                    raise ValueError("Please provide a valid crop height/width or both as crop_h_range or crop_w_range")
            if crop_w_range != None:
                self.image = self.image[:,crop_w_range[0]:crop_w_range[1],:]
            else:
                if crop_h_range != None:
                    self.image = self.image[crop_h_range[0]:crop_h_range[1],:,:]
                else:
                    raise ValueError("Please provide a valid crop height/width or both as crop_h_range or crop_w_range")

            if ((crop_h_range != None)&(crop_w_range != None)):
                self.image = self.image[crop_h_range[0]:crop_h_range[1],crop_w_range[0]:crop_w_range[1],:]
        self.layout = self.model.detect(self.image)
        
    @check_if_file_read  
    def detect_header_v2(self, object_block = 'Title'):
        title_blocks = lp.Layout([b for b in self.layout if b.type==object_block])
        return lp.draw_box(self.image, title_blocks, box_width=3), self.detect_text_from_image(self.image, title_blocks)
    
    def sort_bounding_boxes_as_layout(self, blocks:list):
        h, w = self.image.shape[:2]

        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

        left_blocks = blocks.filter_by(left_interval, center=True)
        left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
        # The b.coordinates[1] corresponds to the y coordinate of the region
        # sort based on that can simulate the top-to-bottom reading order 
        right_blocks = lp.Layout([b for b in blocks if b not in left_blocks])
        right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

        # And finally combine the two lists and add the index
        blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
        return blocks
    
    @check_if_file_read
    def detect_image_title_pair(self):
        text_blocks = lp.Layout([b for b in self.layout if b.type=='Text'])
        figure_blocks = lp.Layout([b for b in self.layout if b.type=='Figure'])
        
        text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

        text_figure_blocks = lp.Layout([b for b in text_blocks] + [c for c in figure_blocks])
        text_figure_blocks = self.sort_bounding_boxes_as_layout(text_figure_blocks)
        figures = [i for i in text_figure_blocks if i.type=='Figure']
        
        fig_title_id = {}
        for fig in figures:
            fig_id = fig.id
            title_id = fig_id+1
            fig_title_id[fig_id] = title_id
            
        temp_segments = {}
        for fig_id, title_id in fig_title_id.items():
            block = [i for i in text_figure_blocks if i.id == title_id][0]
            segment_image = (block
                               .pad(left=5, right=5, top=2, bottom=2)
                               .crop_image(self.image,))
            temp_text = ' '.join(reader.readtext(segment_image,detail = 0))
            if temp_text.startswith('Figure'):
                temp_segments[fig_id] = temp_text
            else:
                try:
                    fig_block = [i for i in text_figure_blocks if i.id == fig_id][0]
                    fig = (fig_block
                                           .pad(left=5, right=5, top=2, bottom=2)
                                           .crop_image(self.image,))
                    title_ = 'Figure'+' '.join(reader.readtext(fig,detail = 0)).split('Figure')[-1]
                    temp_segments[fig_id] = title_
                except:
                    pass
        return temp_segments

    def detect_text_from_image(self, image, blocks):
        ocr_agent = lp.TesseractAgent(languages='eng')
        for block in blocks:
            segment_image = (block
                               .pad(left=5, right=5, top=5, bottom=5)
                               .crop_image(image))

            text = ' '.join(reader.readtext(segment_image,detail = 0)) #ocr_agent.detect(segment_image)
            block.set(text=text, inplace=True)
        titles = []
        for txt in blocks.get_texts():
            print(txt, end='\n---\n')
            txt_ = txt.split("\n")[0]
            txt_ = re.sub(r"[^a-zA-Z0-9]+"," ",txt_)
            txt_ = txt_.strip()
            titles.append(txt_)
        return titles