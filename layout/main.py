from typing import Union

from fastapi import FastAPI
from fastapi import File, UploadFile
from src.detect_layout import LayoutParser_process
from PIL import Image
from io import BytesIO

conf = {'model_path': 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', \
        'extra_config':["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],\
        'label_map': {0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}}

app = FastAPI()
layp = LayoutParser_process(conf)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/images/{fold}/{im_id}")
def extract_layout(im_id: int, fold: Union[str, None] = None, q: Union[str, None] = None):
    im, headers = layp.detect_header(f"{fold}/im{im_id}.jpg")
    return {"folder": fold, "im_id": im_id, "header": headers}

@app.post('/getLayout')
def _file_upload(my_file: UploadFile = File(...)):
    im, headers = layp.detect_header_v2(my_file.file.read())
    return {"header": headers}