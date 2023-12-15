FROM ubuntu
MAINTAINER amrkmrc@gmail.com

RUN apt-get update &&\
    apt-get install wget -y

ENV CONDA_DIR /opt/conda

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh &&\
    /bin/bash ~/anaconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init bash

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get install vim -y &&\
    apt-get install git -y &&\
    apt-get install gcc -y &&\
    apt-get install g++ -y &&\
    apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install layoutparser &&\
    pip install "layoutparser[layoutmodels]" &&\
    pip install "layoutparser[ocr]" &&\
    pip install layoutparser torchvision && pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2" &&\
    apt install tesseract-ocr -y &&\
    apt install libtesseract-dev -y &&\
    pip install fastapi &&\
    pip install "uvicorn[standard]" &&\
    pip install python-multipart &&\
    pip install easyocr

RUN git clone https://github.com/ytiam/Layoutparser-fast-api.git

WORKDIR "/Layoutparser-fast-api/layout/"

EXPOSE 8888 8889

CMD python -m uvicorn main:app --port 8889 --reload --host 0.0.0.0
