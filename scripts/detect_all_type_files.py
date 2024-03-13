#!/usr/bin/env python
# coding: utf-8

from pdf2image import convert_from_path
from pdf2image import convert_from_bytes
import cv2
import numpy as np
from PIL import Image
import scripts.blank_field_detection as bfd
import json
from io import BytesIO
import imageio
from timeit import default_timer as timer
import concurrent.futures
import os
import pypdfium2 as pdfium

class Detection:
    def __init__(self) -> None:
        self.Page = 'page'
        self.Image = 'image'

    def get_detection(self, filename,filebyte):
        # print(filename)
        if filename.lower().endswith(".pdf"):
            res = self.get_detection_for_pdf(filename,filebyte)
        elif filename.lower().endswith(".png") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".jpg") or filename.lower().endswith(".bmp"):
            res = self.get_detection_for_image(filename,filebyte)
        elif filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
            res = self.get_detection_for_tif(filename,filebyte)
        elif filename.lower().endswith(".gif"):
            res = self.get_detection_for_gif(filename,filebyte)
        else:
            tmp_lst = filename.split('.')
            ftype = None if len(tmp_lst)<2 else tmp_lst[-1]
            raise Exception(f"<== Error: Unable to proccess for {ftype} type of files. ==>")
            # return f"<== Error: Unable to proccess for {ftype} type of files. ==>"
        # print(res)
        return res

    def process_page(self, page, page_number):
        page_annot = {'page_number': str(page_number)}
        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        detector = bfd.BlankFieldDetectorFilteredWithOcr(image= image)
        annots = detector.get_annotations()
        page_annot['field_annotations']=json.loads(annots)
        return (page_annot, detector.processed_image())
        
    def get_detection_for_pdf(self, filename = None, filebyte = None):
        # print(filename)
        # thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1000)
        # poppler_path = os.path.join('external_services','poppler')
        if(filebyte is None and filename is not None):
            # pages = convert_from_path(filename,poppler_path=poppler_path,dpi=500)
            pages = pdfium.PdfDocument(filename)
        else:
            # pages = convert_from_bytes(filebyte,poppler_path=poppler_path,dpi=500)
            pages = pdfium.PdfDocument(BytesIO(filebyte))
        final_annots = {'filename': os.path.basename(filename), 'pages': []}
        # print(pages)
        futures = []
        processedPages=[]
        for page_number,page in enumerate(pages):
            bitmap = page.render(
                scale = 5,    # 72dpi resolution
                rotation = 0, # no additional rotation
                # ... further rendering options
            )
            page = bitmap.to_pil()
            annot,processedPage = self.process_page(page=page,page_number=page_number)
            final_annots['pages'].append(annot)
            processedPages.append(processedPage)
        return (json.dumps(final_annots, indent = 4),processedPages)

    def get_detection_for_image(self, filename = None, filebyte = None):
        if(filebyte is None and filename is not None):
            image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.imdecode(filebyte, cv2.IMREAD_UNCHANGED)
        img_annots = {'filename': os.path.basename(filename), 'pages': []}
        detector = bfd.BlankFieldDetectorFilteredWithOcr(image= image)
        annots = detector.get_annotations()
        image_annot = {'page_number': str(1)}
        image_annot['field_annotations']=json.loads(annots)
        img_annots['pages'].append(image_annot)
        processed_images = []
        processed_images.append(detector.processed_image())
        return (json.dumps(img_annots, indent = 4), processed_images)

    def get_detection_for_tif(self, filename = None, filebyte = None):
        images = []
        if(filebyte is None and filename is not None):
            ret, images = cv2.imreadmulti(mats=images,
                                    filename=filename,
                                    start=0,
                                    count=2,
                                    flags=cv2.IMREAD_COLOR)
        else:
            ret, images = cv2.imdecodemulti(mats=images,
                                    buf=filebyte,
                                    flags=cv2.IMREAD_COLOR)
        final_annots = {'filename': os.path.basename(filename), 'pages': []}
        processed_images = []
        for i,image in enumerate(images):
            image_annot = {'page_number': str(i+1)}
            detector = bfd.BlankFieldDetectorFilteredWithOcr(image= image)
            annots = detector.get_annotations()
            image_annot['field_annotations']=json.loads(annots)
            final_annots['pages'].append(image_annot)
            processed_images.append(detector.processed_image())
        return (json.dumps(final_annots, indent = 4),processed_images)

    def get_detection_for_gif(self, filename = None, filebyte = None):
        if(filebyte is None and filename is not None):
            gif = imageio.mimread(filename)
        else:
            gif = imageio.mimread(BytesIO(filebyte))
        images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
        final_annots = {'filename': os.path.basename(filename), 'pages': []}
        processed_images = []
        for i,image in enumerate(images):
            image_annot = {'page_number': str(i+1)}
            detector = bfd.BlankFieldDetectorFilteredWithOcr(image= image)
            annots = detector.get_annotations()
            image_annot['field_annotations']=json.loads(annots)
            final_annots['pages'].append(image_annot)
            processed_images.append(detector.processed_image())
        return (json.dumps(final_annots, indent = 4),processed_images)

