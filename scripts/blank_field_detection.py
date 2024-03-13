#!/usr/bin/env python
# coding: utf-8


# pip install layoutparser
# pip install layoutparser[ocr]
# #install tesseract on system and copy tesseract executable path


# tesseract_path = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


import cv2
import numpy as np
import json
from PIL import Image
from bs4 import BeautifulSoup
import subprocess
import tempfile
import os
import math
Image.MAX_IMAGE_PIXELS = 933120000


class Utils():
    ''' 
    <== Provide coordinates in format(x, y, width, height) ==>
    
    '''
    def __init__(self):
        print(f"<== Provide coordinates in format(x, y, width, height) ==>")
        
    def normalize(self,coords,width,height):
        '''
        The function to Normalize coordinates.
 
        Parameters:
            coords (list or tuple): coordinates in format(x, y, width, height).
            width (int): width of original Image.
            height (int): height of original Image.
 
        Returns:
            List: Normalized coordinates in format(x, y, width, height).
        '''
        return (coords[0]/width,coords[1]/height,coords[2]/width,coords[3]/height)
    
    def denormalize(self,coords,width,height):
        '''
        The function to Normalize coordinates.
 
        Parameters:
            coords (list or tuple): coordinates in format(x, y, width, height).
            width (int): width of Image for which we want to denormalize fields.
            height (int): height of Image for which we want to denormalize fields.
 
        Returns:
            List: Denormalized coordinates in format(x, y, width, height).
        '''
        return (int(round(coords[0]*width)),int(round(coords[1]*height)),int(round(coords[2]*width)),int(round(coords[3]*height)))

    
    @staticmethod
    def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
    
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image
    
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
    
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
    
        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)
    
        # return the resized image
        return resized

class BoundingBox:
    def __init__(self, x, y, w, h, type = "BOX"):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.Type = type

    def get_type(self):
        return self.Type
        
    def is_checkbox(self):
        return ((self.w==self.h) or abs(self.w-self.h)<20) or (self.w<30 and self.h<30)
    
    def get_coordinates(self):
        return (self.x, self.y, self.w, self.h)

    def normalize_coords(self,width,height):
        return BoundingBox((self.x*100)/width,(self.y*100)/height,(self.w*100)/width,(self.h*100)/height)

    def denormalize_coords(self,width,height):
        return BoundingBox((self.x*width)/100,(self.y*height)/100,(self.w*width)/100,(self.h*height)/100)

    def draw(self, image, color = (0, 255, 0)):
        cv2.rectangle(image, (int(round(self.x)), int(round(self.y))), (int(round(self.x + self.w)), int(round(self.y + self.h))), color, -1)

    def mask(self, image):
        cv2.rectangle(image, (int(round(self.x)), int(round(self.y))), (int(round(self.x + self.w)), int(round(self.y + self.h))), (0, 0, 0), -1)

    def is_valid(self, width, height, ocr = None):
        valid = self.x>10 and self.y>10 and (self.x+self.w)<(width-10) and (self.y+self.h)<(height-10) and self.w<(width//2) and self.h<(height//2) and self.w>15 and self.h>15
##        if valid and not self.is_checkbox():
##            filter_box = []
##            if self.y<(2*self.h):
##                filter_box.append(BoundingBox(self.x,0,self.w,(2*self.h)))
##            else:
##                filter_box.append(BoundingBox(self.x,self.y-(2*self.h),self.w,(2*self.h)))
##            w = self.w//2
##            if self.x<self.w//2:
##                filter_box.append(BoundingBox(0,self.y-5,self.w//2,self.h-5))
##            else:
##                filter_box.append(BoundingBox(self.x-self.w//2,self.y-5,self.w//2,self.h-5))
##            if (self.x+self.w+self.w)>width:
##                filter_box.append(BoundingBox(self.x+self.w,self.y-5,width-self.w,self.h-5))
##            else:
##                filter_box.append(BoundingBox(self.x+self.w,self.y-5,self.w//2,self.h-5))
##            filtered_box = ocr.filtered_fields(filter_box)
##            if len(filtered_box) == 3:
##                valid = False
        return valid


class BlankFieldDetectorFilteredWithOcr:
  def __init__(self, filepath = None, image = None):
      if(image is None and filepath is not None):
            self.Image = cv2.imread(filepath, cv2.IMREAD_COLOR)
      else:
            self.Image = image.copy()
      # self.filepath = filepath
      # self.Image = cv2.imread(filepath)
      #print(self.Image.shape)
      h, w=self.Image.shape[:2]
      self.shape = (w,h)
      self.rect_box_detector = RectBoxDetector(image = self.Image)
      self.line_box_detector = LineBoxDetector(image = self.Image,rect_box_detector = self.rect_box_detector)
      self.circle_box_detector = CircleBoxDetector(image = self.Image,rect_box_detector = self.rect_box_detector, line_box_detector = self.line_box_detector)
      self.ocr = OCRModel(image = self.Image)
      self.CheckBox_fields = self.rect_box_detector.detect_checkboxes()
      self.TextBox_fields = self.line_box_detector.Boxes + self.rect_box_detector.detect_textboxes()
      self.RadioButton_fields = self.circle_box_detector.Circles
      self.Colon_fields = self.ocr.get_colon_fields(self.CheckBox_fields+self.TextBox_fields+self.RadioButton_fields)
      self.TextBox_fields += self.Colon_fields
##      self.TextBox_fields += self.ocr.get_dollar_fields(self.CheckBox_fields+self.TextBox_fields+self.RadioButton_fields)
      self.TextBox_fields = self.ocr.filtered_fields(self.TextBox_fields)
      self.CheckBox_fields = self.ocr.filtered_fields(self.CheckBox_fields)
      self.RadioButton_fields = self.ocr.filtered_fields(self.RadioButton_fields)
      self.TextBox_fields = [box for box in  self.TextBox_fields if box.is_valid(self.shape[0],self.shape[1])]
      self.CheckBox_fields = [box for box in self.CheckBox_fields if box.is_valid(self.shape[0],self.shape[1])]
      self.RadioButton_fields = [box for box in self.RadioButton_fields if box.is_valid(self.shape[0],self.shape[1])]

  def detect(self):
      checkBox_fields = self.rect_box_detector.detect_checkboxes()
      textBox_fields = self.line_box_detector.Boxes + self.rect_box_detector.detect_textboxes()
      radiobutton_fields = self.circle_box_detector.Circles
      colon_fields = self.ocr.get_colon_fields(checkBox_fields+textBox_fields+radiobutton_fields)
      textBox_fields += colon_fields
##      textBox_fields += self.ocr.get_dollar_fields(checkBox_fields+textBox_fields+radiobutton_fields)
      #radiobutton_fields = self.ocr.filtered_fields(radiobutton_fields)
      filtered_fields  = self.ocr.filtered_fields( checkBox_fields+textBox_fields+radiobutton_fields)
      # filtered_fields  = checkBox_fields+textBox_fields+radiobutton_fields
      filtered_fields = [box for box in filtered_fields if box.is_valid(self.shape[0],self.shape[1])]
      
      return filtered_fields#checkBox_fields+textBox_fields+radiobutton_fields#

  def get_annotations(self):
    dict_fields = []
    checkBox_fields = self.CheckBox_fields.copy()
    textBox_fields = self.TextBox_fields.copy()
    radiobutton_fields = self.RadioButton_fields.copy()
    checkBox_fields = self.get_normalized(checkBox_fields)
    for box in checkBox_fields:
      box=box.get_coordinates()
      fd={'X':str(box[0]),'Y':str(box[1]),'Width':str(box[2]),'Height':str(box[3]),'type':'CheckBox'}
      dict_fields.append(fd.copy())
    # line_fields = self.ocr.filtered_fields(line_fields)
    textBox_fields = self.get_normalized(textBox_fields)
    for box in textBox_fields:
      box=box.get_coordinates()
      fd={'X':str(box[0]),'Y':str(box[1]),'Width':str(box[2]),'Height':str(box[3]),'type':'TextBox'}
      dict_fields.append(fd.copy())
    # circle_fields = self.ocr.filtered_fields(circle_fields)
    radiobutton_fields = self.get_normalized(radiobutton_fields)
    for box in radiobutton_fields:
      box=box.get_coordinates()
      fd={'X':str(box[0]),'Y':str(box[1]),'Width':str(box[2]),'Height':str(box[3]),'type':'RadioButton'}
      dict_fields.append(fd.copy())
    annots = json.dumps(dict_fields, indent = 4)
    return annots

  def get_normalized(self, fields = None):
      if fields is None:
          fields = self.detect()
      normalized_fields=[]
      for i in fields:
          normalized_fields.append(i.normalize_coords(*self.shape))
      return normalized_fields

  def get_denormalized(self, normalized_fields):
      denormalized_fields=[]
      for i in normalized_fields:
          denormalized_fields.append(i.denormalize_coords(*self.shape))
      return denormalized_fields

  def get_fields_from_json(self, json_string):
    adict = json.loads(json_string)
    fields = []
    for i in adict:
        fields.append(BoundingBox(float(i['X']),float(i['Y']),float(i['Width']),float(i['Height']),i['type']).denormalize_coords(*self.shape))
    return fields.copy()

  def processed_image(self):
      image = self.Image.copy()
      for box in self.CheckBox_fields:
        box.draw(image,(255,0,0))
      for box in self.TextBox_fields:
        box.draw(image,(0,255,0))
      for box in self.RadioButton_fields:
        box.draw(image,(0,0,255))
      image = Image.fromarray(image)
      return image

class OCRModel:
    def __init__(self,filepath = None, image = None):
        # self.model = lp_ocr.TesseractAgent().with_tesseract_executable(tesseract_path)
        if(image is None and filepath is not None):
            self.Image = cv2.imread(filepath)
        else:
            self.Image = image.copy()
        # self.filepath = filepath
        self.TesseractPath = os.path.join('field_extractor','external_services','Tesseract-OCR','tesseract.exe')
        self.ocr_result = self.detect()

    def detect(self):
        # ocr_response = self.model.detect(self.Image.copy(), return_response=True)
        # ocr  = self.model.gather_data(ocr_response, lp.TesseractFeatureType(4))
        return self.detect_tesseract()

    def preprocess(self):
      img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
      # kernal = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
      # img_sharp = cv2.filter2D(src=img,ddepth=-1,kernel=kernal)
      # return img_sharp
      blur = cv2.GaussianBlur(img,(5,5),0)
      
      # # find otsu's threshold value with OpenCV function
      ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      kernel = np.ones((3,3), np.uint8)
      img_erosion = cv2.erode(otsu, kernel, iterations=1)
      kernal = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
      img_sharp = cv2.filter2D(src=img_erosion,ddepth=-1,kernel=kernal)
      blur_f = cv2.GaussianBlur(img_sharp,(3,3),0)
      return blur_f
      # return otsu.copy()

    def detect_tesseract(self):
      def extract_data(xml_data):
          # Parse the XML data using Beautiful Soup and lxml parser
          soup = BeautifulSoup(xml_data, features="html.parser")
          
          # Find all spans with class 'ocrx_word' and extract their text and bounding box coordinates
          ocr_words = soup.find_all('span', class_='ocrx_word')
          extracted_data = []
          
          for word in ocr_words:
              word_text = word.get_text()
              bbox = word['title'].split(';')[0].split('bbox ')[1]
              extracted_data.append({'text': word_text, 'bbox': list(map(int,bbox.split()))})
          return extracted_data
      
      with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
            # Save the image data to the temporary file
            cv2.imwrite(temp_image.name, self.preprocess())#self.preprocess()
      # OCR Engine Mode (OEM) and Page Segmentation Mode (PSM) options
      oem = 3  # OEM_TESSERACT_ONLY
      psm = 3  # PSM_AUTO
      # Construct the Tesseract command with custom options and hocr output format
      tesseract_command = [
          self.TesseractPath,
          temp_image.name,
          "stdout",  # Output to stdout
          "--oem", str(oem),  # Set OCR Engine Mode as separate arguments
          "--psm", str(psm),   # Set Page Segmentation Mode as separate arguments
          "hocr"  # Output in hocr format
      ]

      try:
          # Run Tesseract using subprocess and capture the hocr output
          hocr_output = subprocess.check_output(tesseract_command, stderr=subprocess.PIPE)
          hocr_text = hocr_output.decode("utf-8").strip()
          # f=open("myfile.txt","w")
          # f.write(hocr_text)
          # f.close()
          extracted_data = extract_data(hocr_text)
          return extracted_data
      except subprocess.CalledProcessError as e:
          # Print the error output if Tesseract encounters an error
          print("Error occurred during OCR:")
          print(e.stderr.decode("utf-8").strip())

    def extract_fields(self):
        fields = {'txt_fields':[],'non_txt_fields':[],'colon_txt_fields':[],'word_fields':[],'money_txt_fields':[]}
        for txt_blck in self.ocr_result:
          box = txt_blck['bbox']
          # fields['txt_fields'].append([*box, txt_blck['text']])
          if(txt_blck['text'].strip()=='$'):
              fields['money_txt_fields'].append([*box, txt_blck['text']])
          if(txt_blck['text'].strip()!='' and txt_blck['text'] != '' and txt_blck is not None and len(txt_blck['text'])>1):
            # if ("o" in txt_blck['text'].lower() and txt_blck['text'].lower() not in ["of","on","so","to","do"]):
            #    continue
            fields['txt_fields'].append([*box, txt_blck['text']])
            if(txt_blck['text'].endswith(':')):
              fields['colon_txt_fields'].append([*box,  txt_blck['text']])
            if(len(txt_blck['text'].split(' ')) == 1 and txt_blck['text'].isalnum() and len(txt_blck['text'])>1):
              fields['word_fields'].append([*box, txt_blck['text']])
          else:
            fields['non_txt_fields'].append([*box, txt_blck['text']])
        return fields.copy()

    # def extract_fields(self):
    #     fields = {'txt_fields':[],'non_txt_fields':[],'colon_txt_fields':[],'word_fields':[],'money_txt_fields':[]}
    #     for txt_blck in self.ocr_result:
    #       if(txt_blck.text.strip()=='$'):
    #           fields['money_txt_fields'].append([txt_blck.block.x_1,txt_blck.block.y_1,txt_blck.block.x_2,txt_blck.block.y_2, txt_blck.text])
    #       if(txt_blck.text.strip()!='' and txt_blck.text != '' and txt_blck is not None and not len(txt_blck.text)<=2):
    #         fields['txt_fields'].append([txt_blck.block.x_1,txt_blck.block.y_1,txt_blck.block.x_2,txt_blck.block.y_2, txt_blck.text])
    #         if(txt_blck.text.find(':')!=-1):
    #           fields['colon_txt_fields'].append([txt_blck.block.x_1,txt_blck.block.y_1,txt_blck.block.x_2,txt_blck.block.y_2,  txt_blck.text])
    #         if(len(txt_blck.text.split(' ')) == 1 and txt_blck.text.isalnum() and not len(txt_blck.text)<=2):
    #           fields['word_fields'].append([txt_blck.block.x_1,txt_blck.block.y_1,txt_blck.block.x_2,txt_blck.block.y_2, txt_blck.text])
    #       else:
    #         fields['non_txt_fields'].append([txt_blck.block.x_1,txt_blck.block.y_1,txt_blck.block.x_2,txt_blck.block.y_2, txt_blck.text])
    #     return fields.copy()

    def get_colon_fields(self, other_fields = None, ):
        img = self.Image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2_imshow(gray)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
        # cv2_imshow(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        if other_fields is not None:
            for field in other_fields:
                i = field.get_coordinates()
                cv2.rectangle(opening,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(255,255,255),-1)
        fields = self.extract_fields()
        colon_fields = fields['colon_txt_fields']
        filtered_fields=[]
        for i in colon_fields:
            roi = opening[i[1]-2:i[3]-5,i[2]+20:i[2]+(i[2]-i[0])*2]
            if(np.all(roi<10)):
              filtered_fields.append(BoundingBox(i[2]+20,i[1]-2,(i[2]-i[0])*2,i[3]-i[1]-4))
            else:
              roi = opening[i[1]-5:i[3]-10,i[2]+20:i[2]+(i[2]-i[0])*2]
              if(np.all(roi<10)):
                filtered_fields.append(BoundingBox(i[2]+20,i[1]-2,(i[2]-i[0])*2,i[3]-i[1]-4))
        return filtered_fields.copy()

    def get_dollar_fields(self, other_fields = None, ):
        img = self.Image.copy()
        if other_fields is not None:
            for field in other_fields:
                i = field.get_coordinates()
                cv2.rectangle(img,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,0,0),-1)
        fields = self.extract_fields()
        doller_fields = fields['money_txt_fields']
        filtered_fields=[]
        for i in doller_fields:
            roi = img[i[1]-2:i[3]-2,i[2]+20:i[2]+(i[2]-i[0])*4]
            if(np.all(roi>250)):
              filtered_fields.append(BoundingBox(i[2]+20,i[1]-2,(i[2]-i[0])*4,i[3]-i[1]-4))
            else:
              roi = img[i[1]-5:i[3]-5,i[2]+20:i[2]+(i[2]-i[0])*4]
              if(np.all(roi>250)):
                filtered_fields.append(BoundingBox(i[2]+20,i[1]-2,(i[2]-i[0])*4,i[3]-i[1]-4))
        return filtered_fields.copy()

    def filtered_fields(self, fields = None):
        def is_not_intersecting(b1,b2):
            return b1[2]<b2[0] or b1[0]>b2[2] or b1[3]<b2[1] or b1[1]>b2[3]
        ocr_fields = self.extract_fields()
        word_fields = ocr_fields['txt_fields']
        final_filter_fields=[]#ff+new_fields
        # for j in word_fields:
        #    final_filter_fields.append(BoundingBox(j[0],j[1],j[2]-j[0],j[3]-j[1]))
        for box in fields:
            i = list(box.get_coordinates())
            i[2] = i[0]+i[2]#+5
            i[3] = i[1]+i[3]#+5
            # i[0]-=5
            # i[1]-=5
            intersect = False
            for j in word_fields:
                intersect = not is_not_intersecting(i,j)
                if(intersect):
                    break
            if not intersect:
                final_filter_fields.append(box)
        return final_filter_fields.copy()

class BlankFieldDetector:
  def __init__(self, filepath = None, image = None):
      if(image is None and filepath is not None):
        self.Image = cv2.imread(filepath)
      else:
        self.Image = image.copy()
      
      h, w, c=self.Image.shape
      self.shape = (w,h)
      self.rect_box_detector = RectBoxDetector(image = self.Image)
      self.line_box_detector = LineBoxDetector(image = self.Image,rect_box_detector = self.rect_box_detector)
      self.circle_box_detector = CircleBoxDetector(image = self.Image,rect_box_detector = self.rect_box_detector, line_box_detector = self.line_box_detector)
      self.CheckBox_fields = self.rect_box_detector.detect_checkboxes()
      self.TextBox_fields = self.line_box_detector.Boxes + self.rect_box_detector.detect_textboxes()
      self.RadioButton_fields = self.circle_box_detector.Circles

  def detect(self):
      checkBox_fields = self.rect_box_detector.detect_checkboxes()
      textBox_fields = self.line_box_detector.Boxes + self.rect_box_detector.detect_textboxes()
      radiobutton_fields = self.circle_box_detector.Circles
      # filtered_fields  = self.ocr.filtered_fields(fields)
      return checkBox_fields+textBox_fields+radiobutton_fields #+filtered_fields#

  def get_annotations(self):
    dict_fields = []
    checkBox_fields = self.CheckBox_fields.copy()
    textBox_fields = self.TextBox_fields.copy()
    radiobutton_fields = self.RadioButton_fields.copy()
    checkBox_fields = self.get_normalized(checkBox_fields)
    for box in checkBox_fields:
      box=box.get_coordinates()
      fd={'X':str(box[0]),'Y':str(box[1]),'Width':str(box[2]),'Height':str(box[3]),'type':'CheckBox'}
      dict_fields.append(fd.copy())
    # line_fields = self.ocr.filtered_fields(line_fields)
    textBox_fields = self.get_normalized(textBox_fields)
    for box in textBox_fields:
      box=box.get_coordinates()
      fd={'X':str(box[0]),'Y':str(box[1]),'Width':str(box[2]),'Height':str(box[3]),'type':'TextBox'}
      dict_fields.append(fd.copy())
    # circle_fields = self.ocr.filtered_fields(circle_fields)
    radiobutton_fields = self.get_normalized(radiobutton_fields)
    for box in radiobutton_fields:
      box=box.get_coordinates()
      fd={'X':str(box[0]),'Y':str(box[1]),'Width':str(box[2]),'Height':str(box[3]),'type':'RadioButton'}
      dict_fields.append(fd.copy())
    annots = json.dumps(dict_fields, indent = 4)
    return annots

  def get_normalized(self, fields = None):
      if fields is None:
          fields = self.detect()
      normalized_fields=[]
      for i in fields:
          normalized_fields.append(i.normalize_coords(*self.shape))
      return normalized_fields

  def get_denormalized(self, normalized_fields):
      denormalized_fields=[]
      for i in normalized_fields:
          denormalized_fields.append(i.denormalize_coords(*self.shape))
      return denormalized_fields

  def get_fields_from_json(self, json_string):
    adict = json.loads(json_string)
    fields = []
    for i in adict:
        fields.append(BoundingBox(float(i['X']),float(i['Y']),float(i['Width']),float(i['Height']),i['type']).denormalize_coords(*self.shape))
    return fields.copy()
    
  def processed_image(self):
      image = self.Image.copy()
      for box in self.CheckBox_fields:
        box.draw(image,(255,0,0))
      for box in self.TextBox_fields:
        box.draw(image,(0,255,0))
      for box in self.RadioButton_fields:
        box.draw(image,(0,0,255))
      image = Image.fromarray(image)
      return image


class CircleBoxDetector:
    def __init__(self, filepath = None, image = None, rect_box_detector = None, line_box_detector = None):
      if(image is None):
        self.Image = cv2.imread(filepath)
      else:
        self.Image = image.copy()

      if rect_box_detector is None:
          self.rect_box_detector = RectBoxDetector(image = self.Image)
      else:
          self.rect_box_detector = rect_box_detector
      if line_box_detector is None:
          self.line_box_detector = LineBoxDetector(image = self.Image)
      else:
          self.line_box_detector = line_box_detector
      self.Circles = self.detect()

    def detect(self):
      boxes = self.get_circle_boxes()
      return boxes.copy()

    def pre_proc(self, line_min_width=15):
      image = self.Image.copy()
      # print(image.shape)
      image[np.any(image <= 250, axis=-1),:] = (0,0,0)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (5,5), 0)
      thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
      # cv2_imshow(image)
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
      opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=0)
      return opening

    def get_circle_boxes(self, other_fields = None):
      if other_fields is None:
        rect_fields = self.rect_box_detector.Boxes
        line_fields = self.line_box_detector.Boxes
        other_fields = rect_fields + line_fields
      image = self.Image.copy()
      temp_img = image.copy()
      for box in other_fields:
        box.mask(temp_img)
      opening = self.pre_proc()
      cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      fields = []
      for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.04*peri,True)
        area = cv2.contourArea(c)
        if(area>200 and area <1000):
          # n=approx.ravel()
          # roi = image[n[1]:n[3],n[0]:n[2]]
          # if(np.all(roi==255)):
            ((x,y),r)=cv2.minEnclosingCircle(c)
            #w=h=(2*r)/sqrt(2)
            #y1 = (r/sqrt(2))+y
            #x1 = (r/sqrt(2))+x
            # fields.append([int(x-r),int(y-r),int(x+r),int(y+r)])
            # w = h = (2 * r) / math.sqrt(2)
            # y1 = (r / math.sqrt(2)) + y
            # x1 = (r / math.sqrt(2)) + x
            # roi = temp_img[int(y1):int(y1)+int(h),int(x1):int(x1)+int(w)]
            roi = temp_img[int(y)-int(r)//2:int(y)+int(r)//2,int(x)-int(r)//2:int(x)+int(r)//2]
            if(np.all(roi>250)):
              fields.append(BoundingBox(int(x-r), int(y-r), 2*int(r), 2*int(r)))
      # for l in fields:
      #     cv2.rectangle(temp_img, (l[0],l[1]),(l[2],l[3]), (0, 0, 255), 2)
      return fields.copy()

    def mask_image(self):
      image = self.Image.copy()
      bboxes = self.detect()
      for box in bboxes:
        box.mask(image)
      return image

    def processed_image(self):
      image = self.Image.copy()
      bboxes = self.detect_with_template()
      for box in bboxes:
        box.draw(image)
      image = Image.fromarray(image)
      return image


class RectBoxDetector:
    def __init__(self, filepath = None, image = None):
      if(image is None):
        self.Image = cv2.imread(filepath)
      else:
        self.Image = image.copy()
      self.Boxes = self.detect()

    def detect(self):
      boxes = self.get_all_boxes()
      return boxes.copy()

    def detect_checkboxes(self):
        checkboxes=[]
        for box in self.Boxes:
            if box.is_checkbox():
                checkboxes.append(box)
        return checkboxes

    def detect_textboxes(self):
        textboxes=[]
        for box in self.Boxes:
            if not box.is_checkbox():
                textboxes.append(box)
        return textboxes

    def pre_proc(self, line_min_width=15):
      image = self.Image.copy()
      # image[np.any(image <= 250, axis=-1),:] = (0,0,0)
      gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      th1,img_bin=cv2.threshold(gray_scale,150,225,cv2.THRESH_BINARY)
      kernal_h=np.ones((1,line_min_width), np.uint8)
      kernal_v=np.ones((line_min_width,1), np.uint8)
      img_bin_h=cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal_h)
      img_bin_v=cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal_v)
      img_bin_final=img_bin_h|img_bin_v
      final_kernel=np.ones((3,3), np.uint8)
      img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=1)
      return img_bin_final

    def get_rect_boxes(self):
      img = self.Image.copy()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      processed_image = self.pre_proc()
      ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~processed_image, connectivity=8, ltype=cv2.CV_32S)
      detected_rects = stats[2:]
      bounding_boxes = []
      for rects in detected_rects:
        area = rects[4]
        if area>100:
          roi = gray[rects[1]:rects[1]+rects[3],rects[0]:rects[0]+rects[2]]
          if(np.all(roi == 255)):
            bounding_boxes.append(BoundingBox(rects[0], rects[1], rects[2], rects[3]))
          else:
            roi = gray[rects[1]:rects[1]+rects[3],rects[0]+rects[2]//3:rects[0]+rects[2]]
            if(np.all(roi == 255)):
              bounding_boxes.append(BoundingBox(rects[0]+rects[2]//3, rects[1], rects[2]-(rects[2]//3), rects[3]))
            else:
              roi = gray[rects[1]+rects[3]//2:rects[1]+rects[3],rects[0]:rects[0]+rects[2]]
              if(np.all(roi == 255)):
                bounding_boxes.append(BoundingBox(rects[0], rects[1]+rects[3]//3, rects[2], rects[3] - (rects[3]//3)))

      return bounding_boxes.copy()

    def get_all_boxes(self):
      rect_boxes = self.get_rect_boxes()
      image = self.Image.copy()
      tmp_img = image.copy()
      for box in rect_boxes:
        box.mask(tmp_img)
      # image[np.any(image <= 250, axis=-1),:] = (0,0,0)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      tmp_gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
      # cv2_imshow(gray)
      blur = cv2.GaussianBlur(gray, (5,5), 0)
      thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
      # cv2_imshow(thresh)
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
      opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
      cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      fields = []
      dict_fields = []
      boxes=rect_boxes.copy()
      for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x,y,w,h = cv2.boundingRect(approx)
        # print(approx.ravel())
        if (len(approx) == 4 or len(approx) == 6 or len(approx) == 7) and (area > 200) and (area < 80000):
            # boxes.append(BoundingBox(x,y,w,h))
            roi = tmp_gray[y+10:y+h-10,x+10:x+w-10]
            if roi.shape[0]==0 or roi.shape[1]==0:
              continue
            if h>10 and w>10:
                if((np.all(roi <= 10)) or (np.max(roi)-np.min(roi))<5):
                  # cv2.rectangle(tmp_img, (x,y),(x+w,y+h), (255, 255, 255), -1)
                  cv2.rectangle(tmp_gray, (x,y),(x+w,y+h), (0, 0, 0), -1)
                  boxes.append(BoundingBox(x,y,w,h))
                else:
                  # print(h,w)
                    eh = h//2
                    ew = w//3
                  # if(h>30 and w>60):
                  #   # boxes.append([x+32,y+h-32,x+w-4,y+h-4])
                  #   roi2 = tmp_img[y+eh:y+h-2,x+ew:x+w-2]
                  #   if(np.all(roi2 == 255)):
                  #     cv2.rectangle(tmp_img, (x,y),(x+w-2,y+h-2), (0, 0, 0), -1)
                  #     boxes.append([x+ew,y+eh,x+w-2,y+h-2])
                  # elif w>60:
                    if ew>1:
                        roi2 = tmp_gray[y+2:y+h-2,x+ew:x+w-2]
                        if(np.all(roi2 <= 10) or (np.max(roi)-np.min(roi))<5):
                          cv2.rectangle(tmp_gray, (x,y),(x+w-2,y+h-2), (0, 0, 0), -1)
                          boxes.append(BoundingBox(x+ew,y+2,w-2,h-2))
                      # elif h>30:
                        else:
                          if eh>1:
                              roi2 = tmp_gray[y+eh:y+h-2,x+2:x+w-2]
                              if(np.all(roi2 <= 10) or (np.max(roi)-np.min(roi))<5):
                                cv2.rectangle(tmp_gray, (x,y),(x+w-2,y+h-2), (0, 0, 0), -1)
                                boxes.append(BoundingBox(x+2,y+h//3,w-2,h-2))
      return boxes.copy()

    def mask_image(self):
      image = self.Image.copy()
      bboxes = self.Boxes
      for box in bboxes:
        box.mask(image)
      return image

    def processed_image(self):
      image = self.Image.copy()
      bboxes = self.detect()
      for box in bboxes:
        if box.is_checkbox():
            box.draw(image,(255,0,0))
        else:
            box.draw(image,(0,0,255))
      image = Image.fromarray(image)
      return image


class LineBoxDetector:
    def __init__(self, filepath = None, image = None, rect_box_detector = None):
      if image is None and filepath is not None:
        self.Image = cv2.imread(filepath)
      else:
        self.Image = image.copy()

      h, w=self.Image.shape[:2]
      self.shape = (w,h)
      if rect_box_detector is None:
          self.rect_box_detector = RectBoxDetector(image = self.Image)
      else:
          self.rect_box_detector = rect_box_detector
      self.Boxes = self.detect()

    def detect(self, rect_mask_image = None):
      img = self.Image.copy()
      if rect_mask_image is None:
          line_boxes = self.get_actual_fields()
      else:
          line_boxes = self.get_actual_fields(rect_mask_image.copy())
      line_boxes = [box for box in  line_boxes if box.is_valid(self.shape[0],self.shape[1])]
      return line_boxes.copy()

    def pre_proc(self, line_min_width=15):
      img = self.Image.copy()
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      lower_red = (0, 0, 255)
      upper_red = (255, 255, 255)
      mask = cv2.inRange(img, lower_red, upper_red)
      thresh = cv2.adaptiveThreshold(mask, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 70)
      horizontal_kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(15,1))
      detected_lines = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,horizontal_kernal,iterations=5)
      vertical_kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(1,15))
      detected_vertical_lines = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,vertical_kernal,iterations=5)
      kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
      dilated = cv2.dilate(detected_vertical_lines,kernal,iterations = 2)
      pixels = np.where(dilated==255)
      for i in range(len(pixels[0])):
        detected_lines[pixels[0][i]][pixels[1][i]] = 0
      return detected_lines

    def get_all_lines(self, tmp_img = None):
      img = self.Image.copy()
      # img[np.any(img <= 250, axis=-1),:] = (0,0,0)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      lower_red = (0, 0, 255)
      upper_red = (255, 255, 255)
      mask = cv2.inRange(img, lower_red, upper_red)
      thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 70)
      horizontal_kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(15,1))
      detected_lines = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,horizontal_kernal,iterations=5)
      cnts=cv2.findContours(detected_lines,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cnts=cnts[0] if len(cnts)==2 else cnts[1]
      lines = []
      if tmp_img is not None:
          gray = cv2.cvtColor(tmp_img,cv2.COLOR_BGR2GRAY)
      for cnt in cnts :
          approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
          # print(approx)
          n = approx.ravel()
          if(len(n) == 4):
            if (n[2]-n[0])>5:
              n[0]+=2
              n[3] = n[1]-2
              n[1]-=20
              n[2]-=2
              
              # lines.append(BoundingBox(n[0],n[1],n[2]-n[0],n[3]-n[1]))
              roi = gray[n[1]+2:n[3]-2,n[0]-2:n[2]+2]
              if(np.all(roi == 255)) and n[1]>0 and n[0]>0:
                lines.append(BoundingBox(n[0],n[1]-10,n[2]-n[0],n[3]-n[1]+10))
      return lines.copy()

    def get_fields_coordinates(self):
      detected_lines = self.pre_proc()
      cnts=cv2.findContours(detected_lines,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cnts=cnts[0] if len(cnts)==2 else cnts[1]
      lines = []
      actual_lines=[]
      i=0
      for cnt in cnts :
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        n = approx.ravel()

        if(len(n) == 4):
            if (n[2]-n[0])>5:
              actual_lines.append((n.copy(),i))
              n[0]+=2
              n[3] = n[1]-2
              n[1]-=50
              n[2]-=2
              lines.append(n)
        i+=1
      return {'lines':lines,'actual_lines':actual_lines}

    def get_actual_fields(self, img_tmp = None):
      if img_tmp is None:
        img_tmp = self.rect_box_detector.mask_image()
      line_boxes = self.get_all_lines(img_tmp)
      for box in line_boxes:
        box.mask(img_tmp)
      actual_fields=line_boxes.copy()

      fields=self.get_fields_coordinates()
      lines = fields['lines']
      actual_lines = fields['actual_lines']
      img = self.Image.copy()
      # img[np.any(img <= 250, axis=-1),:] = (0,0,0)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      tmp_gray = cv2.cvtColor(img_tmp,cv2.COLOR_BGR2GRAY)
      lower_red = (0, 0, 255)
      upper_red = (255, 255, 255)
      mask = cv2.inRange(img, lower_red, upper_red)
      # thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 70)
      thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
      list_indxs=[]
      
      for i,l in actual_lines:
        for j,m in actual_lines:
          if abs(j[0]-i[0])<2 and abs(j[2]-i[2])<2 and (j[1]-i[1])>5 and (j[3]-i[3])>5 and abs(j[3]-i[1])>5:
            roi = tmp_gray[i[1]+(j[3]-i[1])//3:j[3]-2,i[0]+32:j[2]-2]
            # cv2_imshow(roi)
            # print('----------')
            if np.all(roi==255):
              if(j[3]-i[1])>30 and (j[2]-i[0])>50:
                list_indxs.append(l)
                list_indxs.append(m)
                # actual_fields.append([i[0]+(j[3]-i[1])//3,i[1]+23,j[2],j[3]-3])
                actual_fields.append(BoundingBox(i[0]+32,i[1]+(j[3]-i[1])//3,j[2]-i[0]-32,(j[3]-i[1])-(j[3]-i[1])//3))
##            else:
##                f abs(j[0]-i[0])<2 and abs(j[2]-i[2])<2 and (j[1]-i[1])>5 and (j[3]-i[3])>5 and abs(j[3]-i[1])>5:
##                roi = tmp_gray[i[1]+(j[3]-i[1])//2 -5:j[3]-2,i[0]+32:j[2]-2]
##                # cv2_imshow(roi)
##                # print('----------')
##                if np.all(roi==255):
##                  if(j[3]-i[1])>30 and (j[2]-i[0])>50:
##                    list_indxs.append(l)
##                    list_indxs.append(m)
##                    # actual_fields.append([i[0]+(j[3]-i[1])//3,i[1]+23,j[2],j[3]-3])
##                    actual_fields.append(BoundingBox(i[0]+32,i[1]+(j[3]-i[1])//3,j[2]-i[0]-32,(j[3]-i[1])-(j[3]-i[1])//3))
      for i in range(len(lines)):
        if i not in list_indxs:
            l = lines[i].copy()
            # actual_fields.append(BoundingBox(l[0],l[1],l[2]-l[0],l[3]-l[1]))
            roi = tmp_gray[l[1]-20:l[3],l[0]:l[2]]
            roi2 = tmp_gray[0:l[3],l[0]:l[2]]
            if roi.shape[0]==0 or roi.shape[1]==0:
              continue
            if np.all(roi == 255) and not np.all(roi2==255):
              cv2.rectangle(tmp_gray, (l[0],l[1]),(l[2],l[3]), (0, 0, 0), -1)
              actual_fields.append(BoundingBox(l[0],l[1]-30,l[2]-l[0],(l[3]-l[1])+30))
            else:
              l[0]+=32
              roi = tmp_gray[l[1]-10:l[3],l[0]:l[2]]
              roi2 = tmp_gray[0:l[3],l[0]:l[2]]
              if roi.shape[0]==0 or roi.shape[1]==0:
                continue
              if np.all(roi == 255) and not np.all(roi2==255):
                cv2.rectangle(tmp_gray, (l[0],l[1]),(l[2],l[3]), (0, 0, 0), -1)
                actual_fields.append(BoundingBox(l[0],l[1]-30,l[2]-l[0],(l[3]-l[1])+30))
      return actual_fields.copy()

    def mask_image(self):
      image = self.Image.copy()
      bboxes = self.Boxes
      for box in bboxes:
        box.mask(image)
      return image

    def processed_image(self):
      image = self.Image.copy()
      bboxes = self.Boxes
      for box in bboxes:
        box.draw(image)
      image = Image.fromarray(image)
      return image
