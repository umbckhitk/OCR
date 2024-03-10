import cv2
import numpy as np
import imutils
import pytesseract
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter import filedialog
import requests 
import PyPDF2
import os
import re
import joblib
import json
from flask import Flask, render_template, request

app = Flask(__name__)


import google.generativeai as genai
googleapikey= 'AIzaSyCuTI33ggZMmp2xPJT_nxwlAr-GIqk0xCI'

genai.configure(api_key=googleapikey)

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

modelsvc = joblib.load("/Users/harishkrishna/Desktop/project/docClasifier.py/text_clf.joblib")

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config)
def textclean(text):
    text_cleaned = re.sub(r'\n', ' ', text)
    
    text_cleaned_lower=text_cleaned.lower()
    return text_cleaned_lower

def image_extract(filename):
    try:
        img = cv2.imread(filename)
        print(filename)
        img = cv2.resize(img, (620,480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
        gray = cv2.bilateralFilter(gray, 11, 18, 17)
        extracted_text = pytesseract.image_to_string(gray,config='--psm 4')
        return extracted_text
    except Exception as e:
        print(str(e))

# def document_clasifier(text):
#     prompt_parts = [ "what type of document that is extracted is this in three words '"+text+"'" ]

#     response = model.generate_content(prompt_parts)
#     text= response.text
#     print(text)
#     messagebox.showinfo("Result", text)

    
# def document_classifier_linearSVC(text):
#     predicted_label = modelsvc.predict([text])
#     messagebox.showinfo("Result", predicted_label[0])


app = Flask(__name__ , template_folder = "template")


@app.route('/submit_data', methods=["POST"])
def submit_data():
    filename = request.form["name"]
    # filename = filedialog.askopenfilename()    
    
    print(filename)
    if(filename.lower().endswith('.jpg') or filename.lower().endswith('.png')):
        text=image_extract(filename)


        # Print the extracted text
        
       
        
    elif (filename.lower().endswith('.pdf')): #pdfextract
        text = ""
        with open(filename, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(num_pages)
            for page_num in range(num_pages):
                page =pdf_reader.pages[page_num]
                text += page.extract_text()

    if(text):
        text = textclean(text)
        # document_clasifier(text)
        # document_classifier_linearSVC(text)
        print(text)
        #result = text.json()
    
    # else :
    #     pdf = PdfDocument.FromFile(filename)

        
    #     # Extract all pages to a folder as image files
    #     pdf.RasterizeToImageFiles("/Users/harishkrishna/Desktop/imgs/*.png",DPI=100)
    #     for i in range(num_pages):
    #         text += image_extract('/Users/harishkrishna/Desktop/imgs/'+str(i+1)+'.png')
            
          
        print(text)
        text= text
        return text

        # document_clasifier(text)
        # document_classifier_linearSVC(text) #pretrained model 
         #summerization model



@app.route('/index')
def index():
    
    
    # Render the template with data
    return render_template('index.html')

if __name__ == '__main__':
 app.run(debug=True)