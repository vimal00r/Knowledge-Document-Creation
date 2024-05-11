from pathlib import Path
from typing import Iterable, Any

from pdfminer.high_level import extract_pages
import pandas as pd
import fitz
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from PyPDF2 import PdfReader
import warnings
warnings.filterwarnings('ignore')
import glob

def create_csv(pdf_path):
        def get_indented_name(o: Any, depth: int) -> str:
            """Indented name of LTItem"""
            if "LTTextBoxHorizontal" in str(o):
                return '  ' * depth + o.__class__.__name__

            else:
                return ''


        def get_optional_bbox(o: Any) -> str:
            """Bounding box of LTItem if available, otherwise empty string"""
            if hasattr(o, 'bbox'):
                return ''.join(f'{i:<4.0f}' for i in o.bbox)
            return ''


        def get_optional_text(o: Any) -> str:
            """Text of LTItem if available, otherwise empty string"""
            if hasattr(o, 'get_text'):
                return o.get_text().strip()
            return ''

        def show_ltitem_hierarchy(o: Any, depth=0):
            dic={}
            #print("=======>",o)
            if "LTTextBoxHorizontal" in str(o):
                dic['element'] = get_indented_name(o, depth)
                dic['x1'] = get_optional_bbox(o).split()[0]
                dic['y1'] = get_optional_bbox(o).split()[1]
                dic['x2'] = get_optional_bbox(o).split()[2]
                dic['y2'] = get_optional_bbox(o).split()[3]
                dic['text'] = get_optional_text(o)
                li.append(dic)    
            if isinstance(o, Iterable):
                for i in o:
                    show_ltitem_hierarchy(i, depth=depth + 1)
            return li

        li=[]


            
        

        return file.split(".")[0]+".csv"

#         print(file.split(".")[0]+".csv")
# create_csv("gemini.pdf")

