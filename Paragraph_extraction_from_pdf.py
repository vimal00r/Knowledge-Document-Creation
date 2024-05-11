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


            
        # if __name__ == "__main__":

                
        folder='V:\\Main project\\cit+f2t+t2t_ui\\'
        pdfsavingpath ='V:\\Main project\\cit+f2t+t2t_ui\\csv\\'
        csvsavingpath = 'V:\\Main project\\cit+f2t+t2t_ui\\csv\\'

        
        files = glob.glob(folder+"\\"+pdf_path)
    #         files=['Bestellung  Purchase Order 9450023201']
        for path in files:
            file=str(path.split("\\")[-1])
            pages = extract_pages(path)
            li=[]
            Whole=pd.DataFrame()
            pagelist=[]
            for page in pages:
                li=[]
                li= show_ltitem_hierarchy(page)
                df= pd.DataFrame(li)
                df['Page'] = str(page).split()[0].lstrip("<LTPage(").rstrip(")")
                pagelist.append( int(str(page).split()[0].lstrip("<LTPage(").rstrip(")")))
                Whole=pd.concat([df,Whole])
            reader = PdfReader(path)
            dic={}
            for i in pagelist:
                boxdim = reader.pages[i-1].mediabox
                x_max=boxdim.width
                y_max=boxdim.height
                dic[i]=[float(x_max),float(y_max)]
            New=pd.DataFrame()
            for page in range(1,max(pagelist)+1):
                    k=int(page)
                    x_max,y_max= dic[k]
                    df1=Whole[Whole['Page']==str(page)]
                    df1['newy1']=df1['y1'].apply(lambda x: abs(float(y_max)-float(x)))
                    df1['newy2']=df1['y2'].apply(lambda x: abs(float(y_max)-float(x)))
                    df1['y1'] = df1['newy2']
                    df1['y2'] = df1['newy1']
                    df1= df1.drop(columns=['newy1','newy2'],axis=1)
                    New=pd.concat([New,df1])
            New.to_csv(csvsavingpath+ file.replace('.pdf','.csv'))
            doc = fitz.open(path)

            for page in doc:

                pageno= int(str(page).split()[1])+1

                df2=New[New['Page']==str(pageno)][['x1','y1','x2','y2']]
                df2['x1']= df2['x1'].apply(lambda x: int(x))
                df2['y1']= df2['y1'].apply(lambda x: int(x))
                df2['x2']= df2['x2'].apply(lambda x: int(x))
                df2['y2']= df2['y2'].apply(lambda x: int(x))
                df2box= df2.values.tolist()

                for rec in df2box:
                    page.draw_rect(rec,  color = (0, 0, 1), width = 1)
            doc.save(pdfsavingpath + file)

        return file.split(".")[0]+".csv"

#         print(file.split(".")[0]+".csv")
# create_csv("gemini.pdf")

