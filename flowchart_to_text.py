import os
import google.generativeai as genai
import PIL.Image
import textwrap
from IPython.display import Markdown

os.environ['GOOGLE_API_KEY']='AIzaSyDQt-KelA3sZl6zNBCTe-PcDelw_tBM4wM'   #            'AIzaSyD7A8yvV81-4y6JGjyqjyMQnFO_CC6H_cY'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def flowchart_image_to_text(image_path):
    model = genai.GenerativeModel('gemini-pro-vision')
    img = PIL.Image.open(image_path)
    response = model.generate_content(["Write a Car infotainment system based description on this picture in English about 70 words.", img], stream=True)
    response.resolve()
    res = response.text

    return res


# flowchart_image_to_text("images.jpg")