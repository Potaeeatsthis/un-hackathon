from pdf2image import convert_from_path
import easyocr
import numpy as np

reader = easyocr.Reader(['en', 'th'])
pages = convert_from_path('./Teeramate-Resume.pdf')
for i, page in enumerate(pages):
    result = reader.readtext(np.array(page), detail=0)
    print(f"--- Page {i+1} ---")
    print('\n'.join(result))
