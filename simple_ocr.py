import os
import PIL.Image
import pytesseract


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'ocr', 'fig')

for image in sorted(os.listdir(data_root_path)):
    im = PIL.Image.open(os.path.join(data_root_path, image))
    print(image, im.mode, im.size, im.info)
    im = im.resize(im.size, PIL.Image.LANCZOS)
    s = pytesseract.image_to_string(im, lang='chi_sim')
    while ' ' in s or '\n\n' in s:
        s = s.replace(' ', '').replace('\n\n', '\n')
    print(s)
