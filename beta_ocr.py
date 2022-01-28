import base64
import PIL.Image
import pytesseract
import requests


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'pnm', 'webp'}


def allowed_file(name: str) -> bool:
    """Check whether the file name has an allowed extension
    :param name: image file name
    :return: True or False
    """
    return '.' in name and name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def replace_ext_to_txt(name: str) -> str:
    """Repalce the file extension to .txt
    :param name: original file name
    :return: file name whose extension replaced with .txt
    """
    return name.replace(name.rsplit('.', 1)[1], 'txt')


def run_ocr(image: str, mode: str) -> {}:
    """Run OCR on given image according to given mode
    :param image: image file path
    :param mode: 'bidu' or 'tess'
    :return: ocr result
    """
    res = []
    if mode == 'bidu':
        res = run_bidu_ocr(image)
    if mode == 'tess':
        res = run_tess_ocr(image)
    return res


def run_bidu_ocr(image: str) -> {}:
    """Run Baidu OCR on given image
    :param image: image file path
    :return: ocr result
    """
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {'grant_type': 'client_credentials',
              'client_id': 'YAdlb1AZWdHxX9mRs5S7Wgny',
              'client_secret': 'IMVSe74vmEqGdmEYEhIcgIww3m4il9Rh'}
    response = requests.get(url, params=params)
    access_token = response.json()['access_token']
    params = {'access_token': access_token}

    f = open(image, 'rb')
    im = base64.b64encode(f.read())
    data = {'image': im}
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/accurate'
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=data, params=params, headers=headers)
    res = response.json()
    return res


def run_tess_ocr(image: str) -> {}:
    """Run Tesseract OCR on given image
    :param image: image file path
    :return: ocr result
    """
    im = PIL.Image.open(image)
    im = im.resize(im.size, PIL.Image.LANCZOS)
    df = pytesseract.image_to_data(im, lang='chi_sim', config='--dpi 300',
                                   output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=['text'])
    df = df[df['text'] != ' ']
    res = {'words_result': [{'location': {'height': int(df.at[i, 'height']),
                                          'left': int(df.at[i, 'left']),
                                          'top': int(df.at[i, 'top']),
                                          'width': int(df.at[i, 'width'])},
                             'words': df.at[i, 'text']} for i in df.index]}
    return res