import base64
import os
import PIL.Image
import pprint
import pytesseract
import requests


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'ocr', 'fig')


def run_ocr(data_dir: str):
    """Run OCR on image files in a directory"""
    for image in sorted(os.listdir(data_dir)):
        im = PIL.Image.open(os.path.join(data_dir, image))
        print(image, im.mode, im.size)
        im = im.resize(im.size, PIL.Image.LANCZOS)
        df = pytesseract.image_to_data(im, lang='chi_sim', config='--dpi 300',
                                       output_type=pytesseract.Output.DATAFRAME)
        df = df.dropna(subset=['text'])
        df = df[df['text'] != ' ']
        rs = {'words_result': [{'location': {'height': int(df.at[i, 'height']),
                                             'left': int(df.at[i, 'left']),
                                             'top': int(df.at[i, 'top']),
                                             'width': int(df.at[i, 'width'])},
                                'words': df.at[i, 'text']} for i in df.index]}
        s = rs['words_result']
        pprint.pprint(s)


def run_baidu_ocr(data_dir: str):
    """Run Baidu OCR on image files in a directory"""
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {'grant_type': 'client_credentials',
              'client_id': 'YAdlb1AZWdHxX9mRs5S7Wgny',
              'client_secret': 'IMVSe74vmEqGdmEYEhIcgIww3m4il9Rh'}
    response = requests.get(url, params=params)
    access_token = response.json()['access_token']
    mode = 'ab'
    url_dict = {
        'ab': 'https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic',
        'accurate': 'https://aip.baidubce.com/rest/2.0/ocr/v1/accurate',
        'gb': 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic',
        'general': 'https://aip.baidubce.com/rest/2.0/ocr/v1/general',
        'form': 'https://aip.baidubce.com/rest/2.0/ocr/v1/form'
    }
    url = url_dict[mode]
    params = {'access_token': access_token}
    for image in sorted(os.listdir(data_dir)):
        im = PIL.Image.open(os.path.join(data_dir, image))
        print(image, im.mode, im.size)
        f = open(os.path.join(data_dir, image), 'rb')
        im = base64.b64encode(f.read())
        data = {'image': im}
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(url, data=data, params=params, headers=headers)
        rs = response.json()
        if 'error_msg' in rs:
            print(rs['error_msg'], rs['error_code'])
            break
        if mode in {'ab', 'gb'}:
            s = '\n'.join([r['words'] for r in rs['words_result']]) + '\n'
            print(s)
        if mode in {'accurate', 'general'}:
            s = rs['words_result']
            pprint.pprint(s)
        if mode in {'form'}:
            for s in rs['forms_result']:
                print('\n'.join([r.get('words', '') for r in s['header']]))
                print('\n'.join([r.get('words', '') for r in s['body']]))


run_baidu_ocr(data_root_path)
