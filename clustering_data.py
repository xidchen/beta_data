import json
import os
import pandas as pd
import re
import requests


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')


def load_choc_data() -> pd.DataFrame:

    choc_data = pd.read_csv(
        'https://download.mlcc.google.com/mledu-datasets/flavors_of_cacao.csv',
        sep=',', encoding='latin-1')

    choc_data.columns = ['maker', 'specific_origin', 'reference_number',
                         'review_date', 'cocoa_percent', 'maker_location',
                         'rating', 'bean_type', 'broad_origin']

    choc_data['bean_type'] = choc_data['bean_type'].fillna('Blend')
    choc_data['cocoa_percent'] = choc_data['cocoa_percent'].str.strip('%')
    choc_data['cocoa_percent'] = pd.to_numeric(choc_data['cocoa_percent'])
    choc_data['maker_location'] = choc_data['maker_location'].replace(
        'Amsterdam', 'Netherlands').replace('U.K.', 'England').replace(
        'Niacragua', 'Nicaragua').replace('Domincan Republic',
                                          'Dominican Republic')

    def cleanup_spelling_abbrev(text):
        replacements = [
            ['-', ', '], ['/ ', ', '], ['/', ', '], [' and', ', '],
            [' &', ', '], [', $', ''], [',  ', ', '], [', ,', ', '],
            ['\xa0', ' '], [r'\(', ', '], [r'\)', ''], [r',\s+', ','],
            ['Dom Rep|DR|Domin Rep|Dominican Rep,|Domincan Republic',
             'Dominican Republic'], ['Mad,|Mad$', 'Madagascar, '],
            ['PNG', 'Papua New Guinea, '], ['Guat,|Guat$', 'Guatemala, '],
            ['Ven,|Ven$|Venez,|Venez$', 'Venezuela, '],
            ['Ecu,|Ecu$|Ecuad,|Ecuad$', 'Ecuador, '],
            ['Nic,|Nic$', 'Nicaragua, '], ['Cost Rica', 'Costa Rica'],
            ['Mex,|Mex$', 'Mexico, '], ['Jam,|Jam$', 'Jamaica, '],
            ['Haw,|Haw$', 'Hawaii, '], ['Gre,|Gre$', 'Grenada, '],
            ['Tri,|Tri$', 'Trinidad, '], ['C Am', 'Central America'],
            ['S America', 'South America'], [' Bali', ',Bali']
        ]
        for i, j in replacements:
            text = re.sub(i, j, text)
        return text

    choc_data['specific_origin'] = choc_data['specific_origin'].replace(
        '.', '').apply(cleanup_spelling_abbrev)
    choc_data['broad_origin'] = choc_data['broad_origin'].fillna(
        choc_data['specific_origin'])
    choc_data['broad_origin'] = choc_data['broad_origin'].replace(
        '.', '').apply(cleanup_spelling_abbrev)
    choc_data.loc[choc_data['bean_type'].isin(
        ['Trinitario, Criollo']), 'bean_type'] = 'Criollo, Trinitario'
    choc_data.loc[choc_data['maker'] == 'Shattel', 'maker'] = 'Shattell'
    choc_data['maker'] = choc_data['maker'].replace('Naï¿½ve', 'Naive')

    return choc_data


def load_beta_user_profile_data() -> pd.DataFrame:

    beta_data = pd.read_csv(
        os.path.join(data_root_dir, 'beta_user_profile.csv'),
        sep=',',
        names=['sex', 'location', 'prod2', 'prod1', 'asset', 'topic'],
        usecols=[1, 2, 3, 4, 5, 6],
        encoding='utf-8')
    return beta_data


def request_location_data(addresses: pd.Series) -> pd.DataFrame:
    """Request geo locations for a list of adresses given"""

    url = 'https://apis.map.qq.com/ws/geocoder/v1/'
    params = {'key': 'TQPBZ-6YHC3-GCU3G-3OJJR-KZLGV-XCBIC'}

    loc_data = []
    for i, address in enumerate(addresses):
        params['address'] = address
        response = requests.get(url, params=params)
        rs = json.loads(response.text)
        lng = lat = None
        if rs['status'] == 0:
            lng = rs['result']['location']['lng']
            lat = rs['result']['location']['lat']
        loc_data.append([address, lng, lat])
        print([i, address, lng, lat])
    loc_data = pd.DataFrame(loc_data, columns=['location', 'lng', 'lat'])
    data_file_str = os.path.join(data_root_dir, 'beta_geo_locations.csv')
    loc_data.to_csv(data_file_str, index=False)
    return loc_data


def load_geo_location_data() -> pd.DataFrame:

    data_file_str = os.path.join(data_root_dir, 'beta_geo_locations.csv')
    geo_data = pd.read_csv(data_file_str, sep=',', encoding='utf-8')
    return geo_data
