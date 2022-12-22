import docx
import json
import numpy as np
import os
import pandas as pd
import pdf2docx
import requests
import shutil
import tabula
import time

import beta_utils


PDF_EXTENSION = '.pdf'
DOC_EXTENSION = '.doc'
DOCX_EXTENSION = '.docx'


def get_df_header_from_excel(file_path: str) -> pd.DataFrame:
    """Get pandas DataFrame of quarterly table header from an Excel file
    :param file_path: Excel file path of table header
    :return: a pandas DataFrame
    """
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df.drop_duplicates().sort_values(by=['label', 'text'])
    return df.reset_index(drop=True)


def run_sentence_encoder(los: [str]) -> [[float]]:
    """Run sentence encoder of list of sentences
    :param los: list of sentences
    :return: the embeddings
    """
    u = 'https://w-1.test.betawm.com/athena/sentence_encoder'
    return json.loads(requests.post(u, data={'s0': str(los)}).text)['s0_embed']


def extract_tables_from_pdf_files_in_dir_using_docx(data_path: str,
                                                    df_header: pd.DataFrame,
                                                    e_header: [[float]]
                                                    ) -> None:
    """Extract tables from PDF files in a directory using python-docx
    :param data_path: data path of PDF files
    :param df_header: pandas DataFrame of header sheet
    :param e_header: header embeddings in dims of (header count, embedding dim)
    :return: None
    """
    for file_name in beta_utils.sort_numerically(os.listdir(data_path)):
        if file_name.endswith(PDF_EXTENSION):
            print(file_name)
            file_path = os.path.join(data_path, file_name)
            temp_file_name = file_name.replace(PDF_EXTENSION, DOCX_EXTENSION)
            temp_file_path = os.path.join(data_path, temp_file_name)
            cv = pdf2docx.Converter(file_path)
            cv.convert(temp_file_path)
            cv.close()
            dfs = extract_tables_from_one_docx(temp_file_path,
                                               df_header,
                                               e_header)
            os.remove(temp_file_path)
            print(*[df for df in dfs], sep='\n\n')
            print()


def extract_tables_from_one_docx(file_path: str,
                                 df_header: pd.DataFrame,
                                 e_header: [[float]]) -> [pd.DataFrame]:
    """Extract tables from a DOCX document
    :param file_path: path of a DOCX document
    :param df_header: pandas DataFrame of header sheet
    :param e_header: header embeddings in dims of (header count, embedding dim)
    :return: list of pandas DataFrame
    """
    dfs = []
    doc = docx.api.Document(file_path)
    for table in doc.tables:
        header = []
        body = []
        for i, row in enumerate(table.rows):
            row_text = [cell.text for cell in row.cells]
            row_text = drop_useless_char_in_list(row_text)
            if row_text_is_table_header(row_text,
                                        df_header,
                                        e_header) and not body:
                header.append(row_text)
            else:
                body.append(row_text)
        header = combine_table_header(header)
        if header:
            dfs = update_dfs_if_header(dfs, header, body)
        else:
            dfs = update_dfs_if_not_header(dfs, body)
    return dfs


def drop_useless_char_in_list(los: [str]) -> [str]:
    """Remove unnecessary characters in a list of string
    :param los: list of string
    :return: list of string
    """
    los = [s.replace('\n', ' ') for s in los]
    los = [s.replace('\r', ' ') for s in los]
    los = [s.replace('\t', ' ') for s in los]
    return [beta_utils.drop_unnecessary_whitespace(s) for s in los]


def row_text_is_table_header(row_text: [str],
                             df_header: pd.DataFrame,
                             e_header: [[float]]) -> bool:
    """Judge whether the row text is potentially a table header
    :param row_text: list of string, representing row text
    :param df_header: pandas DataFrame of header sheet
    :param e_header: header embeddings in dims of (header count, embedding dim)
    :return: whether the row text is a tabel header
    """
    ds = {'Header': 0, 'Non-Header': 1}
    e_row = run_sentence_encoder(row_text)
    ss = calculate_similarity_scores(e_header, e_row)
    loi = find_indices_of_most_similar(ss)
    ms = np.mean([ds[df_header.loc[i, 'label']] for i in loi])
    return True if ms < 0.5 else False


def combine_table_header(header: [[str]]) -> [str]:
    """Combine table header into one header if more than one
        For example, the input header is,
            [['a', 'b', 'c', 'c'], ['a', 'b', '1', '2']].
        The output header would be,
            ['a', 'b', 'c1', 'c2']
    :param header: table headers in a list of lists of strings
    :return: table header in a list of strings
    """
    if not header:
        return []
    res = []
    for i in range(len(header[0])):
        col_name = header[0][i]
        for row in header[1:]:
            if col_name != row[i]:
                col_name += row[i]
        res.append(col_name)
    return res


def update_dfs_if_header(dfs: [pd.DataFrame],
                         header: [str],
                         body: [[str]]) -> [pd.DataFrame]:
    """Update dfs based on header and body of the new table if header exists
    :param dfs: list of pandas DataFrame
    :param header: list of string
    :param body: list of lists of string
    :return: list of pandas DataFrame
    """
    dfs.append(pd.DataFrame(body, columns=header))
    return dfs


def calculate_similarity_scores(se: [[float]], qe: [[float]]) -> np.ndarray:
    """Calculate similarity between two embeddings of list of phrases
    :param se: header embeddings in dims of (header count, embedding dim)
    :param qe: query embeddings in dims of (query count, embedding dim)
    :return: similarity scores in dims of (header count, query count)
    """
    return np.inner(se, qe)


def find_indices_of_most_similar(ss: np.ndarray) -> np.ndarray:
    """Find the indices of most similar header text
    :param ss: similarity scores in dims of (header count, query count)
    :return: list of indices
    """
    return np.argmax(ss, axis=0)


def update_dfs_if_not_header(dfs: [pd.DataFrame],
                             body: [[str]]) -> [pd.DataFrame]:
    """Update dfs based on body of the new table if header does not exist
    :param dfs: list of pandas DataFrame
    :param body: list of lists of string
    :return: list of pandas DataFrame
    """
    if dfs and len(dfs[-1].columns) == len(body[0]):
        if len(dfs[-1].index) in [0, 1]:
            if len(dfs[-1].index) == 0:
                dfs[-1] = combine_df_with_table(dfs[-1], body)
            if len(dfs[-1].index) == 1:
                if contents_relevant_in_corresponding_column(dfs[-1], body):
                    dfs[-1] = combine_df_with_table(dfs[-1], body)
                else:
                    dfs.append(pd.DataFrame(body))
        else:
            if row_broken_into_two(dfs[-1], body):
                dfs[-1] = combine_two_rows_and_combine_table(dfs[-1], body)
            else:
                if contents_relevant_in_corresponding_column(dfs[-1], body):
                    dfs[-1] = combine_df_with_table(dfs[-1], body)
                else:
                    dfs.append(pd.DataFrame(body))
    else:
        dfs.append(pd.DataFrame(body))
    return dfs


def combine_df_with_table(df: pd.DataFrame,
                          table: [[str]]) -> pd.DataFrame:
    """Combine a pandas DataFrame with a 2d array
    :param df: pandas DataFrame, representing the previous table
    :param table: 2d array, representing the new table
    :return: the combined table in pandas DataFrame
    """
    return pd.concat([df, pd.DataFrame(table, columns=df.columns)],
                     ignore_index=True)


def contents_relevant_in_corresponding_column(df: pd.DataFrame,
                                              table: [[str]]) -> bool:
    """Check whether contents in corresponding column are relevant
    :param df: pandas DataFrame, representing the previous table
    :param table: 2d array, representing the new table
    :return: whether contents in corresponding column are relevant
    """
    score = similarity_between_two_lists(df.iloc[-1].tolist(), table[0])
    return True if score > 0.5 else False


def row_broken_into_two(df: pd.DataFrame, table: [[str]]) -> bool:
    """Check whether a row has been broken into two
    :param df: pandas DataFrame, representing the previous table
    :param table: 2d array, representing the new table
    :return: whether a row has been broken into two
    """
    row_u = df.iloc[-1].tolist()
    row_d = table[0]
    row_combined = [u + d for (u, d) in zip(row_u, row_d)]
    other_rows_u = df.iloc[:-1].to_numpy().tolist()
    other_rows_d = table[1:]
    other_rows = other_rows_u + other_rows_d
    scores_btw_u_and_other = [similarity_between_two_lists(row_u, row)
                              for row in other_rows]
    scores_btw_d_and_other = [similarity_between_two_lists(row_d, row)
                              for row in other_rows]
    best_score_u = np.amax(scores_btw_u_and_other)
    best_other_for_u = other_rows[np.argmax(scores_btw_u_and_other)]
    best_score_d = np.amax(scores_btw_d_and_other)
    best_other_for_d = other_rows[np.argmax(scores_btw_d_and_other)]
    if best_score_u > best_score_d:
        best_score = best_score_u
        best_other_row = best_other_for_u
    else:
        best_score = best_score_d
        best_other_row = best_other_for_d
    score_btw_combined_and_best_other = similarity_between_two_lists(
        row_combined, best_other_row)
    return True if score_btw_combined_and_best_other > best_score else False


def combine_two_rows_and_combine_table(df: pd.DataFrame,
                                       table: [[str]]) -> pd.DataFrame:
    """Combine last row in DataFrame and first row in table into one row,
    and combine the DataFrame with the 2d array
    :param df: pandas DataFrame
    :param table: list of lists of string
    :return: pandas DataFrame
    """
    df.iloc[-1] = [a + b for (a, b) in zip(df.iloc[-1], table.pop(0))]
    return pd.concat([df, pd.DataFrame(table, columns=df.columns)],
                     ignore_index=True)


def similarity_between_two_lists(l1: [str], l2: [str]) -> float:
    """Similarity score of corresponding elements of two lists
    :param l1: list of string
    :param l2: list of string
    :return: similarity score
    """
    losim = [similarity_between_two_strs(s1, s2) for (s1, s2) in zip(l1, l2)]
    return float(np.mean(losim))


def similarity_between_two_strs(s1: str, s2: str) -> float:
    """Similarity score between two strings
    :param s1: string
    :param s2: string
    :return: similarity score
    """
    u = 'https://w-1.test.betawm.com/athena/semantic_similarity'
    return json.loads(requests.post(u, data={'s1': s1, 's2': s2}).text)


def extract_tables_from_pdf_files_in_dir_using_tabula(data_path: str,
                                                      df_header: pd.DataFrame,
                                                      e_header: [[float]]
                                                      ) -> None:
    """Extract tables from PDF files in a directory using tabula-py
    :param data_path: data path of PDF files
    :param df_header: pandas DataFrame of header sheet
    :param e_header: header embeddings in dims of (header count, embedding dim)
    :return: None
    """
    for file_name in beta_utils.sort_numerically(os.listdir(data_path)):
        if file_name.endswith(PDF_EXTENSION):
            print(file_name)
            file_path = os.path.join(data_path, file_name)
            temp_file_name = str(time.time()) + PDF_EXTENSION
            temp_file_path = os.path.join(data_path, temp_file_name)
            shutil.copyfile(file_path, temp_file_path)
            tables = tabula.read_pdf(temp_file_path,
                                     guess=False, pages='all', lattice=True)
            os.remove(temp_file_path)
            dfs = []
            for table in tables:
                header = []
                body = []
                l2d = convert_df_to_list(table)
                l2d = drop_useless_char_in_2d_list(l2d)
                for row_text in l2d:
                    row_text = do_specific_operations_on_list(row_text)
                    if row_text_is_table_header(
                            row_text, df_header, e_header) and not body:
                        header.append(row_text)
                    else:
                        body.append(row_text)
                header = combine_table_header(header)
                if header:
                    dfs = update_dfs_if_header(dfs, header, body)
                else:
                    dfs = update_dfs_if_not_header(dfs, body)
                dfs = do_specific_operations_on_last_df(dfs)
            print(*[df for df in dfs], sep='\n\n')
            print()


def convert_df_to_list(df: pd.DataFrame) -> [[str]]:
    """Convert DataFrame into a list with columns names as the first element,
    and the values as the remaining elements
    :param df: a pandas DataFrame
    :return: list of lists of string
    """
    columns = df.columns.astype(str).tolist()
    data = df.values.astype(str).tolist()
    data.insert(0, columns)
    return data


def drop_useless_char_in_2d_list(l2d: [[str]]) -> [[str]]:
    """Remove unnecessary characters in a 2d list
    :param l2d: 2d list
    :return: 2d list without unnecessory characters
    """
    return [[drop_useless_char_in_str(c) for c in r] for r in l2d]


def drop_useless_char_in_str(s: str) -> str:
    """Remove unnecessary characters in a string
    :param s: string
    :return: string
    """
    s = s.replace('\n', ' ')
    s = s.replace('\r', ' ')
    s = s.replace('\t', ' ')
    return beta_utils.drop_unnecessary_whitespace(s)


def do_specific_operations_on_list(los: [str]) -> [str]:
    """Do some specific operations on a list
    :param los: list of string
    :return: list of string after operations
    """
    los = [s.replace('nan', '') for s in los]
    los = process_unnamed_string_in_list(los)
    return los


def process_unnamed_string_in_list(los: [str]) -> [str]:
    """Process unnamed string in a list
    :param: list of strings
    :return: list of strings
    """
    count = 0
    for s in los:
        if 'unnamed' in s.lower():
            count += 1
    if count == 2:
        los = process_two_unnamed_in_list(los)
    return los


def process_two_unnamed_in_list(los: [str]) -> [str]:
    """Process a list that the number of elements containing 'unnamed' is 2
    :param los: list of string
    :return: list of string
    """
    if 'unnamed' in los[-1].lower() and 'unnamed' in los[-2].lower():
        if len(los) >= 4:
            los[-1] = los[-2] = los[-3]
            los[-3] = los[-4]
    return los


def do_specific_operations_on_last_df(dfs: [pd.DataFrame]) -> [pd.DataFrame]:
    """Do some specific operations of the last pandas DataFrame
    :param dfs: list of pandas DataFrame
    :return: list of pandas DataFrame after operations on the last DataFrame
    """
    c_name = '序号'
    if c_name in dfs[-1].columns:
        dfs[-1][c_name] = convert_float_str_to_int_str(dfs[-1][c_name])
    return dfs


def convert_float_str_to_int_str(series: pd.Series) -> pd.Series:
    """Convert float string to int string
    :param series: pandas Series
    :return: pandas Series after conversion
    """
    los = series.to_list()
    int_str_list = []
    for s in los:
        try:
            int_str_list.append(str(int(float(s))))
        except ValueError:
            int_str_list.append(s)
    return pd.Series(int_str_list)
