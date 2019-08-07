""" Generate cut-words txt"""

from src.utils import word2vec


def read_data():
    filepath_dict = {
        '借款审批通过': '../data/借款审批通过.csv',
        '借款拒绝进件': '../data/借款拒绝进件.csv',
        '还款成功': '../data/还款成功.csv',
        '逾期': '../data/逾期.csv'
    }
    df_org = word2vec.read_from_labeled_txt(filepath_dict)
    data = df_org.loc[:, ['content', 'status']]
    return data


if __name__ == '__main__':
    org_data = read_data()
    df = word2vec.load_message_and_save2(org_data, ['sentence'])
    df['label'] = org_data['status']

    # print(df)

    df.to_csv('./tmp_files/sentence_label2.txt')




