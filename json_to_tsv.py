import os
import json
import argparse
from tqdm import tqdm

import pandas as pd


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--dir',
        required=True,
        help="Directory contains dataset."
    )
    p.add_argument(
        '--save_fn',
        required=True,
        help="File name to save tsv data."
    )

    config = p.parse_args()

    return config


def json_to_tsv(file: str):
    cols = ['corpus_id', 'document_id', 'sentence_id', 'sentence', 'label']
    df = pd.DataFrame(columns=cols)
    id = 0


    with open(file) as f:
        DATA = json.loads(f.read())

    label = []
    for document in tqdm(DATA['document']):
        for sentence in document['sentence']:
            df.loc[id, 'corpus_id'] = DATA['id']
            df.loc[id, 'document_id'] = document['id']
            df.loc[id, 'sentence_id'] = sentence['id']
            df.loc[id, 'sentence'] = sentence['form']
            labels = dict()
            for entity in sentence['NE']:
                key = entity['id']
                entity.pop('id')
                labels[key] = entity
            label.append(labels)
            id += 1
    df['label'] = label
    
    return df


def main(config):
    filepath = config.dir
    savepath = config.save_fn

    if os.path.isdir(filepath):
        dfs = []
        for file in tqdm(os.listdir(config.dir)):
            df = json_to_tsv(os.path.join(filepath, file))
            dfs.append(df)
        data = pd.concat(dfs)
    else:
        data = json_to_tsv(filepath)

    data.to_csv(savepath, sep='\t', index=False)

if __name__ == '__main__':
    config = define_argparser()
    main(config)