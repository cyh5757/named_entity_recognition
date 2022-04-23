import os
import json
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--data_path',
        required=True,
        help="Directory where data files located."
    )
    p.add_argument(
        "--save_path",
        required=True,
        help="Directory to save preprocessed dataset."
    )
    p.add_argument(
        '--test_size',
        required=True,
        default=.2,
        type=float,
        help="Set test size. Input float number"
    )
    p.add_argument(
        '--save_all',
        action="store_true",
        help="If true, save concatenated data as well as train and test data"
    )

    config = p.parse_args()

    return config


def main(config):
    """
    Prepreccs data files and split as train and test set.
    1. Read all data files in the directory.
    2. Concatenate files and drop useless columns.
    3. Add labels for experiments and data split.
    """
    filepath = config.data_path
    savepath = config.save_path

    # Read data files.
    if os.path.isdir(filepath):
        file_list = os.listdir(filepath)
    print(f"{len(file_list)} files found : ", file_list)

    for i, file in enumerate(file_list):
        file_list[i] = pd.read_csv(os.path.join(filepath, file), sep='\t')
        print(f"file {i} : ", file_list[i].shape[0])

    # Concatenate all data files in the directory.
    data = pd.concat(file_list, axis=0, ignore_index=True)
    print(f"|data before preprocessing| {data.shape[0]}")

    # 
    data['source'] = data['corpus_id'].str[0]
    data = data.drop(columns=['corpus_id', 'document_id', 'sentence_id'], axis=1)
    data = data.dropna(axis=0).reset_index(drop=True)
    data['label'] = data['label'].map(eval)

    NE_list = ["PS", "FD", "TR", "AF", "OG", "LC", "CV", "DT", "TI", "QT", "EV", "AM", "PT", "MT", "TM"]
    NE_counter = dict(zip(NE_list, [0] * 15))


    def get_label_list(ne_dict):
        label_list = []
        for _, values in ne_dict.items():
            label_list.append(values['label'][:2])

        return label_list

    data['label_list'] = data['label'].map(get_label_list)
    for label_list in data['label_list']:
        for label in label_list:
            for ne in NE_list:
                if label == ne:
                    NE_counter[ne] += 1

    NE_list_sorted = pd.DataFrame(NE_counter, index = ['count']).T.reset_index().sort_values(by='count', ignore_index=True)

    sentence_class = []
    for label_list in data['label_list']:
        if len(label_list) < 1:
            sentence_class.append('Out')
            continue
        else:
            for ne in NE_list_sorted['index']:
                if ne in label_list:
                    sentence_class.append(ne)
                    break

    data['sentence_class'] = sentence_class
    data = data.drop(columns=["label_list"])
    print(f"|data after preprocessing| {data.shape[0]}")

    train, test = train_test_split(data, test_size=config.test_size, stratify=data['sentence_class'])
    print(f"|train| {train.shape[0]} / |test| {test.shape[0]}")

    if config.save_all:
        data.to_csv(os.path.join(savepath, 'data.tsv'), sep='\t', index=False)
    train.to_csv(os.path.join(savepath, 'train.tsv'), sep='\t', index=False)
    test.to_csv(os.path.join(savepath, 'test.tsv'), sep='\t', index=False)


if __name__ == "__main__":
    config = define_argparser()
    main(config)