from os import truncate
import pandas as pd
import torch
from transformers import AutoTokenizer
import argparse
import sys


def argument_parser():

    p = argparse.ArgumentParser()

    p.add_argument('--file_fn', required=True)
    p.add_argument('--pretrained', default='klue/bert-base')
    config = p.parse_args()

    return config


def tokenize_sentence(config):
    data = pd.read_csv(config.file_fn, sep='\t')
    data['sentence'].dropna(axis=0, inplace=True)
    sentences = data['sentence'].values
    labels = data['label'].apply(lambda x: eval(x))

    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    sentence_tok = []
    for sentence in sentences:
        if len(str(sentence)) < 512:
            sentence_tok.append(tokenizer.tokenize(str(sentence)))
        else:
            sentence_tok.append(tokenizer.tokenize(
                sentence, add_special_tokens=True, max_length=len(sentence), truncation=True))

    return sentence_tok, labels


def BIO_tagging(sentence_tok, ne):
    result = ['O' for x in range(len(sentence_tok))]
    ne_no = len(ne.keys())
    if ne_no > 0:
        for idx in range(1, ne_no+1):
            ne_dict = ne[idx]
            isbegin = True
            for word_idx, word in enumerate(sentence_tok):

                if '##' in word:
                    word = word.replace('##', '')

                if word in ne_dict['form']:
                    if isbegin:
                        result[word_idx] = str(
                            ne_dict['label'][:ne_dict['label'].find('_')])+'_B'
                        isbegin = False
                        continue

                    elif isbegin == False & (('_B' in result[word_idx-1]) or ('_I' in result[word_idx-1])):
                        result[word_idx] = str(
                            ne_dict['label'][:ne_dict['label'].find('_')])+'_I'
                        continue
    return result


if __name__ == '__main__':
    config = argument_parser()
    sentence_tok, labels = tokenize_sentence(config)
    for sentence, ne in zip(sentence_tok, labels):
        buf = []
        sentence_bio = BIO_tagging(sentence, ne)
        buf += [sentence, sentence_bio]
        sys.stdout.write(str(buf[0])+'\t')
        sys.stdout.write(str(buf[1])+'\n')
