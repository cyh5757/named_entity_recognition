/content/drive/MyDrive/tsv/19NX.tsv

from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset

def argument_parser():

    p = argparse.ArgumentParser()

    p.add_argument('--file_fn', required=True)
    p.add_argument('--split', default='StratifiedShuffleSplit')
    config = p.parse_args()

    return config


def get_data_loaders(  # 처음엔 데이터로더를 반환할 예정이었어서 함수 이름이 이렇게 됨
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size: int = 32,
    return_loader: bool = True,  # 데이터 로더로 반환할지의 여부
    use_imbalanced: bool = True,  # 데이터 로더로 반환할 때, ImbalancedDatasetSampler를 사용하는지의 여부
    device="cpu",
):
    pass

def Stratified_data(config):
    data = pd.read_csv(config.file_fn, sep='\t')
    split=StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    for train_idx, test_idx in split.split(data)




if __name__ == '__main__':
    config = argument_parser()
    sentence_tok, labels = tokenize_sentence(config)
    for sentence, ne in zip(sentence_tok, labels):
        buf = []
        sentence_bio = BIO_tagging(sentence, ne)
        buf += [sentence, sentence_bio]
        sys.stdout.write(str(buf[0])+'\t')
        sys.stdout.write(str(buf[1])+'\n')
