import argparse
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

import torch

# how to using huggingface tokenizer(bert) 
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

# using monologg/kobert tokenizer and model
from kobert_transformers.tokenization_kobert import KoBertTokenizer
from transformers import BertModel

# huggingface Trainer
from transformers import Trainer
from transformers import TrainingArguments

#Cutomizer encoder
from bert_dataset import TokenCollator
from bert_dataset import TokenDataset


from transformers.data.data_collator import DataCollatorForTokenClassification
from datasets import load_metric


#agrgs 사용
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--file_fn', required=True)
    p.add_argument('--model_fn', required=True)

    # Recommended model list:
    # - monologg/kobert
    # - klue/bert-base
    

    p.add_argument('--pretrained_model_name', type=str, default='klue/bert-base')
    p.add_argument('--use_monologg', action='store_true')

    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--warmup_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config


def get_label_dict():
    labels = ["PS", "FD", "TR", "AF", "OG", "LC", "CV", "DT", "TI", "QT", "EV", "AM", "PT", "MT", "TM"]
    
    BIO_labels = ['O']
    for label in labels:
        BIO_labels.append(label+'_B')
        BIO_labels.append(label+'_I')

    label_to_index = {label:index for index, label in enumerate(BIO_labels)}
    index_to_label = {index:label for index, label in enumerate(BIO_labels)}

    return label_to_index, index_to_label



def get_datasets(fn, valid_ratio=.2):
    #Get list of labels and list of texts
    data = pd.read_csv(fn,sep='\t')
    data['sentence'].dropna(axis=0, inplace=True)

    # Shuffle before split into train and validation set.
    shuffled = data.index.values
    random.shuffle(shuffled)
    sentences = data.loc[shuffled]['sentence'].values
    labels = data.loc[shuffled]['label'].apply(lambda x: eval(x)).values
    idx = int(len(sentences) * (1 - valid_ratio))

    train_dataset = TokenDataset(sentences[:idx], labels[:idx])
    valid_dataset = TokenDataset(sentences[idx:], labels[idx:])

    return train_dataset, valid_dataset

def get_pretrained_model(num_labels: int):

    # tokenizer_load
    tokenizer_load = KoBertTokenizer if config.use_monologg else AutoTokenizer
    tokenizer = tokenizer_load.from_pretrained(
        config.pretrained_model_name
    )
    # model_load
    model_loader = BertModel if config.use_monologg else AutoModelForTokenClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        num_labels = num_labels
    )
    return model, tokenizer

def compute_metrics(pred):
    metric = load_metric('seqeval')

    labels = pred.label_ids
    predictions = pred.predictions.argmax(2)

    true_predictions = [[p for p, l in zip(prediction, label) if l >= 0] for prediction, label in zip(predictions, labels)]
    true_labels = [[l for p, l in zip(prediction, label) if l >= 0] for prediction, label in zip(predictions, labels)]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)

    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }

    return flattened_results


def main(config):

    
    label_to_index, index_to_label = get_label_dict()

    #Get datasets and index to label map.
    train_dataset, valid_dataset = get_datasets(
        config.file_fn,
        valid_ratio=config.valid_ratio
    )
    
    # load model, tokenizer    
    model, tokenizer = get_pretrained_model(len(label_to_index))

    print(
        model,
        tokenizer
    )
    print(
        '|train| = ', len(train_dataset),
        '|valid| = ', len(valid_dataset)
    )







    # * torch.cuda.device_count()
    total_batch_size = config.batch_size_per_device 
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )


    
    training_args = TrainingArguments(
        output_dir='./.checkpoint',
        num_train_epochs=7,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        # fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=n_total_iterations, #// 100,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )




    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TokenCollator(tokenizer=tokenizer,
                                  max_length=config.max_length,
                                  labels_map=label_to_index,
                                  with_text=False),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': trainer.model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)