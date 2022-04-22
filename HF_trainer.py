import argparse
import random
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

import torch

# how to using huggingface tokenizer(bert) 
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification

# using monologg/kobert tokenizer and model
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel

# huggingface Trainer
from transformers import Trainer
from transformers import TrainingArguments

#Cutomizer encoder
from bert_dataset import TokenCollator
from bert_dataset import TokenDataset


from transformers.data.data_collator import DataCollatorForTokenClassification
from datasets import load_metric



def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    # Recommended model list:
    # - kykim/bert-kor-base //예시
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

def get_datasets(fn, valid_ratio=.2):
    #Get list of labels and list of texts
    labels, texts = pd.read_csv(config.data_fn)

    #Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}

    for i, label in enumerate(unique_labels):
        label_to_index[label]= i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    #Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]

    idx = int(len(texts)*(1-valid_ratio))

    train_dataset = TokenDataset(texts[:idx], labels[:idx])
    test_dataset = TokenDataset(texts[idx:], labels[idx:])
     
    return train_dataset, valid_dataset, index_to_label

def get_pretrained_model(num_labels: int):

    # tokenizer_load
    tokenizer_load = KoBertTokenizer if config.use_monologg else Autotokenizer
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

def main(config):

    # load model, tokenizer    
    model, tokenizer = get_pretrained_model(32)

    #Get datasets and index to label map.
    train_dataset, valid_dataset, index_to_label = get_datasets(
        config.train_fn,
        valid_ratio=config.valid_ratio
    )
    
    print(
        '|train| = ', len(train_dataset),
        '|valid| = ', len(valid_dataset)
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )



metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [list_of_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [list_of_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


batchify = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)
    # training_aregs = TrainingArguments(
    #     output_dir='./.checkpoints',
    #     num_train_epochs=config.n_epochs,
    #     per_device_train_batch_size=config.batch_size_per_device,
    #     per_device_eval_batch_size=config.batch_size_per_device,
    #     warmup_steps=n_warmup_steps,
    #     weight_decay=0.01,
    #     fp16=True,
    #     evaluation_strategy='epoch',
    #     save_strategy='epoch',
    #     logging_steps=n_total_iterations // 100,
    #     save_steps=n_total_iterations // config.n_epochs,
    #     load_best_model_at_end=True,
    # )
    # #

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=TokenCollator(tokenizer,
    #                                 config.max_length,
    #                                 with_text=False),
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     compute_metrics=compute_metrics
    # )

    # trainer.train()

training_args = TrainingArguments(
        output_dir='./results',     
        evaluation_strategy="epoch",
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        adam_beta1=.9,
        adam_beta2=.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.,
        num_train_epochs=2,
        lr_scheduler_type="linear",
        warmup_steps=100,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=100,
        save_strategy="epoch",
        seed=42,
        dataloader_drop_last=False,
        dataloader_num_workers=2
)

trainer = Trainer(
    args=training_args,
    data_collator=batchify,
    model=model,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics
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