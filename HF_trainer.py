import argparse
import random

from sklearn.metrics import f1_score

import torch

# how to using huggingface tokenizer(roberta, bert) 
from transformers import Autotokenizer
# how to using huggingface token classification model
# using etc.
# config = AutoConfig.from_pretrained("bert-base-cased")
# model = AutoModelForTokenClassification.from_config(config)
from transformers import AutoModelForTokenClassification

# using monologg/kobert tokenizer and model
from tokenization_kobert import KoBertTokenizer
from transformers import BertModel
# huggingface Trainer
from transformers import Trainer
from transformers import TrainingArguments


from bert_dataset import TokenCollator
from bert_dataset import TokenDataset



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




def main(config):
    # Get pretrained tokenizer.
    tokenizer_load = KoBertTokenizer if config.use_monologg else Autotokenizer
    tokenizer = tokenizer_load.from_pretrained(
        config.pretrained_model_name
    )
    
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

    # model_load
    model_loader = BertModel if config.use_monologg else AutoModelForTokenClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        # label num
        num_labels = len(index_to_label)
    )

    training_aregs = TrainingArguments(
        output_dir='./.checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        return {
            'f1_score': f1_score(labels, preds)
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TokenCollator(tokenizer,
                                    config.max_length,
                                    with_text=False),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
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