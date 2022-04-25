import torch
from torch.utils.data import Dataset


class TokenCollator():
    
    def __init__(self, tokenizer, max_length, labels_map, with_text=True) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels_map = labels_map
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        
        encoded = self.tokenizer(texts, 
                                   add_special_tokens=True,
                                   padding=True,
                                   truncation=True,
                                   return_tensors='pt',
                                   return_attention_mask=True,
                                   return_length=True)

        label_ids = []
        for text, label in zip(texts, labels):
            text_token= self.tokenizer.tokenize(text, 
                                                add_special_tokens=True, 
                                                max_length=encoded['length'][0],
                                                truncation=True,
                                                padding='max_length')
            label_sequence = self.__BIO_tagging(text_token, label)
            label_id = [self.labels_map[key] if key in self.labels_map.keys() else -100 for key in label_sequence]  ### KLUE only.  
            label_ids.append(label_id)

        return_value = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': torch.tensor(label_ids, dtype=torch.long),
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value

    def __tokenize_sentence(self, sentences, length):
        sentence_tok = []
        for sentence in sentences:
            sentence_tok.append(self.tokenizer.tokenize(sentence,
                                                        add_special_tokens=True,
                                                        max_length=length,
                                                        truncation=True,
                                                        padding='max_length'))
        return sentence_tok

    def __BIO_tagging(sentence_tok, ne):

        special_token = ['[CLS]', '[PAD]', '[SEP]', '[UNK]']
        result = []
        temp = [sentence_tok[x] if sentence_tok[x] in special_token else 'O' for x in range(len(sentence_tok))]
        ne_no = len(ne.keys())
        if ne_no > 0:

            for idx in range(1, ne_no+1):
                ne_dict = ne[idx]
                label_length = len(ne_dict['form'].replace(' ', ''))
                isbegin = True
                for word_idx, word in enumerate(sentence_tok):
                    if label_length == 0:
                        break
                if ('##' in word) or ('▁' in word):
                    if word == '▁':
                        continue
                    word = word.replace('##', '')
                    word = word.replace('▁', '')
                    
                    
                    if word in ne_dict['form']:
                        if isbegin:
                            temp[word_idx] = str(ne_dict['label'][:2])+'_B'
                            isbegin = False
                            label_length = label_length - len(word)
                            continue
                        elif (isbegin == False) & (label_length > 0) & (('_B' in temp[word_idx-1]) or ('_I' in temp[word_idx-1])):
                            temp[word_idx] = str(ne_dict['label'][:2])+'_I'
                            label_length = label_length - len(word)
                            continue

        return temp


class TokenDataset(Dataset):

    def __init__(self, texts, labels) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        }