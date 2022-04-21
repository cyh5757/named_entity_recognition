def get_datasets(fn, valid_ratio=.2):
     # Get list of labels and list of texts.
    labels, texts = read_text(fn)

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    train_dataset = TextClassificationDataset(texts[:idx], labels[:idx])
    valid_dataset = TextClassificationDataset(texts[idx:], labels[idx:])

    return train_dataset, valid_dataset, index_to_label