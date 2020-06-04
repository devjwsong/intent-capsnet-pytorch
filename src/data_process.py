from kobert_transformers import get_distilkobert_model
from kobert_transformers import get_tokenizer
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import torch
import tool


def process_label(classes, tokenizer, w2v, pad_id):
    # Tokenize each label and convert each word into idxs in vocab.

    label_idxs = []
    lens = []
    max_len = 0

    for class_name in classes:
        # bert_capsnet & basic_capsent use KoBERT Tokenizer and w2v_capsnet uses whitespace tokenizer.
        if tokenizer is None:
            label = class_name.split(' ')
            label = [w2v.vocab[w].index for w in label if w in w2v.vocab]
        else:
            label = tokenizer.tokenize(class_name)
            label = tokenizer.convert_tokens_to_ids(label)

        # Keep label lengths for later.
        lens.append(len(label))
        max_len = max(max_len, len(label))
        label_idxs.append(label)

    # To make each label into tensor, add paddings.
    for i, class_name in enumerate(classes):
        label = label_idxs[i]
        if len(label) < max_len:
            label += ([pad_id] * (max_len - len(label)))

        label_idxs[i] = label

    return torch.LongTensor(label_idxs), lens  # (num_intents, max_len)


def processing(lines, tokenizer, w2v, max_len, pad_id):
    # Read dataset, tokenize each sentences, and extract classes in it.

    not_padded_x = []
    x = []
    y = []
    lens = []
    class_dict = {}

    # Process raw data
    for i, line in enumerate(tqdm(lines)):
        arr = line.strip().split('\t')

        # Since each word in labels is separated in _, we should split them first.
        class_words = [w for w in arr[0].split('_')]
        class_name = ' '.join(class_words)

        # bert_capsnet & basic_capsent use KoBERT Tokenizer and w2v_capsnet uses whitespace tokenizer.
        if tokenizer is None:
            x_arr_tok = arr[1].split(' ')
            x_arr = [w2v.vocab[w].index for w in x_arr_tok if w in w2v.vocab]
        else:
            x_arr = tokenizer.tokenize('[CLS] ' + arr[1] + ' [SEP]')
            x_arr = tokenizer.convert_tokens_to_ids(x_arr)

        x_len = len(x_arr)

        if x_len <= 1:
            continue

        not_padded_x.append(x_arr)

        # Note class dictionary.
        if class_name not in class_dict:
            class_dict[class_name] = len(class_dict)
        y.append(class_dict[class_name])
        lens.append(x_len)

    # Add paddings
    for i, text in enumerate(not_padded_x):
        if max_len < lens[i]:
            x.append(not_padded_x[i][0:max_len])
            lens[i] = max_len
        else:
            temp = not_padded_x[i] + [pad_id] * (max_len - lens[i])
            x.append(temp)

    return torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(lens), class_dict


def encode_label(data_y):
    # Make one-hot encoded label vectors for calculating margin losses.

    sample_num = data_y.shape[0]
    labels = np.unique(data_y)
    class_num = labels.shape[0]
    labels = range(class_num)

    # Get one-hot-encoded label tensors
    vecs = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        vecs[data_y == labels[i], i] = 1

    return torch.LongTensor(vecs)


class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, w2v, max_len, pad_id):
        if w2v is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                lines = f.readlines()

        print(f"Processing {file_path}...")
        self.x, self.y, self.lens, self.class_dict = processing(lines, tokenizer, w2v, max_len, pad_id)

        self.one_hot_label = encode_label(self.y)

        classes = [k for k, v in self.class_dict.items()]
        self.label_idxs, self.label_lens = process_label(classes, tokenizer, w2v, pad_id)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.lens[idx], self.one_hot_label[idx]


def load_w2v(file_path):
    # Load w2v model in case of using w2v_capsnet

    w2v = KeyedVectors.load_word2vec_format(
        file_path, binary=False, encoding='ISO-8859-1'
    )

    return w2v


def read_datasets(data_path, model_type):
    # Read datasets and give the data dictionary containing essential data objects for training.

    data = {}
    
    train_data_path = f"{data_path}/train.txt"
    test_data_path = f"{data_path}/test.txt"

    # Load tokenizer and set max length of sentences.
    tokenizer = None
    w2v = None
    bert_config = None
    max_len = 50
    if model_type == 'bert_capsnet' or model_type == 'basic_capsnet':
        print("Loading KoBertTokenizer...")
        tokenizer = get_tokenizer()
        if model_type == 'bert_capsnet':
            bert_config = get_distilkobert_model().config
            max_len = bert_config.max_position_embeddings

    else:
        w2v_path = '../data/cc.ko.300.vec'
        w2v = load_w2v(w2v_path)

    # Keep the index of padding.
    if tokenizer is None:
        pad_id = 0
    else:
        pad_id = tokenizer.token2idx['[PAD]']

    # Preprocess train/test data
    print("Preprocessing train/test data...")
    train_set = CustomDataset(train_data_path, tokenizer, w2v, max_len, pad_id)
    test_set = CustomDataset(test_data_path, tokenizer, w2v, max_len, pad_id)
    data['train_set'] = train_set
    data['test_set'] = test_set

    # These are train/test custom dataset object.
    data['train_class_dict'] = train_set.class_dict
    data['test_class_dict'] = test_set.class_dict

    # These are tensors containing tokenized label words.
    data['train_label_idxs'] = train_set.label_idxs
    data['test_label_idxs'] = test_set.label_idxs

    # These are lengths of labels.
    data['train_label_lens'] = train_set.label_lens
    data['test_label_lens'] = test_set.label_lens

    data['max_len'] = max_len
    data['pad_id'] = pad_id

    # Depending the model type, vocab_size, word_emb_size, and w2v object can be different.
    if model_type == 'bert_capsnet':
        data['vocab_size'] = len(tokenizer.token2idx)
        data['word_emb_size'] = bert_config.dim
        data['embedding'] = None
    elif model_type == 'basic_capsnet':
        data['vocab_size'] = len(tokenizer.token2idx)
        data['word_emb_size'] = 300
        data['embedding'] = None
    elif model_type == 'w2v_capsnet':
        w2v_shape = w2v.wv.vectors.shape
        data['vocab_size'] = w2v_shape[0]
        data['word_emb_size'] = w2v_shape[1]
        data['embedding'] = tool.norm_matrix(w2v.syn0)

    return data
