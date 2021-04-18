from transformers import BertTokenizer, BertConfig
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import torch
import tool
import os
import urllib.request
import random


class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, w2v, max_len, pad_id):
        print(f"Processing {file_path}...")
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self.x, self.y, self.lens, self.class_dict = process_texts(lines, tokenizer, w2v, max_len, pad_id)

        self.one_hot_labels = encode_label(self.y)

        classes = [k for k, v in self.class_dict.items()]
        self.label_idxs, self.label_lens = process_label(classes, tokenizer, w2v, pad_id)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.lens[idx], self.one_hot_labels[idx]


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


def process_texts(lines, tokenizer, w2v, max_len, pad_id):
    # Read dataset, tokenize each sentences, and extract classes in it.
    x = []
    y = []
    lens = []
    class_dict = {}

    # Process raw data
    for i, line in enumerate(tqdm(lines)):
        label = line.strip().split('\t')[0]
        text = line.strip().split('\t')[1]

        # Split the labels first.
        class_words = []
        word = ""
        for c, char in enumerate(label):
            word += char
            if c == len(label)-1 or label[c+1].isupper():
                class_words.append(word.lower())
                word = ""
        class_name = ' '.join(class_words)

        # bert_capsnet & basic_capsent use BERT Tokenizer and w2v_capsnet uses whitespace tokenizer.
        if tokenizer is None:
            x_tokens = text.split(' ')
            x_token_ids = [w2v.vocab[w].index for w in x_tokens if w in w2v.vocab]
        else:
            x_tokens = tokenizer.tokenize('[CLS] ' + text + ' [SEP]')
            x_token_ids = tokenizer.convert_tokens_to_ids(x_tokens)

        seq_len = len(x_token_ids)

        if seq_len <= 1:
            continue
            
        # Padding or truncating
        if seq_len <= max_len:
            x_token_ids += [pad_id] * (max_len-seq_len)
        else:
            x_token_ids = x_token_ids[:max_len]
            
        x.append(x_token_ids)

        # Note class dictionary.
        if class_name not in class_dict:
            class_dict[class_name] = len(class_dict)
        y.append(class_dict[class_name])
        lens.append(seq_len)

    return torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(lens), class_dict


def encode_label(y):
    # Make one-hot encoded label vectors for calculating margin losses.

    num_samples = y.shape[0]
    labels = np.unique(y)
    num_classes = labels.shape[0]
    labels = range(num_classes)

    # Get one-hot-encoded label tensors
    vecs = np.zeros((num_samples, num_classes), dtype=np.float32)
    for i in range(num_classes):
        vecs[y == labels[i], i] = 1

    return torch.LongTensor(vecs)


def load_w2v(file_path):
    # Load w2v model in case of using w2v_capsnet
    w2v = KeyedVectors.load_word2vec_format(file_path, binary=True)

    return w2v


def split_data(from_data_dir, to_data_dir, args):
    train_lines = []
    valid_lines = []
    
    file_list = [file for file in os.listdir(from_data_dir) if file.endswith('.txt')]
    for file in tqdm(file_list):
        with open(f"{from_data_dir}/{file}", 'r') as f:
            lines = f.readlines()
        
        random.seed(args.seed)
        random.shuffle(lines)
        train_lines += lines[:int(len(lines)*args.train_frac)]
        valid_lines += lines[int(len(lines)*args.train_frac):]
        
    print(f"The size of train set: {len(train_lines)}")
    print(f"The size of valid set: {len(valid_lines)}")
    
    train_data_path = f"{to_data_dir}/{args.train_prefix}.txt"
    valid_data_path = f"{to_data_dir}/{args.valid_prefix}.txt"
    with open(train_data_path, 'w') as f:
        for line in tqdm(train_lines):
            f.write(f"{line.strip()}\n")
    with open(valid_data_path, 'w') as f:
        for line in tqdm(valid_lines):
            f.write(f"{line.strip()}\n")
            
    return train_data_path, valid_data_path


def read_datasets(from_data_dir, to_data_dir, args):
    # Read datasets and make the data dictionary containing essential data objects for training.
    print("Splitting raw data and saving into txt files...") 
    train_data_path, valid_data_path = split_data(from_data_dir, to_data_dir, args)

    # Setting configurations.
    tokenizer = None
    w2v = None
    bert_config = None
    if args.model_type == 'bert_capsnet' or args.model_type == 'basic_capsnet':
        print("Loading BertTokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if args.model_type == 'bert_capsnet':
            bert_config = BertConfig.from_pretrained("bert-base-uncased")
            args.max_len = min(args.max_len, bert_config.max_position_embeddings)
    else:
        w2v_path = f'{data_dir}/GoogleNews-vectors-negative300.bin'
        assert os.path.isfile(w2v_path), f"There is no Korean w2v file. Please download w2v file from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit, extract it and put it in {args.data_dir}."
  
        w2v = load_w2v(w2v_path)

    # Keep the index of padding.
    if tokenizer is None:
        args.pad_id = 0
    else:
        args.pad_id = tokenizer.get_vocab()['[PAD]']

    # Preprocess train/test data
    print("Preprocessing train/test data...")
    train_set = CustomDataset(train_data_path, tokenizer, w2v, args.max_len, args.pad_id)
    valid_set = CustomDataset(valid_data_path, tokenizer, w2v, args.max_len, args.pad_id)

    # Depending the model type, vocab_size, word_emb_size, and w2v object can be different.
    if args.model_type == 'bert_capsnet':
        args.vocab_size = len(tokenizer.get_vocab())
        args.word_emb_size = bert_config.hidden_size
        args.embedding = None
    elif args.model_type == 'basic_capsnet':
        args.vocab_size = len(tokenizer.get_vocab())
        args.word_emb_size = 300
        args.embedding = None
    elif args.model_type == 'w2v_capsnet':
        w2v_shape = w2v.wv.vectors.shape
        args.vocab_size = w2v_shape[0]
        args.word_emb_size = w2v_shape[1]
        args.embedding = tool.norm_matrix(w2v.syn0)

    return train_set, valid_set, args
