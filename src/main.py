from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import torch.optim as optim
import os
import time
import torch
import argparse
import data_process, capsule_nn, tool


def setting(model_type, ckpt_dir, mode, bert_embedding_frozen):
    # Load data dict.
    print("Reading dataset...")
    data_path = f"../data/{mode}"
    data = data_process.read_datasets(data_path, model_type)

    train_class_num = len(data['train_class_dict'])
    test_class_num = len(data['test_class_dict'])

    # label_data is used for zero shot evalutation, so keep this separately.
    label_data = {
        'train_label_idxs': data['train_label_idxs'],
        'test_label_idxs': data['test_label_idxs'],
        'train_label_lens': data['train_label_lens'],
        'test_label_lens': data['test_label_lens']
    }

    # Set basic configs for training.
    config = {'keep_prob': 0.8,
              'hidden_size': data['word_emb_size'] if model_type == 'bert_capsnet' else 768,
              'batch_size': 16,
              'vocab_size': data['vocab_size'],
              'epoch_num': 200,
              'seq_len': data['max_len'],
              'pad_id': data['pad_id'],
              'train_class_num': train_class_num,
              'test_class_num': test_class_num,
              'word_emb_size': data['word_emb_size'],
              'd_a': 20,
              'd_m': 256,
              'caps_prop': 10,
              'r': 3,
              'iter_num': 3,
              'alpha': 0.0001,
              'learning_rate': 0.0001,
              'sim_scale': 4,
              'num_layers': 2,
              'w2v': data['w2v'],
              'ckpt_dir': ckpt_dir,
              'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
              'model_type': model_type,
              'mode': mode,
              'bert_embedding_frozen': bert_embedding_frozen
    }

    # Initialize dataloaders.
    train_sampler = RandomSampler(data['train_set'], replacement=True, num_samples=data['train_set'].__len__())
    train_loader = DataLoader(data['train_set'], batch_size=config['batch_size'], sampler=train_sampler)
    test_loader = DataLoader(data['test_set'], batch_size=config['batch_size'], shuffle=True)

    return config, label_data, train_loader, test_loader


def train(config, label_data, train_loader, test_loader):
    # Total training/testing time.
    total_train_time = 0.0
    total_test_time = 0.0

    # Initialize the model and optimizer.
    model = capsule_nn.CapsuleNetwork(config).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # If the saving directory does not exist, make one.
    if not os.path.exists(config['ckpt_dir']):
        print("Making check point directory...")
        os.mkdir(config['ckpt_dir'])

    # Training starts.
    print("Training starts.")
    best_test_acc = 0.0
    for epoch in range(1, config['epoch_num']+1):
        model.train()
        y_list = []
        pred_list = []
        loss_val = None

        # One batch.
        start_time = time.time()
        for batch in tqdm(train_loader):
            batch_x, batch_y, batch_lens, batch_one_hot_label = batch

            attentions, output_logits, prediction_vecs, _ = model(batch_x, batch_lens, is_train=True)
            loss_val = model.get_loss(batch_one_hot_label, output_logits, attentions)
        
            optimizer.zero_grad()
            loss_val.backward()

            optimizer.step()

            batch_pred = torch.argmax(output_logits, 1)
            y_list += batch_y.tolist()
            pred_list += batch_pred.tolist()

        train_time = time.time() - start_time
        total_train_time += train_time

        # Calculate accuracy and f1 score of one epoch.
        acc = accuracy_score(y_list, pred_list)
        f1 = f1_score(y_list, pred_list, average='weighted')

        print(f"################### Epoch: {epoch} ###################")
        print(f"Train loss: {loss_val.item()}")
        print(f"Train Acc: {round(acc, 4)} || Train F1: {round(f1, 4)}")
        print(f"Train time: {round(train_time, 4)}")

        # Execute evaluation depending on each task.
        cur_test_acc = 0.0
        cur_test_f1 = 0.0
        test_time = 0.0
        if config['mode'] == 'zero_shot':
            cur_test_acc, cur_test_f1, test_time = evaluate_zero_shot(test_loader, label_data, config, model)
        elif config['mode'] == 'seen_class':
            cur_test_acc, cur_test_f1, test_time = evaluate_seen_class(test_loader, model)

        # If f1 score has increased, save the model.
        if cur_test_acc > best_test_acc:
            best_test_acc = cur_test_acc
            torch.save(model.state_dict(), f"{config['ckpt_dir']}/best_model.pth")
            print("************ Best model saved! ************")

        print("------------------------------------------------------")
        print(f"Best Test Acc: {round(best_test_acc, 4)}")
        print(f"Test Acc: {round(cur_test_acc, 4)} || Current Test F1: {round(cur_test_f1, 4)}")
        total_test_time += test_time
        print("Testing time", round(test_time, 4))

    print(f"Overall training time: {total_train_time}")
    print(f"Overall testing time: {total_test_time}")
    
    
def evaluate_seen_class(test_loader, model):
    # This is for seen class classification which we usually conduct.

    model.eval()
    y_list = []
    pred_list = []

    # Evaluation starts.
    with torch.no_grad():
        start_time = time.time()

        # One batch.
        for batch in tqdm(test_loader):
            batch_x, batch_y, batch_lens, batch_one_hot_label = batch

            attentions, output_logits, prediction_vecs, _ = model(batch_x, batch_lens, is_train=False)

            y_list += batch_y.tolist()
            pred_list += torch.argmax(output_logits, 1).tolist()

        test_time = time.time() - start_time
        acc = accuracy_score(y_list, pred_list)
        f1 = f1_score(y_list, pred_list, average='weighted')
        
    return acc, f1, test_time
    

def evaluate_zero_shot(test_loader, label_data, config, model):
    model.eval()
    y_list = []
    pred_list = []

    # Label indices(tokens) tensors and their sequence lengths.
    train_label_idxs = label_data['train_label_idxs'].to(config['device'])
    test_label_idxs = label_data['test_label_idxs'].to(config['device'])
    train_label_lens = label_data['train_label_lens']
    test_label_lens = label_data['test_label_lens']

    # Use the model's embedding layer to calculate each label embedding
    embedding = model.get_embedding()
    train_class_embeddings = embedding(train_label_idxs).cpu().detach().numpy() # (num_intents, max_len, D_W)
    test_class_embeddings = embedding(test_label_idxs).cpu().detach().numpy()

    # Combine word embeddings of each label to get label embedding.
    train_class_embeddings = tool.get_label_embedding(train_class_embeddings, train_label_lens)
    test_class_embeddings = tool.get_label_embedding(test_class_embeddings, test_label_lens)
    
    # Get unseen and seen categories similarity
    sim = torch.from_numpy(
        tool.get_sim(train_class_embeddings, test_class_embeddings, config['sim_scale'])
    ).to(config['device'])  # (L, K)

    # Evaluation starts.
    with torch.no_grad():

        # One batch.
        start_time = time.time()
        for batch in tqdm(test_loader):
            batch_x, batch_y, batch_lens, batch_one_hot_label = batch

            # attention: A (B, R, L), seen_logits: logits from v (B, num_properties), seen_prediction: p (B, R, K, num_properties), seen_c: c (B, R, K)
            attentions, seen_logits, seen_prediction, seen_c = model(batch_x, batch_lens, is_train=False)

            # Get vote vector using similarities.
            sim = torch.unsqueeze(sim, 0) # (1, L, K)
            sim = sim.repeat([seen_prediction.shape[1], 1, 1]) # (R, L, K)
            sim = torch.unsqueeze(sim, 0) # (1, R, L, K)
            sim = sim.repeat([seen_prediction.shape[0], 1, 1, 1]) # (B, R, L, K)
            seen_c = seen_c.unsqueeze(-1)
            seen_c = seen_c.repeat([1, 1, 1, config['caps_prop']]) # (B, R, K, num_properties)
            vote_vec = seen_prediction * seen_c # (B, R, K, num_properties)

            # Compute unseen prediction vector.
            unseen_prediction = torch.matmul(sim, vote_vec) # (B, R, L, num_properties)

            # v: (B, L, num_properties), c: (B, R, L)
            logit_shape = [unseen_prediction.shape[0], config['r'], config['uc_num']]
            unseen_v, _, unseen_c = model.routing(unseen_prediction, logit_shape, num_dims=4, is_train=False)

            unseen_logits = torch.norm(unseen_v, dim=-1) # (B, L)

            y_list += batch_y.tolist()
            pred_list += torch.argmax(unseen_logits, 1).tolist()

        test_time = time.time() - start_time
        acc = accuracy_score(y_list, pred_list)
        f1 = f1_score(y_list, pred_list, average='weighted')

    return acc, f1, test_time


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Argument parser for various parameters.")
    parser.add_argument('--model_type', type=str, required=True, help="The model type for training & testing")
    parser.add_argument('--mode', type=str, required=True, help="seen class or zero shot?")
    parser.add_argument('--bert_embedding_frozen', type=bool, default=False, help="Do you want to freeze BERT's embedding layer or not?")

    args = parser.parse_args()

    assert args.model_type == ('bert_capsnet' or 'basic_capsnet' or 'w2v_capsnet'), "Please specify correct model type."
    assert args.mode == ('seen_class' or 'zero_shot'), "Please specify correct mode."

    ckpt_dir = f"../saved_models/{args.model_type}/{args.mode}"

    config, label_data, train_loader, test_loader,  = setting(args.model_type, ckpt_dir, args.mode, args.bert_embedding_frozen)
    train(config, label_data, train_loader, test_loader)
