from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import torch.optim as optim
import os
import time
import torch
import argparse
import data_process, capsule_nn, tool
import random


def setting(args):
    # Load data dict.
    print("Reading dataset...")
    from_data_dir = f"{args.data_dir}/{args.raw_dir}"
    to_data_dir = f"{args.data_dir}/{args.mode}"

    if not os.path.isdir(to_data_dir):
        os.makedirs(to_data_dir)
    train_set, valid_set, args = data_process.read_datasets(from_data_dir, to_data_dir, args)

    args.train_num_classes = len(train_set.class_dict)
    args.valid_num_classes = len(valid_set.class_dict)

    # label_data is used for zero shot evalutation, so keep this separately.
    label_data = {
        'train_label_idxs': train_set.label_idxs,
        'valid_label_idxs': valid_set.label_idxs,
        'train_label_lens': train_set.label_lens,
        'valid_label_lens': valid_set.label_lens,
    }
    
    args.hidden_size = args.word_emb_size if args.model_type == 'bert_capsnet' else 768
    args.device = torch.device(f'cuda: {args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize dataloaders.
    random.seed(args.seed)
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=len(train_set))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)

    return args, label_data, train_loader, valid_loader


def train(args, label_data, train_loader, valid_loader):
    # Total training/evaluating time.
    total_train_time = 0.0
    total_valid_time = 0.0

    # Initialize the model and optimizer.
    print("Loading model & optimizer...")
    model = capsule_nn.CapsuleNetwork(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # If the saving directory does not exist, make one.
    if not os.path.exists(args.ckpt_dir):
        print("Making check point directory...")
        os.makedirs(args.ckpt_dir)

    # Training starts.
    print("Training starts.")
    best_valid_acc = 0.0
    for epoch in range(1, args.num_epochs+1):
        model.train()
        y_list = []
        pred_list = []
        loss_val = None

        # One batch.
        start_time = time.time()
        for batch in tqdm(train_loader):
            batch_x, batch_y, batch_lens, batch_one_hot_label = batch
            batch_x, batch_y, batch_lens, batch_one_hot_label = sort_batch(batch_x, batch_y, batch_lens, batch_one_hot_label)

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
        cur_valid_acc = 0.0
        cur_valid_f1 = 0.0
        valid_time = 0.0
        if args.mode == 'zero_shot':
            cur_valid_acc, cur_valid_f1, valid_time = evaluate_zero_shot(args, valid_loader, label_data, model)
        elif args.mode == 'seen_class':
            cur_valid_acc, cur_valid_f1, valid_time = evaluate_seen_class(valid_loader, model)

        # If f1 score has increased, save the model.
        if cur_valid_acc > best_valid_acc:
            best_valid_acc = cur_valid_acc
            torch.save(model.state_dict(), f"{args.ckpt_dir}/best_model.ckpt")
            print("************ Best model saved! ************")

        print("------------------------------------------------------")
        print(f"Best Validation Acc: {round(best_valid_acc, 4)}")
        print(f"Validation Acc: {round(cur_valid_acc, 4)} || Current Validation F1: {round(cur_valid_f1, 4)}")
        total_valid_time += valid_time
        print("Validation time", round(valid_time, 4))

    print(f"Overall training time: {total_train_time}")
    print(f"Overall validation time: {total_valid_time}")
    
    
def evaluate_seen_class(valid_loader, model):
    # This is for seen class classification which we usually conduct.
    model.eval()
    y_list = []
    pred_list = []

    # Evaluation starts.
    with torch.no_grad():
        start_time = time.time()

        # One batch.
        for batch in tqdm(valid_loader):
            batch_x, batch_y, batch_lens, batch_one_hot_label = batch
            batch_x, batch_y, batch_lens, batch_one_hot_label = sort_batch(batch_x, batch_y, batch_lens,
                                                                           batch_one_hot_label)

            attentions, output_logits, prediction_vecs, _ = model(batch_x, batch_lens, is_train=False)

            y_list += batch_y.tolist()
            pred_list += torch.argmax(output_logits, 1).tolist()

        valid_time = time.time() - start_time
        acc = accuracy_score(y_list, pred_list)
        f1 = f1_score(y_list, pred_list, average='weighted')
        
    return acc, f1, valid_time
    

def evaluate_zero_shot(args, valid_loader, label_data, model):
    model.eval()
    y_list = []
    pred_list = []

    # Label indices(tokens) tensors and their sequence lengths.
    train_label_idxs = label_data['train_label_idxs'].to(args.device)
    valid_label_idxs = label_data['valid_label_idxs'].to(args.device)
    train_label_lens = label_data['train_label_lens']
    valid_label_lens = label_data['valid_label_lens']

    # Use the model's embedding layer to calculate each label embedding
    embedding = model.get_embedding()
    train_class_embeddings = embedding(train_label_idxs).cpu().detach().numpy() # (num_intents, max_len, D_W)
    valid_class_embeddings = embedding(valid_label_idxs).cpu().detach().numpy()

    # Combine word embeddings of each label to get label embedding.
    train_class_embeddings = tool.get_label_embedding(train_class_embeddings, train_label_lens)
    valid_class_embeddings = tool.get_label_embedding(valid_class_embeddings, valid_label_lens)
    
    # Get unseen and seen categories similarity
    sim_ori = torch.from_numpy(
        tool.get_sim(train_class_embeddings, valid_class_embeddings, args.sim_scale)
    ).to(args.device)  # (L, K)

    # Evaluation starts.
    with torch.no_grad():
        # One batch.
        start_time = time.time()
        for batch in tqdm(valid_loader):
            batch_x, batch_y, batch_lens, batch_one_hot_label = batch
            batch_x, batch_y, batch_lens, batch_one_hot_label = sort_batch(batch_x, batch_y, batch_lens,
                                                                           batch_one_hot_label)

            # attention: A (B, R, L), seen_logits: logits from v (B, num_properties), seen_prediction: p (B, R, K, num_properties), seen_c: c (B, R, K)
            attentions, seen_logits, seen_prediction, seen_c = model(batch_x, batch_lens, is_train=False)

            # Get vote vector using similarities.
            sim = torch.unsqueeze(sim_ori, 0) # (1, L, K)
            sim = sim.repeat([seen_prediction.shape[1], 1, 1]) # (R, L, K)
            sim = torch.unsqueeze(sim, 0) # (1, R, L, K)
            sim = sim.repeat([seen_prediction.shape[0], 1, 1, 1]) # (B, R, L, K)
            seen_c = seen_c.unsqueeze(-1)
            seen_c = seen_c.repeat([1, 1, 1, args.num_props]) # (B, R, K, num_properties)
            vote_vec = seen_prediction * seen_c # (B, R, K, num_properties)

            # Compute unseen prediction vector.
            unseen_prediction = torch.matmul(sim, vote_vec) # (B, R, L, num_properties)

            # v: (B, L, num_properties), c: (B, R, L)
            logit_shape = [unseen_prediction.shape[0], args.r, args.valid_num_classes]
            unseen_v, _, unseen_c = model.routing(unseen_prediction, logit_shape, num_dims=4, is_train=False)

            unseen_logits = torch.norm(unseen_v, dim=-1) # (B, L)

            y_list += batch_y.tolist()
            pred_list += torch.argmax(unseen_logits, 1).tolist()

        valid_time = time.time() - start_time
        acc = accuracy_score(y_list, pred_list)
        f1 = f1_score(y_list, pred_list, average='weighted')

    return acc, f1, valid_time


def sort_batch(batch_x, batch_y, batch_lens, batch_one_hot_label):
    batch_lens_sorted, sorted_idx = batch_lens.sort(0, descending=True)
    batch_x_sorted = batch_x[sorted_idx]
    batch_y_sorted = batch_y[sorted_idx]
    batch_one_hot_label_sorted = batch_one_hot_label[sorted_idx]

    return batch_x_sorted, batch_y_sorted, batch_lens_sorted, batch_one_hot_label_sorted


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Argument parser for various parameters.")
    parser.add_argument('--seed', type=int, required=True, default=0, help="The random seed.")
    parser.add_argument('--batch_size', type=int, required=True, default=1, help="The batch size.")
    parser.add_argument('--learning_rate', type=float, required=True, default=1e-4, help="The learning rate.")
    parser.add_argument('--num_epochs', type=int, required=True, default=1, help="The total number of epochs.")
    parser.add_argument('--max_len', type=int, required=True, default=128, help="The maximum input length.")
    parser.add_argument('--dropout', type=float, required=True, default=0.0, help="The dropout rate.")
    parser.add_argument('--d_a', type=int, required=True, default=20, help="The dimension size of internal vector during self-attention.")
    parser.add_argument('--num_props', type=int, required=True, default=10, help="The number of properties in each capsule.")
    parser.add_argument('--r', type=int, required=True, default=3, help="The number of semantic features.")
    parser.add_argument('--num_iters', type=int, required=True, default=1, help="The number of iterations for the dynamic routing algorithm.")
    parser.add_argument('--alpha', type=float, required=True, default=1e-4, help="The coefficient value for encouraging the discrepancies among different attention heads in the loss function.")
    parser.add_argument('--sim_scale', type=int, required=True, default=1, help="The scaling factor for intent similarity.")
    parser.add_argument('--num_layers', type=int, default=1, help="The number of layers for an LSTM encoder.")
    parser.add_argument('--ckpt_dir', type=str, default="saved_models", help="The directory for trained ckpts.")
    parser.add_argument('--data_dir', type=str, default="data", help="The directory for data.")
    parser.add_argument('--raw_dir', type=str, default="raw", help="The directory for raw data.")
    parser.add_argument('--train_frac', type=float, default=0.8, help="The ratio of the conversations to be included in the train set.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The train data file name's prefix.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The validation data file name's prefix.")
    parser.add_argument('--model_type', type=str, required=True, help="The model type for training & evaluation.")
    parser.add_argument('--mode', type=str, required=True, help="seen class or zero shot?")
    parser.add_argument('--bert_embedding_frozen', type=str, default="False", help="Do you want to freeze BERT's embedding layer or not?")
    parser.add_argument('--gpu', type=str, default="0", help="The index of gpu to use.")

    args = parser.parse_args()

    assert args.model_type == 'bert_capsnet' or args.model_type == 'basic_capsnet' or args.model_type == 'w2v_capsnet', "Please specify correct model type."
    assert args.mode == 'seen_class' or args.mode == 'zero_shot', "Please specify correct mode."
    assert args.bert_embedding_frozen == 'True' or args.bert_embedding_frozen == 'False', "Please specify bert_embedding_frozen argument to among two options: True/False."
    
    args.bert_embedding_frozen = True if args.bert_embedding_frozen == "True" else False
    args.ckpt_dir = f"{args.ckpt_dir}/{args.model_type}/{args.mode}"
    
    print(args)

    args, label_data, train_loader, valid_loader = setting(args)
    train(args, label_data, train_loader, valid_loader)
    
    print("GOOD BYE.")
