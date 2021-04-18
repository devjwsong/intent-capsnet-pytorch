from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import tool
import random


class CapsuleNetwork(nn.Module):
    def __init__(self, args):
        super(CapsuleNetwork, self).__init__()

        # Seed fixing
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

        self.args = args

        # The embedding layer and encoder can be different according to the model type.
        self.embedding = None
        self.encoder = None
        output_size = 0.0
        if self.args.model_type == 'bert_capsnet':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
            output_size = self.args.hidden_size

            # BERT's embedding layer might be frozen in some cases.
            if self.args.bert_embedding_frozen:
                for p in self.encoder.embeddings.parameters():
                    p.requires_grad = False
        else:
            self.encoder = nn.LSTM(self.args.word_emb_size, self.args.hidden_size,
                                                 self.args.num_layers, bidirectional=True)
            self.embedding = nn.Embedding(self.args.vocab_size, self.args.word_emb_size)
            output_size = self.args.hidden_size * 2

            # Load pre-treind w2v model.
            if self.args.model_type == 'w2v_capsnet':
                self.embedding = self.embedding.from_pretrained(torch.from_numpy(self.args.embedding))
                
        self.drop = nn.Dropout(self.args.dropout)

        # Parameters for self attention.
        self.ws1 = nn.Linear(output_size, self.args.d_a, bias=False)
        self.ws2 = nn.Linear(self.args.d_a, self.args.r, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Parameters for linear transformation before DetectCaps.
        self.capsule_weights = nn.Parameter(torch.zeros((self.args.r, output_size,
                                                         self.args.train_num_classes * self.args.num_props)))

        # Initialize parameters.
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ws1.weight)
        nn.init.xavier_uniform_(self.ws2.weight)
        nn.init.xavier_uniform_(self.capsule_weights)

        self.ws1.weight.requires_grad_(True)
        self.ws2.weight.requires_grad_(True)
        self.capsule_weights.requires_grad_(True)

    def get_embedding(self):
        if self.args.model_type == 'bert_capsnet':
            return self.encoder.embeddings
        else:
            return self.embedding

    def forward(self, input_x, lens, is_train=True):
        # SemanticCaps
        if self.args.model_type == 'bert_capsnet':
            # When using BERT, make attention mask and conduct encoder process.
            attention_mask = (input_x != self.args.pad_id).float()
            output = self.encoder(input_x, attention_mask=attention_mask)[0]  # (B, L, d_h)
        else:
            # When using BiLSTM, extra embedding layer and pad-packing should be applied.
            input_x = torch.transpose(input_x, 0, 1) # (L, B, d_w)

            embedded_input = self.embedding(input_x) # (L, B, d_w)
            packed_input = pack_padded_sequence(embedded_input, lens)

            # Initialize hidden states.
            h_0 = torch.zeros(self.args.num_layers * 2, embedded_input.shape[1], self.args.hidden_size).to(self.args.device)
            c_0 = torch.zeros(self.args.num_layers * 2, embedded_input.shape[1], self.args.hidden_size).to(self.args.device)

            output = self.encoder(packed_input, (h_0, c_0))[0] # (L, B, d_h)
            output = pad_packed_sequence(output)[0].transpose(0, 1).contiguous() # (B, L, d_h)
        
        shape = output.shape
        compressed_embeddings = output.view(-1, shape[2])  # (B * L, d_h)
        pre_attention = self.tanh(self.ws1(self.drop(compressed_embeddings))) # (B * L, d_a)

        attention = self.ws2(pre_attention).view(shape[0], shape[1], -1)  # (B, L, R)
        attention = torch.transpose(attention, 1, 2).contiguous() # (B, R, L)
        attention = F.softmax(attention, dim=-1)

        semantic_vecs = torch.bmm(attention, output) # (B, R, d_h) -> This is the final output of SemanticCaps

        # DetectionCaps
        semantic_vecs_tiled = torch.unsqueeze(semantic_vecs, -1).repeat(1, 1, 1, self.args.train_num_classes * self.args.num_props) # (B, R, D_H, 1) => (B, R, D_H, K * num_properties)
        prediction_vecs = torch.sum(semantic_vecs_tiled * self.capsule_weights, dim=2) # (B, R, D_H, K * num_properties) => (B, R, K * num_properties)
        prediction_vecs_reshaped = torch.reshape(prediction_vecs, [-1, self.args.r, self.args.train_num_classes, self.args.num_props]) # (B, R, K, num_properties)

        semantic_vecs_shape = semantic_vecs.shape
        logits_shape = np.stack([semantic_vecs_shape[0], self.args.r, self.args.train_num_classes])

        # v: (B, K, num_properties), b: (B, R, K), c: (B, R, K)
        v, b, c = self.routing(prediction_vecs_reshaped, logits_shape, num_dims=4, is_train=is_train)

        logits = self.get_logits(v) # (B, K)
        prediction = prediction_vecs_reshaped # (B, R, K, num_properties)

        return attention, logits, prediction, c

    def get_logits(self, activation):
        logits = torch.norm(activation, dim=-1)
        return logits

    def routing(self, prediction_vecs_reshaped, logits_shape, num_dims, is_train):
        prediction_shape = [3, 0, 1, 2]
        for i in range(num_dims - 4):
            prediction_shape += [i + 4]
        r_t_shape = [1, 2, 3, 0]
        for i in range(num_dims - 4):
            r_t_shape += [i + 4]

        prediction_vecs_trans = prediction_vecs_reshaped.permute(prediction_shape)  # (num_properties, B, R, K)
        logits = torch.zeros(logits_shape[0], logits_shape[1], logits_shape[2]).to(self.args.device)  # (B, R, K) This is bkr in the paper
        if is_train:
            logits = nn.Parameter(logits) # (B, R, K) This is bkr in the paper
        activations = []
        routes = None

        # Iterative routing
        for i in range(self.args.num_iters):
            routes = F.softmax(logits, dim=2).to(self.args.device) # (B, R, K) This is cr in the paper
            vote_vecs_unrolled = routes * prediction_vecs_trans # (num_properties, B, R, K)
            vote_vecs = vote_vecs_unrolled.permute(r_t_shape) # (B, R, K, num_properties)

            preactivate = torch.sum(vote_vecs, dim=1) # (B, K, num_properties) This is sk in the paper
            activation = tool.squash(preactivate) # (B, K, num_properties) This is vk in the paper
            activations.append(activation)

            act_extended = activation.unsqueeze(1) # (B, 1, K, num_properties)
            tile_shape = np.ones(num_dims, dtype=np.int32).tolist() # [1, 1, 1, 1]
            tile_shape[1] = self.args.r # [1, R, 1, 1]
            act_replicated = act_extended.repeat(tile_shape) # (B, R, K, num_properties)
            distances = torch.sum(prediction_vecs_reshaped * act_replicated, dim=3) # (B, R, K, num_properties) => (B, R, K)
            logits = logits + distances # (B, R, K)

        return activations[self.args.num_iters-1], logits, routes

    def get_margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        logits = raw_logits - 0.5
        positive_cost = labels * (logits < margin).float() * ((logits - margin) ** 2)
        negative_cost = (1 - labels) * (logits > -margin).float() * ((logits + margin) ** 2)
        
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    def get_loss(self, label, logits, attention):
        label = label.to(self.args.device)
        loss_value = self.get_margin_loss(label.float(), logits)
        loss_value = torch.mean(loss_value)

        self_atten_mul = torch.matmul(attention, attention.permute([0, 2, 1])).float()
        sample_num, att_matrix_size, _ = self_atten_mul.shape
        self_atten_loss = (torch.norm(self_atten_mul - torch.from_numpy(np.identity(att_matrix_size)).float().to(self.args.device)).float()) ** 2

        return 1000 * loss_value + self.args.alpha * torch.mean(self_atten_loss)