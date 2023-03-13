import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class TuckER(nn.Module):
    def __init__(self, args, num_entities):
        super(TuckER, self).__init__()
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.register_parameter('W', nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.relation_dim,
                                self.entity_dim, self.entity_dim)), dtype=torch.float, device="cuda", requires_grad=True)))
        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.4)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

        self.bn0 = torch.nn.BatchNorm1d(self.entity_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.entity_dim)

    def forward(self, e1_idx, r_idx, kg):
        e1 = kg.get_entity_embeddings(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = kg.get_relation_embeddings(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, kg.get_all_entity_embeddings().transpose(1, 0))
        # pred = torch.nn.functional.softmax(x, dim=-1)
        pred = torch.sigmoid(x)
        return pred

    def forward_fact(self, e1_idx, r_idx, e2_idx, kg):
        E2 = kg.get_entity_embeddings(e2_idx)

        e1 = kg.get_entity_embeddings(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = kg.get_relation_embeddings(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        X = self.hidden_dropout2(x)

        X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
        S = torch.sigmoid(X)
        return S


class LowFER(nn.Module):
    def __init__(self, args):
        super(LowFER, self).__init__()
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        # Latent dimension of MFB.
        self.k, self.o = args.k, self.entity_dim

        self.U = nn.Parameter(torch.tensor(np.random.uniform(-0.01, 0.01, (self.entity_dim, self.k * self.o)),
                                           dtype=torch.float, device="cuda", requires_grad=True))
        self.V = nn.Parameter(torch.tensor(np.random.uniform(-0.01, 0.01, (self.relation_dim, self.k * self.o)),
                                           dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.4)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

        self.bn0 = nn.BatchNorm1d(self.entity_dim)
        self.bn1 = nn.BatchNorm1d(self.entity_dim)

        self.nonlinearity = nn.PReLU()

    def forward(self, e1_idx, r_idx, kg):
        e1 = kg.get_entity_embeddings(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)

        r = kg.get_relation_embeddings(r_idx)

        x = torch.mm(e1, self.U) * torch.mm(r, self.V)
        x = self.hidden_dropout1(x)
        x = x.view(-1, self.o, self.k)
        x = x.sum(-1)
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = self.nonlinearity(x)
        # x = torch.nn.functional.leaky_relu(x)
        x = torch.mm(x, kg.get_all_entity_embeddings().transpose(1, 0))

        x = torch.sigmoid(x)

        return x

    def forward_fact(self, e1_idx, r_idx, e2_idx, kg):
        e1 = kg.get_entity_embeddings(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        e2 = kg.get_entity_embeddings(e2_idx)

        r = kg.get_relation_embeddings(r_idx)

        x = torch.mm(e1, self.U) * torch.mm(r, self.V)
        x = self.hidden_dropout1(x)
        x = x.view(-1, self.o, self.k)
        x = x.sum(-1)
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = nn.functional.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = self.nonlinearity(x)

        x = torch.matmul(x.unsqueeze(1), e2.unsqueeze(2)).squeeze(2)

        x = torch.sigmoid(x)

        return x


class DistMult(nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_all_entity_embeddings()
        S = torch.mm(E1 * R, E2.transpose(1, 0))
        S = torch.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        S = torch.sum(E1 * R * E2, dim=1, keepdim=True)
        S = torch.sigmoid(S)
        return S
