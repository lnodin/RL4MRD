import torch
import torch.nn as nn
import torch.nn.functional as F

class HypER(nn.Module):
    def __init__(self, args, num_entities):
        super(HypER, self).__init__()
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.filt_h = args.filt_h
        self.filt_w = args.filt_w

        self.inp_drop = nn.Dropout(args.input_dropout_rate)
        self.hidden_drop = nn.Dropout(args.hidden_dropout_rate)
        self.feature_map_drop = nn.Dropout(args.feat_dropout_rate)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.entity_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        fc_length = (1-self.filt_h+1)*(self.entity_dim-self.filt_w+1)*self.out_channels

        self.fc = torch.nn.Linear(fc_length, self.entity_dim)

        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        self.fc1 = torch.nn.Linear(self.relation_dim, fc1_length)

    def forward(self, e1, r, kg):
        ent = kg.get_entity_embeddings(e1).view(-1, 1, 1, kg.entity_embeddings.size(1))
        rel = kg.get_relation_embeddings(r)
        targets = kg.get_all_entity_embeddings()

        x = self.bn0(ent)
        x = self.inp_drop(x)

        k = self.fc1(rel)

        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(ent.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=ent.size(0))
        x = x.view(ent.size(0), 1, self.out_channels, 1-self.filt_h+1, ent.size(3)-self.filt_w+1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(ent.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, targets.transpose(1,0))
        x += self.b.expand_as(x)

        pred = torch.sigmoid(x)
        return pred

    def forward_fact(self, e1, r, e2, kg):
        ent = kg.get_entity_embeddings(e1).view(-1, 1, 1, kg.entity_embeddings.size(1))
        rel = kg.get_relation_embeddings(r)
        targets = kg.get_all_entity_embeddings(e2)

        x = self.bn0(ent)
        x = self.inp_drop(x)

        k = self.fc1(rel)

        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(ent.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=ent.size(0))
        x = x.view(ent.size(0), 1, self.out_channels, 1-self.filt_h+1, ent.size(3)-self.filt_w+1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(ent.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, targets.transpose(1,0))
        x += self.b.expand_as(x)

        pred = torch.sigmoid(x)
        return pred

class ConvE(nn.Module):
    def __init__(self, args, num_entities):
        super(ConvE, self).__init__()
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        assert (args.emb_2D_d1 * args.emb_2D_d2 == args.entity_dim)
        assert (args.emb_2D_d1 * args.emb_2D_d2 == args.relation_dim)
        self.emb_2D_d1 = args.emb_2D_d1
        self.emb_2D_d2 = args.emb_2D_d2
        self.num_out_channels = args.num_out_channels
        self.w_d = args.kernel_size
        self.HiddenDropout = nn.Dropout(args.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(args.feat_dropout_rate)

        # stride = 1, padding = 0, dilation = 1, groups = 1
        self.conv1 = nn.Conv2d(1, self.num_out_channels,
                               (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(self.entity_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        h_out = 2 * self.emb_2D_d1 - self.w_d + 1
        w_out = self.emb_2D_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        self.fc = nn.Linear(self.feat_dim, self.entity_dim)

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(
            e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(
            r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = kg.get_all_entity_embeddings()

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.mm(X, E2.transpose(1, 0))
        X += self.b.expand_as(X)

        S = torch.sigmoid(X)
        # S = torch.nn.functional.softmax(X, dim=-1)
        return S

    def forward_fact(self, e1, r, e2, kg):
        """
        Compute network scores of the given facts.
        :param e1: [batch_size]
        :param r:  [batch_size]
        :param e2: [batch_size]
        :param kg:
        """
        # print(e1.size(), r.size(), e2.size())
        # print(e1.is_contiguous(), r.is_contiguous(), e2.is_contiguous())
        # print(e1.min(), r.min(), e2.min())
        # print(e1.max(), r.max(), e2.max())
        E1 = kg.get_entity_embeddings(
            e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(
            r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = kg.get_entity_embeddings(e2)

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
        X += self.b[e2].unsqueeze(1)
        S = torch.sigmoid(X)
        return S
