import torch
import torch.nn as nn
from torch.autograd import Variable

from .utils import euc_sqdistance, givens_rotations, givens_reflection

class TransE(nn.Module):
    def __init__(self, args):
        super(TransE, self).__init__()

    def forward_train(self, e1, e2, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        return torch.abs(E1 + R - E2)

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_all_entity_embeddings()
        size_e1 = E1.size()
        size_e2 = E2.size()

        A = torch.sum((E1 + R) * (E1 + R), dim=1)
        B = torch.sum(E2 * E2, dim=1)
        AB = torch.mm((E1 + R), E2.transpose(1, 0))
        S = A.view(size_e1[0], 1) + B.view(1, size_e2[0]) - 2 * AB

        return torch.sigmoid(-torch.sqrt(S))

    def forward_fact(self, e1, r, e2, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        return torch.sigmoid(-torch.sqrt(torch.sum((E1 + R - E2) * (E1 + R - E2), dim=1, keepdim=True)))

class PTransE(nn.Module):
    def __init__(self, args):
        super(PTransE, self).__init__()

    def forward_train(self, e1, e2, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        return torch.abs(E1 + R - E2)

    def forward_train_relation(self, e1, e2, r, kg, corrupt=False, corrupt_r=None):
        batch_size = e1.size()[0]
        relation_weight = Variable(torch.zeros(
            batch_size, kg.num_relations), requires_grad=False).cuda()
        for i in range(batch_size):
            e1_id, e2_id, r_id = int(e1[i]), int(e2[i]), int(r[i])
            path_info = kg.triple2path[(e1_id, e2_id, r_id)]
            for path in path_info:
                prob = path[-1]
                for relation in path[0]:
                    relation_weight[i][relation] += prob
        P = torch.mm(relation_weight, kg.get_all_relation_embeddings())
        if corrupt == False:
            R = kg.get_relation_embeddings(r)
        else:
            R = kg.get_relation_embeddings(corrupt_r)
        return torch.abs(P - R)

    def forward(self, e1, r, kg, path_trace):
        return None

    def forward_fact(self, e1, r, e2, kg, path_trace):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        first = torch.abs(E1 + R - E2)
        second = R
        for i in range(1, len(path_trace)):
            second = R - kg.get_relation_embeddings(path_trace[i][0])
        S = torch.sum(first + torch.abs(second), dim=1, keepdim=True)
        return (torch.sigmoid(1 / S) - 0.5) * 2

class CP(nn.Module):
    def __init__(self) -> None:
        super(CP, self).__init__()

    def forward_train(self, e1, e2, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)

        return E1 * R * E2

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_all_entity_embeddings()

        size_e1 = E1.size()
        size_e2 = E2.size()

        A = torch.sum((E1 + R) * (E1 + R), dim=1)
        B = torch.sum(E2 * E2, dim=1)
        AB = torch.mm((E1 + R), E2.transpose(1, 0))
        S = A.view(size_e1[0], 1) + B.view(1, size_e2[0]) - 2 * AB

        return torch.sigmoid(-torch.sqrt(S))

    def forward_fact(self, e1, r, e2, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        return torch.sigmoid(-torch.sqrt(torch.sum((E1 + R - E2) * (E1 + R - E2), dim=1, keepdim=True)))


class MurE(nn.Module):
    def __init__(self, args) -> None:
        super(MurE, self).__init__()


class RotE(nn.Module):
    def __init__(self, args) -> None:
        super(RotE, self).__init__()


class RefE(nn.Module):
    def __init__(self, args) -> None:
        super(RefE, self).__init__()


class AttE(nn.Module):
    def __init__(self, args) -> None:
        super(AttE, self).__init__()