import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplEx(nn.Module):
    def __init__(self, args):
        super(ComplEx, self).__init__()

    def forward(self, e1, r, kg):
        def dist_mult(E1, R, E2):
            return torch.mm(E1 * R, E2.transpose(1, 0))

        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E2_real = kg.get_all_entity_embeddings()
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)
        E2_img = kg.get_all_entity_img_embeddings()

        rrr = dist_mult(R_real, E1_real, E2_real)
        rii = dist_mult(R_real, E1_img, E2_img)
        iri = dist_mult(R_img, E1_real, E2_img)
        iir = dist_mult(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2, kg):
        def dist_mult_fact(E1, R, E2):
            return torch.sum(E1 * R * E2, dim=1, keepdim=True)

        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E2_real = kg.get_entity_embeddings(e2)
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)
        E2_img = kg.get_entity_img_embeddings(e2)

        rrr = dist_mult_fact(R_real, E1_real, E2_real)
        rii = dist_mult_fact(R_real, E1_img, E2_img)
        iri = dist_mult_fact(R_img, E1_real, E2_img)
        iir = dist_mult_fact(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S

class RotatE(nn.Module):
    def __init__(self, args):
        super(RotatE, self).__init__()

    def forward(self, e1, r, kg):
        h_re, h_im = kg.get_entity_embeddings(e1), kg.get_entity_img_embeddings(e1)
        t_re, t_im = kg.get_all_entity_embeddings(), kg.get_all_entity_img_embeddings()

        r_re = torch.cos(kg.get_relation_embeddings(r))
        r_im = torch.sin(kg.get_relation_embeddings(r))

        score_re = h_re * r_re - h_im * r_im
        score_im = h_re * r_im + h_im * r_re
        score = score_re * t_re + score_im * t_im

        return torch.sum(score, dim=-1)

    def forward_fact(self, e1, r, e2, kg):
        E1_real = kg.get_entity_embeddings(e1)
        E1_img = kg.get_entity_img_embeddings(e1)


        E2_real = kg.get_entity_embeddings(e2)
        E2_img = kg.get_entity_img_embeddings(e2)


        R_real = kg.get_relation_embeddings(r)
        R_img = kg.get_relation_img_embeddings(r)

        re_score = R_real * E1_real + R_img * E1_img
        im_score = R_real * E1_img - R_img * E1_real

        re_score = re_score - E2_real
        im_score = im_score - E2_img

        score = torch.stack([re_score, im_score], dim=0)

        score = score.norm(dim=0).sum(dim=-1)

        return score


class Fourier(nn.Module):
    def __init__(self,  args):
        super(Fourier, self).__init__()

    def forward(self, e1, r, kg):
        # entities & relation embedding
        ent_real = kg.get_entity_embeddings(e1)
        ent_img = kg.get_entity_img_embeddings(e1)

        rel_real = kg.get_relation_embeddings(r)
        rel_img = kg.get_relation_img_embeddings(r)

        target_real = kg.get_all_entity_embeddings()
        target_img = kg.get_all_entity_img_embeddings()

        dim = 2 * ent_real.shape[1] // 2 - 1

        # IFFT
        ent = ent_real + 1j * ent_img
        ent = torch.fft.irfft(ent, norm='ortho', n=dim)

        # high-pass & low-pass
        rel_real = rel_real[..., :dim]
        rel_img = rel_img[..., dim:]
        hplp = 0.5 * (torch.min(ent, rel_real) + torch.max(ent, rel_img))

        # FFT
        hplp = torch.fft.rfft(hplp, norm='ortho', n=2*ent_real.shape[1]-1)

        result = torch.mm(torch.cat((hplp.real, hplp.imag), -1), torch.cat((target_real, target_img), -1).transpose(1, 0))

        result = torch.sigmoid(result)
        return result

    def forward_fact(self, e1, r, e2, kg):
        # entities & relation embedding
        ent_real = kg.get_entity_embeddings(e1)
        ent_img = kg.get_entity_img_embeddings(e1)

        rel_real = kg.get_relation_embeddings(r)
        rel_img = kg.get_relation_img_embeddings(r)

        target_real = kg.get_entity_embeddings(e2)
        target_img = kg.get_entity_img_embeddings(e2)

        dim = 2 * ent_real.shape[1] // 2 - 1

        # IFFT
        ent = ent_real + 1j * ent_img
        ent = torch.fft.irfft(ent, norm='ortho', n=dim)

        # high-pass & low-pass
        rel_real = rel_real[..., :dim]
        rel_img = rel_img[..., dim:]
        hplp = 0.5 * (torch.min(ent, rel_real) + torch.max(ent, rel_img))

        # FFT
        hplp = torch.fft.rfft(hplp, norm='ortho', n=2*ent_real.shape[1]-1)

        result = torch.mm(torch.cat((hplp.real, hplp.imag), -1), torch.cat((target_real, target_img), -1).transpose(1, 0))

        result = torch.sigmoid(result)
        return result
