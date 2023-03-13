"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Fact scoring networks.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/model.py
"""

import copy

import torch
import torch.nn as nn

from .cnn import ConvE
from .complex import ComplEx
from .decom import DistMult

from .utils import get_conve_nn_state_dict

#########################################################
# Ensemble
#########################################################


class TripleE(nn.Module):
    def __init__(self, args, num_entities):
        super(TripleE, self).__init__()
        conve_args = copy.deepcopy(args)
        conve_args.model = 'conve'
        self.conve_nn = ConvE(conve_args, num_entities)
        conve_state_dict = torch.load(args.conve_state_dict_path)
        conve_nn_state_dict = get_conve_nn_state_dict(conve_state_dict)
        self.conve_nn.load_state_dict(conve_nn_state_dict)

        complex_args = copy.deepcopy(args)
        complex_args.model = 'complex'
        self.complex_nn = ComplEx(complex_args)

        distmult_args = copy.deepcopy(args)
        distmult_args.model = 'distmult'
        self.distmult_nn = DistMult(distmult_args)

    def forward(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        distmult_kg = secondary_kgs[1]
        return (self.conve_nn.forward(e1, r, conve_kg)
                + self.complex_nn.forward(e1, r, complex_kg)
                + self.distmult_nn.forward(e1, r, distmult_kg)) / 3

    def forward_fact(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        distmult_kg = secondary_kgs[1]
        return (self.conve_nn.forward_fact(e1, r, conve_kg)
                + self.complex_nn.forward_fact(e1, r, complex_kg)
                + self.distmult_nn.forward_fact(e1, r, distmult_kg)) / 3


class HyperE(nn.Module):
    def __init__(self, args, num_entities):
        super(HyperE, self).__init__()
        self.conve_nn = ConvE(args, num_entities)
        conve_state_dict = torch.load(args.conve_state_dict_path)
        conve_nn_state_dict = get_conve_nn_state_dict(conve_state_dict)
        self.conve_nn.load_state_dict(conve_nn_state_dict)

        complex_args = copy.deepcopy(args)
        complex_args.model = 'complex'
        self.complex_nn = ComplEx(complex_args)

    def forward(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        return (self.conve_nn.forward(e1, r, conve_kg)
                + self.complex_nn.forward(e1, r, complex_kg)) / 2

    def forward_fact(self, e1, r, e2, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        return (self.conve_nn.forward_fact(e1, r, e2, conve_kg)
                + self.complex_nn.forward_fact(e1, r, e2, complex_kg)) / 2
