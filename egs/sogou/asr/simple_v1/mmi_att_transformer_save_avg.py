#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
#                2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import argparse
import k2
import logging
import numpy as np
import os
import torch
from k2 import Fsa, SymbolTable
from kaldialign import edit_distance
from pathlib import Path
from typing import List
from typing import Union

from snowfall.common import average_checkpoint, store_transcripts
from snowfall.common import find_first_disambig_symbol
from snowfall.common import get_texts
from snowfall.common import write_error_stats
from snowfall.common import load_checkpoint
from snowfall.common import setup_logger
from snowfall.common import str2bool
from snowfall.data import SogouAsrDataModule
from snowfall.decoding.graph import compile_HLG
from snowfall.decoding.lm_rescore import decode_with_lm_rescoring
from snowfall.models import AcousticModel
from snowfall.models.transformer import Transformer
from snowfall.models.conformer import Conformer
from snowfall.models.contextnet import ContextNet
from snowfall.training.ctc_graph import build_ctc_topo
from snowfall.training.mmi_graph import create_bigram_phone_lm
from snowfall.training.mmi_graph import get_phone_symbols
from snowfall.common import save_checkpoint


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model-type',
        type=str,
        default="conformer",
        choices=["transformer", "conformer", "contextnet"],
        help="Model type.")
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help="Decoding epoch.")
    parser.add_argument(
        '--avg',
        type=int,
        default=5,
        help="Number of checkpionts to average. Automaticly select "
             "consecutive checkpoints before checkpoint specified by'--epoch'. ")
    parser.add_argument(
        '--att-rate',
        type=float,
        default=0.0,
        help="Attention loss rate.")
    parser.add_argument(
        '--nhead',
        type=int,
        default=4,
        help="Number of attention heads in transformer.")
    parser.add_argument(
        '--attention-dim',
        type=int,
        default=256,
        help="Number of units in transformer attention layers.")
    parser.add_argument(
        '--output-beam-size',
        type=float,
        default=8,
        help='Output beam size. Used in k2.intersect_dense_pruned.'\
             'Choose a large value (e.g., 20), for 1-best decoding '\
             'and n-best rescoring. Choose a small value (e.g., 8) for ' \
             'rescoring with the whole lattice')
    parser.add_argument(
        '--use-lm-rescoring',
        type=str2bool,
        default=True,
        help='When enabled, it uses LM for rescoring')
    parser.add_argument(
        '--num-paths',
        type=int,
        default=-1,
        help='Number of paths for rescoring using n-best list.' \
             'If it is negative, then rescore with the whole lattice.'\
             'CAUTION: You have to reduce max_duration in case of CUDA OOM'
             )
    return parser


def main():
    parser = get_parser()
    SogouAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    model_type = args.model_type
    epoch = args.epoch
    avg = args.avg
    att_rate = args.att_rate
    num_paths = args.num_paths
    use_lm_rescoring = args.use_lm_rescoring
    use_whole_lattice = False
    if use_lm_rescoring and num_paths < 1:
        # It doesn't make sense to use n-best list for rescoring
        # when n is less than 1
        use_whole_lattice = True

    output_beam_size = args.output_beam_size
    exp_dir = Path('exp-' + model_type + '-noam-mmi-att-sogou500h')
    setup_logger('{}/log/log-decode'.format(exp_dir), log_level='debug')

    logging.info(f'output_beam_size: {output_beam_size}')

    # load L, G, symbol_table
    lang_dir = Path('data/lang_nosp')
    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')

    phone_ids = get_phone_symbols(phone_symbol_table)
    P = create_bigram_phone_lm(phone_ids)

    phone_ids_with_blank = [0] + phone_ids
    ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))

    logging.debug("About to load model")
    # Note: Use "export CUDA_VISIBLE_DEVICES=N" to setup device id to N
    # device = torch.device('cuda', 1)
    device = torch.device('cuda')

    if att_rate != 0.0:
        num_decoder_layers = 6
    else:
        num_decoder_layers = 0

    if model_type == "transformer":
        model = Transformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=True)
    elif model_type == "conformer":
        model = Conformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=False,
            is_espnet_structure=False)
    elif model_type == "contextnet":
        model = ContextNet(
        num_features=80,
        num_classes=len(phone_ids) + 1) # +1 for the blank symbol
    else:
        raise NotImplementedError("Model of type " + str(model_type) + " is not implemented")

    model.P_scores = torch.nn.Parameter(P.scores.clone(), requires_grad=False)

    if avg == 1:
        checkpoint = os.path.join(exp_dir, 'epoch-' + str(epoch - 1) + '.pt')
        load_checkpoint(checkpoint, model)
    else:
        checkpoints = [os.path.join(exp_dir, 'epoch-' + str(avg_epoch) + '.pt') for avg_epoch in
                       range(epoch - avg, epoch)]
        average_checkpoint(checkpoints, model)
    best_model_path = os.path.join(exp_dir, 'avg_model.pt')
    save_checkpoint(filename=best_model_path,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    model=model,
                    epoch=None,
                    learning_rate=None,
                    objf=None,
                    valid_objf=None,
                    global_batch_idx_train=None,
                    local_rank=0)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == '__main__':
    main()
