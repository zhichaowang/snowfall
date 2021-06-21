#!/usr/bin/env bash

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# Example of how to build L and G FST for K2. Most scripts of this example are copied from Kaldi.

set -eou pipefail

stage=4
train_lm=false

if [ $stage -le 1 ]; then
  local/download_lm.sh "openslr.org/resources/11" data/local/lm
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh data/local/lm data/local/dict_nosp
fi

if [ $stage -le 3 ]; then
  echo "Language dir preparation"
  local/prepare_lang.sh \
    --position-dependent-phones false \
    data/local/dict_0528_char \
    "<UNK>" \
    data/local/lang_tmp_char_nosp \
    data/lang_0528_char_nosp

  echo "To load L:"
  echo "    Lfst = k2.Fsa.from_openfst(<string of data/lang_nosp/L.fst.txt>, acceptor=False)"
fi

if [ $stage -le 4 ]; then
  echo "LM training..."
  if $train_lm; then
    local2/sogou_train_lms.sh data/local/train/text data/local/dict_0528_char/lexicon.txt data/local/lm_char
    gunzip -c data/local/lm_char/3gram-mincount/lm_unpruned.gz >data/local/lm_char/lm_tgmed.arpa
  fi

  # Build G
  python3 -m kaldilm \
    --read-symbol-table="data/lang_0528_char_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=1 \
    data/local/lm_char/lm_tgmed.arpa >data/lang_0528_char_nosp/G_uni.fst.txt

  python3 -m kaldilm \
    --read-symbol-table="data/lang_0528_char_nosp/words.txt" \
    --disambig-symbol='#0' \
    --max-order=3 \
    data/local/lm_char/lm_tgmed.arpa >data/lang_0528_char_nosp/G.fst.txt

#  python3 -m kaldilm \
#    --read-symbol-table="data/lang_nosp/words.txt" \
#    --disambig-symbol='#0' \
#    --max-order=4 \
#    data/local/lm/lm_fglarge.arpa >data/lang_nosp/G_4_gram.fst.txt

  echo ""
  echo "To load G:"
  echo "Use::"
  echo "  with open('data/lang_nosp/G.fst.txt') as f:"
  echo "    G = k2.Fsa.from_openfst(f.read(), acceptor=False)"
  echo ""
fi
exit 0
if [ $stage -le 5 ]; then
  python3 ./prepare.py
fi

if [ $stage -le 6 ]; then
  # python3 ./train.py # ctc training
  # python3 ./mmi_bigram_train.py # ctc training + bigram phone LM
  #  python3 ./mmi_mbr_train.py

  # Single node, multi-GPU training
  # Adapting to a multi-node scenario should be straightforward.
  ngpus=2
#  python3 -m torch.distributed.launch --nproc_per_node=$ngpus ./mmi_bigram_train.py --world_size $ngpus
  CUDA_VISIBLE_DEVICES=4,5,6,7 python3 ./mmi_att_transformer_train_att0.5.py --world-size=4 --amp=False --use-ali-model=False --max-duration=300 --att-rate=0.5 --num-epochs=15 --master-port=12360
fi

if [ $stage -le 7 ]; then
  # python3 ./decode.py # ctc decoding
  #python3 ./mmi_bigram_decode.py --epoch 9
  #  python3 ./mmi_mbr_decode.py
  ./mmi_att_transformer_decode.py --epoch=10 --avg=5 --use-lm-rescoring=False --num-path=100 --max-duration=200 --output-beam-size=7
fi
