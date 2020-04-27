#!/bin/bash

# Train a CTCF model with DragoNN
# https://github.com/kundajelab/dragonn
dragonn train --pos-sequences ../sequences/pos_seqs.fa --neg-sequences ../sequences/neg_seqs.fa --prefix CTCF
