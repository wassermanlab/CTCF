#!/bin/bash

# Train a CTCF model with DragoNN
# https://github.com/kundajelab/dragonn
dragonn train --pos-sequences ../Data/pos_seqs.fa --neg-sequences ../Data/neg_seqs.fa --prefix CTCF
dragonn train --pos-sequences ../Data/pos_seqs.fwd.fa --neg-sequences ../Data/neg_seqs.fwd.fa --prefix CTCF-fwd
dragonn train --pos-sequences ../Data/pos_seqs.rev.fa --neg-sequences ../Data/neg_seqs.rev.fa --prefix CTCF-rev
