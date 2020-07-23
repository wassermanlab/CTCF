#!/bin/bash

# Train a CTCF model with DragoNN
# https://github.com/kundajelab/dragonn
dragonn train --pos-sequences ../sequences/pos_seqs.fa --neg-sequences ../sequences/neg_seqs.fa --prefix CTCF
dragonn train --pos-sequences ../sequences/pos_seqs.fwd.fa --neg-sequences ../sequences/neg_seqs.fwd.fa --prefix CTCF-fwd
dragonn train --pos-sequences ../sequences/pos_seqs.rev.fa --neg-sequences ../sequences/neg_seqs.rev.fa --prefix CTCF-rev
