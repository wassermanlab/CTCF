#!/usr/bin/env bash

# Train a CTCF model with DragoNN
# https://github.com/kundajelab/dragonn
dragonn train --pos-sequences ../Data/pos_seqs.fa --neg-sequences ../Data/neg_seqs.fa --prefix CTCF
# Finished training after 18 epochs.
# final validation metrics:
# Loss: 0.3182	Balanced Accuracy: 87.63%	 auROC: 0.935	 auPRC: 0.947	 auPRG: 0.910
# 	Recall at 5%|10%|20% FDR: 76.5%|85.4%|91.9%	 Num Positives: 15880	 Num Negatives: 15714
dragonn train --pos-sequences ../Data/pos_seqs.fwd.fa --neg-sequences ../Data/neg_seqs.fwd.fa --prefix CTCF-fwd
# Finished training after 22 epochs.
# final validation metrics:
# Loss: 0.1642	Balanced Accuracy: 94.54%	 auROC: 0.981	 auPRC: 0.984	 auPRG: 0.976
# 	Recall at 5%|10%|20% FDR: 95.2%|97.1%|98.1%	 Num Positives: 11494	 Num Negatives: 11355
