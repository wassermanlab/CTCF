import numpy as np

def one_hot_encode(seqs):
    """
    Adapted from:
    https://github.com/kundajelab/dragonn/blob/master/dragonn/utils/__init__.py

    One hot encodes a list of sequences.
    """

    # Initialize
    encoded_seqs = []
    one_hot_encoder = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0]
    }

    for seq in seqs:
        encoded_seq = np.array(
            [one_hot_encoder.get(s, [0, 0, 0, 0]) for s in seq]
        )
        encoded_seqs.append(encoded_seq)

    return(np.array(encoded_seqs))

def reverse_complement(encoded_seqs):
    """
    Adapted from:
    https://github.com/kundajelab/dragonn/blob/master/dragonn/utils/__init__.py

    Reverse complements a list of one hot encoded sequences.
    """
    return(encoded_seqs[..., ::-1, ::-1])