from PIL import Image
import numpy as np

def save_gif(single_seq, fname):
    """Save a single gif consisting of image sequence in single_seq to fname."""
    img_seq = [Image.fromarray(img.astype(np.float32) * 255, 'F').convert("L") for img in single_seq]
    img = img_seq[0]
    img.save(fname, save_all=True, append_images=img_seq[1:])

def save_gifs(seq, prefix):
    """Save several gifs.

    Args:
      seq: Shape (num_gifs, IMG_SIZE, IMG_SIZE)
      prefix: prefix-idx.gif will be the final filename.
    """

    for idx, single_seq in enumerate(seq):
        save_gif(single_seq, "{}-{}.gif".format(prefix, idx))
