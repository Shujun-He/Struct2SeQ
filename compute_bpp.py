import subprocess
import tempfile
import os
from turtle import pd
import numpy as np

CONTRA_PATH = "/scratch/shujun/PK_design_240/EternaFold/src/contrafold"
PARAMS_PATH = "/scratch/shujun/PK_design_240/EternaFold/parameters/EternaFoldParams.v1"


def compute_bpp(sequence, posterior_cutoff=1e-5):
    """
    Compute base-pairing probabilities using CONTRAfold/EternaFold.

    Returns:
        BPP matrix of shape (L, L)
    """
    L = len(sequence)

    # --- Step 1: write sequence to temporary .seq file ---
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_path = os.path.join(tmpdir, "input.seq")
        out_path = os.path.join(tmpdir, "bps.txt")

        # CONTRAfold expects FASTA-like format but plain sequence is fine
        with open(seq_path, "w") as f:
            f.write(sequence + "\n")

        # --- Step 2: run contraFold ---
        cmd = [
            CONTRA_PATH, "predict", seq_path,
            "--params", PARAMS_PATH,
            "--posteriors", str(posterior_cutoff),
            out_path
        ]

        # subprocess.run(cmd, check=True)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,     # capture STDOUT
            stderr=subprocess.DEVNULL,  # hide STDERR noise
            text=True,
            check=True,
        )

        # Initialize BPP matrix
        bpp = np.zeros((L, L), dtype=np.float32)

        # Parse output
        with open(out_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                i = int(parts[0]) - 1    # 1-based -> 0-based
                # parts[1] = nucleotide, ignore it

                # If no pairs:
                if len(parts) == 2:
                    continue

                # Parse j:p tokens
                for token in parts[2:]:
                    if ":" not in token:
                        continue
                    j_str, p_str = token.split(":")
                    j = int(j_str) - 1
                    p = float(p_str)

                    bpp[i, j] = p
                    bpp[j, i] = p  # symmetric


    return bpp
def compute_mean_bpp_for_seq(sequence):
    bpp = compute_bpp(sequence)
    return float(np.mean(bpp.sum(axis=1)))

import polars as pl
from tqdm import tqdm

df = pl.read_csv("filtered_replicase.csv")

# mean_bpp=[]
# for seq in tqdm(df["sequence"]):
#     bpp = compute_bpp(seq)
#     mean_bpp.append(np.mean(bpp.sum(1)))
#     break
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd

df = pd.read_csv("filtered_replicase.csv")
sequences = df["sequence"].to_list()

num_workers = mp.cpu_count()  # or specify manually

with mp.Pool(num_workers) as pool:
    mean_bpp = list(tqdm(pool.imap(compute_mean_bpp_for_seq, sequences),
                         total=len(sequences)))

df['mean_bpp'] = mean_bpp
df.to_csv("filtered_replicase_with_bpp.csv", index=False)

# seq = "AGUCAUUGCCGCACCAAGACAAAUCUCCCCCCAGAGCCUGAGAACAUCCACGGAUGCAGAGGAGGGAGCCUUCGGUGGAUUAAUGGUGCACCACCGUUCUCAGCACGUACCCGAACGAAAAAGACCUGACAGAAAGGCGUUGUUAGACACGCACAGGUACCAUGCCCAACACAUGGCUGAC"
# bpp = compute_bpp(seq)

# import matplotlib.pyplot as plt
# import numpy as np

# def plot_bpp(bpp, cmap="viridis"):
#     """
#     Plot a base-pairing probability (BPP) matrix using imshow.
#     """
#     L = bpp.shape[0]
    
#     plt.figure(figsize=(6, 5))
#     plt.imshow(bpp, cmap=cmap, origin="lower", vmin=0, vmax=1)

#     plt.colorbar(label="Base-pairing probability")
#     plt.title("Base Pair Probability Matrix (BPP)")
#     plt.xlabel("Position j")
#     plt.ylabel("Position i")
#     plt.tight_layout()
#     plt.savefig("bpp_matrix.png")

# plot_bpp(bpp)