import typer
import tqdm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
from typing import Iterable, Tuple
import numpy as np

from FEGS import FEGS

def readfasta(file_path: str) -> Iterable[SeqRecord]:
    fasta = list(SeqIO.parse(file_path, "fasta"))
    return fasta

def readcsv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def main(file_path, output_path):
    # fasta = readfasta(file_path)
    # seqs = [str(seq.seq) for seq in fasta]
    df = readcsv(file_path)
    seqs = df["sequence"].tolist()
    l_seqs = len(seqs)
    result = []
    for i in tqdm.tqdm(range(l_seqs)):
        res = FEGS(seqs[i])
        result.append(res)
    result = np.array(result)
    np.save(output_path, result)

if __name__ == "__main__":
    typer.run(main)