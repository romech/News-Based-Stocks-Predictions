import pandas as pd


def csv_to_tsv(path, header=False, index=False):
    out_path = path.replace(".csv", ".tsv").replace(".txt", ".tsv")
    pd.read_csv(path).to_csv(out_path, header=header, index=index)


def matrix_to_tsv(path, out_path=None):
    out_path = out_path or path.replace(".csv", ".tsv").replace(".txt", ".tsv")
    with open(path, "r") as inp, \
         open(out_path, "w") as out:

        for line in inp:
            out.write(line.replace(" ", "\t"))
