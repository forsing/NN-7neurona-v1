"""
https://github.com/Ramakm
"""


"""
Loto 7/39: 7 neurona, samo forward (bez fit).
Ulaz: empirijska distribucija. Težine fiksne SEED=39. Dva CSV: loto + plus.
"""



# =============================================================================
# Loto 7/39 — 7 neurona, samo forward (bez fit / bez backprop)
# Ulaz: empirijska distribucija udeo[b]=p(b) nad 1..39
# Težine: determinističke iz SEED=39 
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

SEED = 39
MIN_BROJ = 1
MAX_BROJ = 39
BROJEVA_U_KOMBINACIJI = 7
N_NEURONA = 7
N_ULAZA = MAX_BROJ - MIN_BROJ + 1  # 39
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_LOTO = DATA_DIR / "loto7_4658_k60_loto_2951.csv"
CSV_PLUS = DATA_DIR / "loto7_4658_k60_loto_plus_1707.csv"
NUM_COLS = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]


def csv_jobs():
    return [
        (CSV_LOTO, "LOTO", "next_loto"),
        (CSV_PLUS, "LOTO PLUS", "next_loto_plus"),
    ]


def ucitaj_distribuciju(csv_path):
    """Empirijska distribucija udeo[b]=p(b) nad 1..39 (suma=1)."""
    peek = pd.read_csv(csv_path, nrows=0)
    if all(c in peek.columns for c in NUM_COLS):
        df = pd.read_csv(csv_path)[NUM_COLS].astype(int)
    else:
        df = pd.read_csv(csv_path, header=None).iloc[:, :7].astype(int)
    flat = df.to_numpy().ravel()
    pojave = {b: 0 for b in range(MIN_BROJ, MAX_BROJ + 1)}
    for v in flat:
        if MIN_BROJ <= int(v) <= MAX_BROJ:
            pojave[int(v)] += 1
    ukupno = float(sum(pojave.values())) or 1.0
    udeo = {b: pojave[b] / ukupno for b in pojave}
    n_kola = int(ukupno) // BROJEVA_U_KOMBINACIJI
    return udeo, n_kola


def tezine_7_neurona():
    """W (39,7), b (7,) — fiksno iz SEED."""
    W = np.zeros((N_ULAZA, N_NEURONA))
    b = np.zeros(N_NEURONA)
    for i in range(N_ULAZA):
        for j in range(N_NEURONA):
            W[i, j] = 0.3 * np.sin((i + 1) * (j + 1) * SEED * 0.01)
    for j in range(N_NEURONA):
        b[j] = 0.1 * np.cos((j + 1) * SEED * 0.01)
    return W, b

 
class SevenNeurons:
    """7 neurona: ulaz udeo (39,) → h=sigmoid(xW+b) → skor po broju."""

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        """x shape (39,); vraća h shape (7,)."""
        z = x @ self.W + self.b
        return sigmoid(z)

    def skorovi(self, udeo):
        x = np.array([udeo[b] for b in range(MIN_BROJ, MAX_BROJ + 1)], dtype=float)
        h = self.forward(x)
        # skor(b) = p(b) * (W[b,:] · h) — 7 neurona glasa za broj
        raw = self.W @ h
        return {b: float(udeo[b] * raw[b - 1]) for b in range(MIN_BROJ, MAX_BROJ + 1)}, h


def next_kombinacija(skorovi):
    poredak = sorted(skorovi.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(sorted(b for b, _ in poredak[:BROJEVA_U_KOMBINACIJI]))


def main(csv_path, label, next_key):
    udeo, n_kola = ucitaj_distribuciju(csv_path)
    W, b = tezine_7_neurona()
    net = SevenNeurons(W, b)
    skorovi, h = net.skorovi(udeo)
    nxt = list(next_kombinacija(skorovi))

    print(f"NN_v1 — 7 neurona forward (bez fit) | {label}")
    print(f"SEED={SEED} | CSV={Path(csv_path).name} | kola={n_kola}")
    print(f"h (7 neurona): {np.round(h, 4)}")
    print()
    print("broj | udeo | skor")
    for b, s in sorted(skorovi.items(), key=lambda kv: (-kv[1], kv[0]))[:15]:
        print(f"{b:4d} | {udeo[b]:.6f} | {s:.8f}")
    print("...")
    print()
    print(f"{next_key}: {nxt}")
    return nxt


if __name__ == "__main__":
    for _csv, _label, _next_key in csv_jobs():
        print(f"=== {_label} ===")
        main(_csv, _label, _next_key)
        print()



"""
RUN:

=== LOTO ===
NN_v1 — 7 neurona forward (bez fit) | LOTO
SEED=39 | CSV=loto7_4658_k60_loto_2951.csv | kola=2951
h (7 neurona): [0.5327 0.5175 0.5126 0.5008 0.4908 0.4829 0.4774]

broj | udeo | skor
  33 | 0.026480 | 0.02139049
  17 | 0.024205 | 0.01935040
   1 | 0.024302 | 0.01846773
   3 | 0.025706 | 0.00575522
  19 | 0.025802 | 0.00521254
  35 | 0.026771 | 0.00456902
  34 | 0.026529 | 0.00346176
   5 | 0.025560 | 0.00246110
  36 | 0.023818 | 0.00198742
  21 | 0.026577 | 0.00193057
  38 | 0.026529 | 0.00184237
  18 | 0.024447 | 0.00171781
  37 | 0.026141 | 0.00122290
  22 | 0.027400 | 0.00116337
  20 | 0.023624 | 0.00111682
...

next_loto: [1, 3, 17, 19, 33, 34, 35]

=== LOTO PLUS ===
NN_v1 — 7 neurona forward (bez fit) | LOTO PLUS
SEED=39 | CSV=loto7_4658_k60_loto_plus_1707.csv | kola=1707
h (7 neurona): [0.5324 0.5187 0.5127 0.5004 0.4917 0.482  0.476 ]

broj | udeo | skor
  33 | 0.026027 | 0.02101726
   1 | 0.024772 | 0.01882486
  17 | 0.022429 | 0.01792677
   3 | 0.025274 | 0.00564813
  19 | 0.023851 | 0.00480874
  35 | 0.024605 | 0.00419079
  34 | 0.027450 | 0.00360481
   5 | 0.025525 | 0.00243864
  36 | 0.024772 | 0.00207654
  18 | 0.026948 | 0.00191487
  38 | 0.024939 | 0.00170991
  21 | 0.023433 | 0.00168901
  37 | 0.027032 | 0.00125487
  20 | 0.024102 | 0.00115088
  22 | 0.024102 | 0.00100530
...

next_loto_plus: [1, 3, 17, 19, 33, 34, 35]
"""



"""
BackTest:

Backtest NN_7neurona_v1 forward (n−500):

Loto (2451 → actual 2452)

pred: [1, 3, 17, 19, 33, 34, 35]
actual: [3, 5, 11, 15, 20, 21, 25]
HIT: False
· 1/7 (3)


Loto Plus (1207 → actual 1208)

pred: [1, 3, 17, 19, 33, 34, 35]
actual: [3, 7, 27, 29, 35, 37, 38]
HIT: False
· 2/7 (3, 35)


Backtest NN_7neurona_v1 forward (n−1000):

Loto (1951 → actual 1952)

pred: [1, 3, 17, 19, 33, 34, 35]
actual: [2, 7, 9, 13, 14, 29, 31]
HIT: False
· 0/7 (-)


Loto Plus (707 → actual 708)

pred: [1, 3, 17, 19, 33, 34, 35]
actual: [12, 14, 18, 22, 31, 38, 39]
HIT: False
· 0/7 (-)


Backtest NN_7neurona_v1 forward (n−1500):

Loto (1451 → actual 1452)

pred: [1, 3, 17, 19, 33, 34, 35]
actual: [3, 14, 15, 16, 26, 27, 36]
HIT: False
· 1/7 (3)


Loto Plus (207 → actual 208)

pred: [1, 3, 17, 19, 33, 34, 35]
actual: [2, 12, 15, 16, 19, 21, 27]
HIT: False
· 1/7 (19)
"""



"""
ANALIZA — NN_7neurona_v1.py:

1. Ulaz — dva CSV-a odvojeno: Loto (2951) i Plus (1707), bez miksa.

2. Empirijska distribucija — udeo[b] = p(b) nad {1..39}, suma = 1.
   Nikad sirova frekvencija kao skor.

3. Mreža — 7 neurona, samo forward (bez fit / backprop).
   W (39×7), b (7) fiksni iz SEED=39: W[i,j]=0.3·sin((i+1)(j+1)·39·0.01).
   h = sigmoid(x·W + b); skor(b) = p(b)·(W[b,:]·h).
   next = top 7 po skoru (tie-break manji broj).

4. Izlaz RUN:
   next_loto:      [1, 3, 17, 19, 33, 34, 35]
   next_loto_plus: [1, 3, 17, 19, 33, 34, 35]
   Ista kombinacija na oba CSV — težine fiksne, udeo skoro uniforman,
   pa redosled skorova vodi istih 7 brojeva (33,17,1 dominantni).

5. Backtest (pred n−k → actual n−k+1), k=500,1000,1500:
   Loto: 1/7, 0/7, 1/7
   Plus: 2/7, 0/7, 1/7
   Pred isti na sva 6 mesta (fiksne težine + stabilan udeo).
"""



"""
Beleske:

prosta NN bez fit — 7 neurona, ulaz empirijska distribucija
tezine determinističke SEED=39
dva csv: loto + plus
"""
