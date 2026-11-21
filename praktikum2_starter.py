"""
IVDA 2 - Praktikum 2: Labeling & Active Learning auf Fashion-MNIST
Starter-Skript.

Fest vorgegeben (bitte nicht aendern): Daten, Pool, Cold Start, Modell, Trainingsfunktion.
Alles mit TODO ist eure Aufgabe. Ihr duerft die Struktur frei umbauen.

Aufruf: python praktikum2_starter.py --aufgabe 1      nur Aufgabe 1 (Labeling)
        python praktikum2_starter.py --aufgabe 2      nur Aufgabe 2 (Active Learning)
        python praktikum2_starter.py                  beide (Standard)

"""
import argparse
import csv
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from skactiveml.base import SkactivemlClassifier                       
from skactiveml.pool import (Badge, RandomSampling, k_greedy_center, uncertainty_scores)


CLASSES = ["T-Shirt/Top", "Hose", "Pullover", "Kleid", "Mantel",
           "Sandale", "Hemd", "Sneaker", "Tasche", "Stiefelette"]
UNKLAR = -1                     # unklar; fuer Mehrheit und Dawid-Skene kein Label
MEAN, STD = 0.2860, 0.3530


# ---------------------------------------------------------------------------
# Daten 
# ---------------------------------------------------------------------------
def load_fmnist(root="./data"):
    """Normierte Tensoren: X_train (60000,1,28,28), y_train, X_test (10000,1,28,28), y_test, dazu die Rohbilder
    des Test-Splits."""
    train = datasets.FashionMNIST(root, train=True, download=True)
    test = datasets.FashionMNIST(root, train=False, download=True)
    normalize = lambda d: ((d.float() / 255 - MEAN) / STD).unsqueeze(1)
    return normalize(train.data), train.targets, normalize(test.data), test.targets, test.data


# ---------------------------------------------------------------------------
# Modell (vorgegeben)
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """3 x (Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2) -> Flatten -> FC 64 (Embedding) -> Dropout -> FC 10 (Logits)."""

    def __init__(self, embedding_dim=64, dropout=0.3, num_classes=10):
        super().__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                 nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(
            conv_block(1, 16),     # 28x28x1  -> 14x14x16
            conv_block(16, 32),    # 14x14x16 -> 7x7x32
            conv_block(32, 64),    # 7x7x32   -> 3x3x64
            nn.Flatten())          # -> 576
        self.embed = nn.Sequential(nn.Linear(576, embedding_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, images, return_embedding=False):
        embedding = self.embed(self.features(images))   # vorletzte Schicht = Embedding (64D)
        logits = self.head(self.dropout(embedding))     # Dropout auf das Embedding, dann FC 64->10; Softmax steckt in F.cross_entropy
        return (logits, embedding) if return_embedding else logits


# ---------------------------------------------------------------------------
# Teil A: Labels zusammenfuehren und aggregieren (vorgegeben)
# ---------------------------------------------------------------------------
def merge_label_csvs(paths, out_csv="labels_alle.csv"):
    """Fuehrt mehrere Label-CSVs zu einer Tabelle zusammen: eine Zeile je Bild, eine Spalte je Annotator*in.
    Akzeptiert die ausgefuellte Vorlage (labels_<kuerzel>.csv mit Spalten nr, index, label; Name aus dem
    Dateinamen) und Dateien mit Spalte "annotator" (annotator_labels/).
    Gibt (namen, bildindizes, labels) zurueck; labels: (n_bilder, n_annotatoren), Werte 0-9, -1 = unklar.
    Bricht ab bei leerem oder ungueltigem Label und wenn nicht alle Annotator*innen dieselben Bilder gelabelt haben."""
    ann = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                lab = r.get("label", "").strip().lower()
                if lab not in [str(c) for c in range(-1, 10)]:
                    raise ValueError(f"{p}: ungueltiges oder leeres Label {lab!r} bei index {r['index']} (erlaubt: 0-9, -1)")
                name = r.get("annotator") or (stem[len("labels_"):] if stem.startswith("labels_") else stem)
                ann.setdefault(name, {})[int(r["index"])] = int(lab)
    names = list(ann)
    items = sorted(set().union(*ann.values()))
    for n in names:
        fehlt = [i for i in items if i not in ann[n]]
        if fehlt:
            raise ValueError(f"{n}: keine Labels fuer die Bilder {fehlt[:5]}{' ...' if len(fehlt) > 5 else ''}")
    labels = np.array([[ann[n][i] for n in names] for i in items])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index"] + names)
        for i, row in zip(items, labels):
            w.writerow([i] + list(row))
    return names, items, labels


def dawid_skene(labels, K=10, iters=50, smooth=0.01):
    """Dawid & Skene (1979): schaetzt gleichzeitig das wahre Label jedes Bildes und die Zuverlaessigkeit
    jeder annotierenden Person, ohne Ground Truth, per Expectation-Maximization (EM).

    Modell: Jedes Bild i hat eine unbekannte wahre Klasse k (A-priori-Wahrscheinlichkeit prior[k]).
    Annotator*in j vergibt bei wahrer Klasse k das Label l mit Wahrscheinlichkeit pi[j, k, l]
    (Konfusionsmatrix je Annotator*in). Die Annotator*innen entscheiden unabhaengig voneinander.

    Ablauf:
      Start   T = Stimmanteile je Bild (entspricht einer weichen Mehrheitsentscheidung).
      M-Schritt  prior und pi aus den aktuellen Klassenwahrscheinlichkeiten T schaetzen.
      E-Schritt  T neu berechnen: P(wahre Klasse k | Labels von Bild i)
                 ~ prior[k] * Produkt ueber die Annotator*innen j von pi[j, k, Label von j].
      M- und E-Schritt werden iters-mal wiederholt.

    Parameter:
      labels  (n_bilder, n_annotatoren), Werte 0..K-1; -1 = kein Label ("unklar", siehe merge_label_csvs)
              und wird in allen Schritten ignoriert.
      K       Anzahl Klassen.
      iters   Anzahl EM-Iterationen.
      smooth  Pseudo-Zaehlung je Zelle von pi. Verhindert Wahrscheinlichkeit 0 (log(0)), wenn eine Person
              ein Label bei einer wahren Klasse nie vergeben hat.

    Rueckgabe:
      T   (n_bilder, K): geschaetzte Wahrscheinlichkeit je Klasse; T.argmax(1) = geschaetztes Label.
      pi  (n_annotatoren, K, K): geschaetzte Konfusionsmatrix je Annotator*in,
          Zeile = wahre Klasse, Spalte = vergebenes Label; jede Zeile summiert zu 1."""
    n, J = labels.shape

    # Start: Stimmen je Bild und Klasse zaehlen und zu Anteilen normieren (1e-9 vermeidet Division durch 0,
    # falls ein Bild gar kein Label hat).
    T = np.zeros((n, K))
    for j in range(J):
        has_label = labels[:, j] >= 0                                  # nur vorhandene Labels
        T[has_label, labels[has_label, j]] += 1
    T = (T + 1e-9) / (T + 1e-9).sum(1, keepdims=True)

    for _ in range(iters):
        # M-Schritt: Klassenhaeufigkeit und Konfusionsmatrizen aus den weichen Labels T schaetzen.
        prior = T.mean(0)                                  # erwarteter Anteil jeder Klasse
        pi = np.full((J, K, K), smooth)
        for j in range(J):
            has_label = labels[:, j] >= 0
            for k in range(K):
                # Spalte k: erwartete Anzahl Bilder je wahrer Klasse, denen Annotator*in j das Label k gab
                pi[j][:, k] += (T[has_label] * (labels[has_label, j] == k)[:, None]).sum(0)
            pi[j] /= pi[j].sum(1, keepdims=True)           # Zeilen normieren: P(Label | wahre Klasse)

        # E-Schritt: Posterior der wahren Klasse je Bild, im Log-Raum gegen numerischen Unterlauf.
        logT = np.log(prior)[None, :].repeat(n, 0)
        for j in range(J):
            has_label = labels[:, j] >= 0
            logT[has_label] += np.log(pi[j][:, labels[has_label, j]]).T       # log P(Label von j | wahre Klasse k) fuer alle k
        T = np.exp(logT - logT.max(1, keepdims=True))      # stabil exponieren
        T /= T.sum(1, keepdims=True)                       # je Bild zu Wahrscheinlichkeiten normieren
    return T, pi


def show_images(raw, ids, titles=None, path=None, ncols=6):
    """Zeigt Bilder des Test-Splits als Raster.
    raw: Rohbilder (z. B. raw_test aus load_fmnist), ids: Bildindizes im Test-Split,
    titles (optional): ein Titel je Bild, z. B. offizielles Label und eure Labels.
    path gesetzt: Abbildung als Datei speichern, sonst anzeigen."""
    import matplotlib.pyplot as plt
    ids = list(ids)
    titles = titles or [f"#{i}" for i in ids]
    nrows = max(1, -(-len(ids) // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i, t in zip(axes.flat, ids, titles):
        ax.imshow(raw[i].numpy(), cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=7)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # Ausgabeordner anlegen, falls er fehlt
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Teil B: Active Learning (vorgegeben)
# ---------------------------------------------------------------------------

POOL_SIZE, INIT_PER_CLASS, BATCH, ROUNDS, STEPS = 10_000, 5, 20, 10, 200
SEEDS = tuple(range(5))           # 0-4 fuer alle Strategien (gepaarter Vergleich)


def make_pool(n_train=60_000):
    """Fester Pool: 10.000 Bilder aus dem offiziellen Train-Split (fuer alle Gruppen gleich)."""
    return np.random.default_rng(0).choice(n_train, POOL_SIZE, replace=False)


def cold_start(y_pool, rng):
    """Zieht 5 Bilder je Klasse aus dem Pool (Indizes relativ zum Pool). rng = np.random.default_rng(seed)."""
    return np.concatenate([rng.choice(np.where(y_pool == c)[0], INIT_PER_CLASS, replace=False) for c in range(10)])


def train_model(X, y, idx, steps=STEPS, seed=0, batch_size=64, lr=1e-3, weight_decay=0.01):
    """Trainiert ein neues SmallCNN von Grund auf auf den gelabelten Bildern und gibt es im eval-Modus zurueck.

    X, y        Bilder und Labels des Pools (Tensoren)
    idx         Pool-Indizes der gelabelten Bilder; nur diese werden zum Training benutzt
    steps       feste Zahl an Gradientenschritten (200), unabhaengig davon, wie viele Bilder gelabelt sind.
                Es gibt keine Epochen, keinen Validierungs-Split und kein Early Stopping: Val-Labels wuerden
                Budget kosten. Bewertet wird das Modell nach dem letzten Schritt.
    seed        legt Startgewichte, Dropout-Masken und die Reihenfolge der Batches fest
                (im Loop: 1000 * Seed des Laufs + Runde)."""
    torch.manual_seed(seed)                     # Startgewichte und Dropout reproduzierbar
    model = SmallCNN()                          # immer neu: nichts aus der vorigen Runde uebernehmen
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)   # Lernrate konstant
    rng = np.random.default_rng(seed)           # eigener Generator fuer die Auswahl der Batches
    model.train()                               # Dropout an, BatchNorm nutzt die Statistik des Batches
    for _ in range(steps):
        # Je Schritt 64 gelabelte Bilder ohne Zuruecklegen ziehen, unabhaengig von den vorigen Schritten; bei weniger als 64 gelabelten Bildern alle.
        batch = torch.as_tensor(rng.choice(idx, min(batch_size, len(idx)), replace=False))
        loss = F.cross_entropy(model(X[batch]), y[batch])
        opt.zero_grad()      # alte Gradienten loeschen (PyTorch addiert sie sonst auf)
        loss.backward()      # berechnet per Backpropagation den Gradienten
        opt.step()           # Parameter mit AdamW aktualisieren        
    return model.eval()                         

@torch.no_grad()
def predict(model, X, batch_size=2000):
    """Gibt (probs, emb) zurueck: Wahrscheinlichkeiten (n, 10) und Embeddings h(x) (n, 64)."""
    probs, emb = [], []
    for start in range(0, len(X), batch_size):
        logits, z = model(X[start:start + batch_size], return_embedding=True)
        probs.append(F.softmax(logits, 1))
        emb.append(z)
    return torch.cat(probs).numpy(), torch.cat(emb).numpy()


def y_masked(y_pool, labeled):
    """scikit-activeml-Konvention: Labels nur fuer die gelabelten Bilder (labeled = Pool-Indizes), sonst NaN.
    scikit-activeml kennt keine getrennten Listen fuer gelabelte und ungelabelte Daten. Stattdessen bekommt
    jede Strategie den ganzen Pool X und dazu einen Vektor y gleicher Laenge, in dem fehlende Labels mit NaN
    markiert sind. Die Strategien waehlen nur unter den Bildern mit NaN aus;
    """
    y = np.full(len(y_pool), np.nan)
    y[labeled] = y_pool[labeled]
    return y


class FixedOutputs(SkactivemlClassifier):
    """Wrapper fuer skactiveml.pool.Badge. Badge erwartet einen scikit-activeml-Klassifikator und ruft intern
    clf.predict_proba(X, return_embeddings=True) auf, um Wahrscheinlichkeiten und Embeddings h(x) zu bekommen.
    Das SmallCNN ist kein solcher Klassifikator. Deshalb rechnet ihr probs und emb wie sonst mit predict() und
    uebergebt sie als Spalten von X; FixedOutputs gibt sie unveraendert zurueck.

        badge = Badge(clf_embedding_flag_name="return_embeddings", random_state=seed)
        new = badge.query(np.hstack([emb, probs]), y, clf=FixedOutputs(), fit_clf=False, batch_size=batch_size)

    - np.hstack([emb, probs]): je Bild erst die 64 Embedding-Werte, dann die 10 Wahrscheinlichkeiten
    - clf_embedding_flag_name="return_embeddings": Name des Arguments, mit dem predict_proba die Embeddings liefert
    - fit_clf=False: das Modell ist schon trainiert, Badge soll es nicht neu anpassen
    - y = y_masked(...): NaN = ungelabelt; Badge waehlt nur unter diesen Bildern"""

    def __init__(self, emb_dim=64, classes=np.arange(10), missing_label=np.nan):
        super().__init__(classes=classes, missing_label=missing_label)
        self.emb_dim = emb_dim

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X, return_embeddings=False):
        probs, emb = X[:, self.emb_dim:], X[:, :self.emb_dim]
        return (probs, emb) if return_embeddings else probs

"""
 Strategien. Alle bekommen dieselben Argumente:
   unlabeled        ungelabelte Pool-Indizes 
   batch_size       Zahl der anzufragenden Bilder 
   probs, emb       Vorhersagen (n_pool, 10) und Embeddings h(x) (n_pool, 64) des aktuellen Modells auf dem Pool
   y                y_masked(...): Label der gelabelten Bilder, sonst NaN
   seed             Seed fuer die Bibliothek
   
 Rueckgabe: batch_size Indizes in den Pool (Werte zwischen 0 und 9999).
 Achtung: Berechnet ihr Scores nur fuer probs[unlabeled], gehoert scores[i] zum Pool-Bild unlabeled[i].
 Die besten Positionen top = np.argsort(-scores)[:batch_size] muesst ihr deshalb noch umrechnen:
 Pool-Indizes = unlabeled[top].
 Nutzt die Funktionen aus scikit-activeml (https://pypi.org/project/scikit-activeml/).
"""
def select_random(unlabeled, batch_size, emb, y, seed, **_):
    return np.asarray(RandomSampling(random_state=seed).query(emb, y, batch_size=batch_size))


def select_entropy(unlabeled, batch_size, probs, **_):
    raise NotImplementedError("TODO: skactiveml.pool.uncertainty_scores(..., method='entropy')")


def select_coreset(unlabeled, batch_size, emb, y, seed, **_):
    raise NotImplementedError("TODO: skactiveml.pool.k_greedy_center")


def select_badge(unlabeled, batch_size, probs, emb, y, seed, **_):
    raise NotImplementedError("TODO: skactiveml.pool.Badge mit FixedOutputs (siehe Docstring)")


STRATEGIES = {"Random": select_random, "Entropy": select_entropy, "Core-Set": select_coreset,
              "BADGE": select_badge}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--aufgabe", choices=["1", "2", "alle"], default="alle",
                        help="nur Aufgabe 1, nur Aufgabe 2 oder beide ausfuehren")
    parser.add_argument("--plot_only", action="store_true",
                        help="Aufgabe 2 nur aus den gespeicherten JSON-Dateien auswerten, ohne Training")
    args = parser.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    X_train, y_train, X_test, y_test, raw_test = load_fmnist(args.data)

    # --- Aufgabe 1: Labeling ------------------------------------------------
    if args.aufgabe in ("1", "alle"):
        # TODO: Aufgabe 1 (siehe Aufgabenblatt). y_test ist ein Tensor: zum Vergleich y_test.numpy()[items] nutzen.
        #       Annotator*innen-CSVs mit sorted(glob("annotator_labels/*.csv")) einlesen, damit die Spalten immer
        #       in derselben Reihenfolge stehen.
        pass

    # --- Aufgabe 2: Active Learning -----------------------------------------
    if args.aufgabe in ("2", "alle"):
        pool = make_pool(len(X_train))
        X_pool, y_pool = X_train[pool], y_train[pool]
        y_pool_np = y_pool.numpy()
        for name, strategy in STRATEGIES.items():
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                labeled = cold_start(y_pool_np, rng)                             # gelabelte Pool-Indizes
                for r in range(ROUNDS + 1):
                    model = train_model(X_pool, y_pool, labeled, seed=seed * 1000 + r)  # TODO: Zeit messen (time.perf_counter())
                    probs, emb = predict(model, X_pool)
                    # TODO: Test-Accuracy auf X_test/y_test.numpy(); nach der letzten Runde F1 je Klasse (sklearn.metrics.f1_score)
                    if r == ROUNDS:
                        break
                    unlabeled = np.setdiff1d(np.arange(len(pool)), labeled)     # ungelabelte Pool-Indizes
                    new = strategy(unlabeled=unlabeled, batch_size=BATCH, probs=probs, emb=emb,
                                   y=y_masked(y_pool_np, labeled), seed=seed * 1000 + r)
                    # TODO: gewaehlte Indizes (Klassen: y_pool_np[new]) und Zeit der Auswahl festhalten
                    new = np.asarray(new, dtype=int)
                    assert len(np.unique(new)) == BATCH and np.isin(new, unlabeled).all(), \
                        f"{name}: {BATCH} verschiedene Indizes aus unlabeled erwartet"
                    labeled = np.concatenate([labeled, new])
                # TODO: Ergebnisse dieses Laufs als JSON speichern (Inhalt siehe Aufgabenblatt)
        # TODO: mit --plot_only (args.plot_only) den Loop oben ueberspringen; alle Abbildungen und Kennzahlen
        #       entstehen dann nur aus den JSON-Dateien (im Testat wird das AL nicht erneut ausgefuehrt)
        # TODO: Lernkurven (Mittel, Min/Max ueber Seeds), Label-Effizienz, F1 je Klasse, Klassenverteilung,
        #       UMAP des Pools mit den gewaehlten Bildern etc.
