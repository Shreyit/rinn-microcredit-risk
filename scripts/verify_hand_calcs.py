"""Cross-check every number in Section II (hand exercises) of the written report.

Run:  .venv/bin/python scripts/verify_hand_calcs.py
"""
import math

import numpy as np


def check(label, got, expected, tol=5e-5):
    ok = abs(got - expected) < tol
    print(f"{'OK  ' if ok else 'FAIL'} {label}: {got:.4f} (report: {expected})")
    return ok


results = []

# Exercise 1: linear score
w, b = np.array([1.5, -0.5, 2.0]), -2.0
loans = {"A": np.array([3, 4, 2]), "B": np.array([1, 5, 0]), "C": np.array([2, 6, 1])}
s0 = {k: float(w @ x + b) for k, x in loans.items()}
results += [check("E1 s_A", s0["A"], 4.5), check("E1 s_B", s0["B"], -3), check("E1 s_C", s0["C"], 0)]

# Exercise 2: one perceptron update on loan A
eta, y_A, yhat_A = 0.5, 0, 1
w1 = w + eta * (y_A - yhat_A) * loans["A"]
b1 = b + eta * (y_A - yhat_A)
results += [check(f"E2 w1[{i}]", w1[i], e) for i, e in enumerate([0, -2.5, 1])]
results += [check("E2 b1", b1, -2.5)]
s1 = {k: float(w1 @ x + b1) for k, x in loans.items()}
results += [check("E2 s_A", s1["A"], -10.5), check("E2 s_B", s1["B"], -15), check("E2 s_C", s1["C"], -16.5)]

# Exercise 3: logistic probabilities
for name, (x1, x2), p_rep in [("P", (2, 2), 0.5000), ("Q", (2, 0), 0.7311), ("R", (0, 0), 0.1192)]:
    z = -2 + 1.5 * x1 - 0.5 * x2
    results.append(check(f"E3 p_{name}", 1 / (1 + math.exp(-z)), p_rep))

# Exercise 4: confusion-matrix metrics
TP, FN, FP, TN = 24, 16, 12, 148
prec, rec = TP / (TP + FP), TP / (TP + FN)
results += [
    check("E4 accuracy", (TP + TN) / 200, 0.8600),
    check("E4 specificity", TN / (TN + FP), 0.9250),
    check("E4 precision", prec, 0.6667),
    check("E4 recall", rec, 0.6000),
    check("E4 F1", 2 * prec * rec / (prec + rec), 0.6316),
    check("E4 baseline accuracy", (FP + TN) / 200, 0.8000),
]

# Exercise 5: KNN distances and votes
borrowers = [((10, 2), 0), ((12, 5), 0), ((8, 9), 1), ((14, 3), 0), ((9, 9), 1), ((6, 4), 1)]
dists = [math.dist(p, (10, 6)) for p, _ in borrowers]
for i, (d, e) in enumerate(zip(dists, [4.0, 2.2361, 3.6056, 5.0, 3.1623, 4.4721]), start=1):
    results.append(check(f"E5 d_{i}", d, e))
order = np.argsort(dists)
for k, e in [(1, 0), (3, 1), (5, 1)]:
    votes = [borrowers[i][1] for i in order[:k]]
    results.append(check(f"E5 K={k} prediction", int(sum(votes) > k / 2), e))

# Exercise 6: naive Bayes
yes = 0.4 * (3 / 4) * (3 / 4)
no = 0.6 * (2 / 6) * (1 / 6)
results += [
    check("E6 unnormalised Yes", yes, 0.2250),
    check("E6 unnormalised No", no, 0.0333),
    check("E6 P(Yes|X)", yes / (yes + no), 0.8710),
    check("E6 P(No|X)", no / (yes + no), 0.1290),
]

print(f"\n{sum(results)}/{len(results)} checks passed")
