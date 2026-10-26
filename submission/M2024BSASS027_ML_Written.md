# Introduction to Machine Learning

**Written Component: Sections I (Short Answers) and II (Hand Exercises)**

Tata Institute of Social Sciences, School of Analytics

| | |
|---|---|
| **Submitted by** | Shreyit Sinha |
| **Roll number** | M2024BSASS027 |
| **Version ID** | 20AA1102C3 |
| **Programme** | BS Analytics (Batch 2024-2028), Semester V |
| **Submitted to** | Dr Purushottam Bhandare |
| **Date of submission** | 26 October 2026 |
| **Section I answered** | Questions 1, 2, 3, 4 and 5 |

**AI-use declaration.** Claude (Anthropic) was used in this assignment. **Sections I and II:** to typeset the document in LaTeX and Markdown (KaTeX), format the equations and tables, and edit the wording of my answers for clarity and simplicity. **Section III:** for help with the Python code, error messages and explanations of the script/syntax used.

**Note on formulas.** Mathematical expressions in this file are written in LaTeX syntax for KaTeX rendering (`$...$` inline, `$$...$$` display). They display correctly in any KaTeX or MathJax viewer, such as VS Code Markdown preview, GitHub or Typora. A typeset PDF version of the same content is submitted alongside this file.

---

## Contents

- [Section I: Short-Answer Questions](#section-i-short-answer-questions)
  - [Question 1: Linear classifiers and the perceptron](#question-1-linear-classifiers-and-the-perceptron)
  - [Question 2: Probabilities, classes and thresholds](#question-2-probabilities-classes-and-thresholds)
  - [Question 3: Evaluation under class imbalance](#question-3-evaluation-under-class-imbalance)
  - [Question 4: KNN and decision trees](#question-4-knn-and-decision-trees)
  - [Question 5: Margins and support vectors](#question-5-margins-and-support-vectors)
- [Section II: Hand Exercises](#section-ii-hand-exercises)
- [Notation](#notation)
- [References](#references)

---

# Section I: Short-Answer Questions

**Questions answered:** 1, 2, 3, 4 and 5. Each answer is at most 300 words. Equations and tables are excluded from the count.

## Question 1: Linear classifiers and the perceptron

*Setting: a cooperative bank flags a loan as risky when $s = \mathbf{w}\cdot\mathbf{x} + b \geq 0$.*

### 1.1 Random linear classifier (2 marks)

> *Why might a random linear classifier look acceptable on a handful of loans yet be unreliable? Name one thing you would do to judge it fairly.*

A random $\mathbf{w}$ and $b$ have learned nothing from the data, but on a handful of loans they can look good by luck. With few points, a random boundary may happen to put most of them on the correct side. If most sampled loans are safe, a boundary that calls almost everything safe also scores a high accuracy. Such results come from chance, not from a real pattern, so they will not repeat on new loans.

To judge it fairly, I would test it on a large held-out set of loans that were not used to choose $\mathbf{w}$ and $b$, compare it with a majority-class baseline, and report recall and precision, not only accuracy.

### 1.2 Perceptron update (2 marks)

> *What does the perceptron do to $\mathbf{w}$ and $b$ on a misclassified loan, and why does this move the boundary helpfully?*

The perceptron changes $\mathbf{w}$ and $b$ only when it gets a loan wrong. With labels $y, \hat y \in \{0, 1\}$ and learning rate $\eta > 0$:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat y)\,\mathbf{x}, \qquad b \leftarrow b + \eta\,(y - \hat y)
$$

If a risky loan ($y = 1$) is wrongly called safe ($\hat y = 0$), it adds $\eta\mathbf{x}$ to $\mathbf{w}$ and $\eta$ to $b$. If a safe loan is wrongly called risky, it subtracts them.

This helps because the new score of that loan moves towards the correct side:

$$
s_{\text{new}} = (\mathbf{w} + \eta\mathbf{x})\cdot\mathbf{x} + (b + \eta) = s + \eta\left(\lVert\mathbf{x}\rVert^2 + 1\right)
$$

Since $\lVert\mathbf{x}\rVert^2 + 1 > 0$, a missed risky loan's score always rises and a wrongly flagged safe loan's score always falls. The boundary $\mathbf{w}\cdot\mathbf{x} + b = 0$ shifts towards putting this loan on its correct side.

### 1.3 When the perceptron never stops (2 marks)

> *Describe a situation in which a perceptron would never stop updating, and explain why.*

This happens when the training data are **not linearly separable**: no straight boundary can put every risky loan on one side and every safe loan on the other. For example, two loans with identical features where one defaulted and one was repaid.

The perceptron stops only after a full pass with zero mistakes. If the data cannot be separated, some loan is always misclassified, so every pass triggers another update and the weights keep moving for ever. In practice we cap the number of passes, or use a method that accepts errors, such as logistic regression or a soft-margin SVM.

## Question 2: Probabilities, classes and thresholds

*Setting: logistic regression predicts the probability that a patient misses a follow-up visit.*

### 2.1 Probability, class and threshold (2 marks)

> *Distinguish a predicted probability from a predicted class, and explain the role of the classification threshold.*

The **predicted probability** $\hat p$ is a number between 0 and 1 giving the model's estimate of how likely the patient is to miss the visit, for example $\hat p = 0.73$. It comes from the sigmoid function, $\hat p = 1/(1 + e^{-z})$.

The **predicted class** $\hat y$ is a yes or no decision: $1$ means "will miss" and $0$ means "will attend".

The threshold $t$ turns the probability into the class:

$$
\hat y =
\begin{cases}
1 & \text{if } \hat p \geq t \\
0 & \text{if } \hat p < t
\end{cases}
$$

The threshold is a policy choice, not learned from the data. A lower $t$ flags more patients, missing fewer real cases but raising more false alarms. A higher $t$ flags fewer.

### 2.2 Patients P and Q (2 marks)

> *Patients P ($\hat p = 0.51$) and Q ($\hat p = 0.97$) are both flagged at threshold 0.5. Is it reasonable to treat them identically? Defend a position.*

**Position: no, they should not be treated identically.**

Both cross the threshold, but their risks differ greatly. About half of patients like P would miss the visit, while almost all patients like Q would. The binary flag hides this.

Staff time is limited. Giving both the same costly action, such as a home visit, spreads effort thinly. A tiered response is better: Q gets a phone call and a home visit, while P gets a low-cost SMS reminder.

If the only action is cheap, such as one SMS for everyone flagged, identical treatment would also be reasonable.

### 2.3 Thresholds below and above 0.5 (2 marks)

> *Describe one setting for a threshold below 0.5 and one for a threshold above 0.5, with reasons.*

**Below 0.5 (for example $t = 0.3$).** Use this when missing a real case is costly and the action is cheap. For tuberculosis patients, a missed follow-up can cause treatment failure, while an SMS reminder costs almost nothing. A low threshold catches more patients who will miss (higher recall), and the extra false alarms only cost a few messages.

**Above 0.5 (for example $t = 0.8$).** Use this when the action is expensive or limited. With only a few health workers for home visits, a high threshold sends them only to patients very likely to miss (higher precision), so their time is not wasted.

## Question 3: Evaluation under class imbalance

*Setting: about 2 in 100 households are eligible but unregistered, and a vendor claims "98% accuracy".*

### 3.1 Why 98% accuracy is not enough (2 marks)

> *Why does 98% accuracy not show the model is useful? Use a trivial classifier in your explanation.*

Take a trivial classifier that predicts "not eligible" for every household. In a group of 100 households it is right for the 98 that are not eligible and wrong for the 2 that are:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{0 + 98}{100} = 0.98
$$

So 98% accuracy is what you get by doing nothing. This classifier finds none of the eligible households:

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{0}{0 + 2} = 0
$$

Because eligible households are so rare, accuracy is dominated by the easy majority and says nothing about whether the model finds the households the scheme is looking for.

### 3.2 Two better metrics (2 marks)

> *Which two metrics would you request instead, and which question does each answer in this scheme?*

**Recall**, $\text{TP}/(\text{TP} + \text{FN})$. It answers: "Of all the eligible households that are not yet registered, how many does the model find?" Low recall means families who need food support are left out.

**Precision**, $\text{TP}/(\text{TP} + \text{FP})$. It answers: "Of the households the model flags, how many are really eligible?" Low precision means field staff waste visits verifying households that are not eligible.

### 3.3 ROC-AUC of 0.5 (2 marks)

> *Explain in plain language what a ROC-AUC of 0.5 means.*

ROC-AUC measures how well the model ranks households. If we pick one eligible household and one non-eligible household at random, the AUC is the chance that the model gives the eligible one the higher score.

An AUC of 0.5 means this happens only half the time, which is the same as tossing a coin. The model cannot tell eligible households apart from the rest. Changing the threshold will not help, because the scores themselves carry no useful information.

## Question 4: KNN and decision trees

*Setting: predicting seed adoption from land size, distance to market, rainfall and schooling.*

### 4.1 Rainfall in metres instead of millimetres (2 marks)

> *If rainfall were recorded in metres instead of millimetres, what happens to KNN and decision-tree predictions, and why do they differ?*

**KNN: predictions change.** KNN finds neighbours using a distance over all features:

$$
d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_j (u_j - v_j)^2}
$$

In millimetres, a rainfall gap such as $1200 - 800 = 400$ dominates the gaps in land size or schooling. In metres it becomes $1.2 - 0.8 = 0.4$ and almost disappears. Different farmers become the nearest neighbours, so the predictions change.

**Decision tree: predictions stay the same.** A tree splits on one feature at a time, such as "rainfall $\leq 900$ mm", which simply becomes "rainfall $\leq 0.9$ m". The same farmers fall on each side, so the tree and its predictions are unchanged.

**Why they differ:** KNN mixes all features into one distance, so units matter. A tree compares each feature only with itself, so only the order of values matters.

### 4.2 Overfitting in KNN and trees (2 marks)

> *How does overfitting show up in KNN (as K changes) and in a decision tree (as depth changes)?*

**KNN:** overfitting appears when $K$ is very small. With $K = 1$ each prediction copies the single nearest farmer, noise included, so training accuracy is perfect but accuracy on new farmers is lower. A larger $K$ smooths the predictions, and a very large $K$ underfits by predicting the majority class.

**Decision tree:** overfitting appears when the tree is very deep. Leaves end up holding one or a few farmers, so the tree memorises the training data. Training accuracy keeps rising while validation accuracy falls. A very shallow tree underfits.

In both, overfitting shows as a large gap between training and validation performance.

### 4.3 Model for a panchayat committee (2 marks)

> *For a panchayat committee, which model would you choose and what would you give up?*

**Choice: a decision tree.**

A tree can be shown as a few if-then rules, for example: "If land size is above 2 acres and the market is within 5 km, the farmer is likely to adopt." Committee members can follow, question and check each rule against local knowledge. KNN can only say "similar farmers adopted", which hides which factors matter.

**What I give up:** a readable tree must be shallow, so it may be less accurate than KNN. Trees also draw step-like boundaries and can change a lot with small data changes.

## Question 5: Margins and support vectors

*Setting: one feature $x$ (years of schooling). Class $-1$ at $x \in \{2, 5\}$ and class $+1$ at $x \in \{8, 10\}$. Hard margin.*

### 5.1 Boundary, margin and support vectors (2 marks)

> *State the maximum-margin boundary, the total margin width, and the support vectors.*

The closest points from the two classes are $x = 5$ (class $-1$) and $x = 8$ (class $+1$). The maximum-margin boundary lies halfway between them:

$$
x^{*} = \frac{5 + 8}{2} = 6.5, \qquad \text{total margin width} = 8 - 5 = 3
$$

The support vectors are $x = 5$ and $x = 8$. The rule is: predict $+1$ if $x > 6.5$, otherwise $-1$.

### 5.2 Adding a class −1 point at x = 7 (2 marks)

> *A class −1 point is added at x = 7. Give the new boundary and support vectors. What does this show?*

The closest opposite pair is now $x = 7$ (class $-1$) and $x = 8$ (class $+1$):

$$
x^{*} = \frac{7 + 8}{2} = 7.5, \qquad \text{total margin width} = 8 - 7 = 1
$$

The new support vectors are $x = 7$ and $x = 8$. The point $x = 5$ is no longer a support vector.

This shows that the boundary depends only on the points closest to it (the support vectors). Points far from the boundary, such as $x = 2$ and $x = 10$, can move or be removed without changing the boundary. A single new point near the boundary can change it a lot.

### 5.3 A non-separable point at x = 9 (2 marks)

> *Instead, a class −1 point is added at x = 9 (non-separable). How do small and large C treat it, and which do you prefer?*

The point $x = 9$ lies between the class $+1$ points at 8 and 10, so no single cut-off can classify every point correctly. A soft-margin SVM allows violations and charges a penalty of $C$ for each unit of violation.

**Small $C$:** violations are cheap. The SVM keeps a wide margin decided by the main groups and simply accepts $x = 9$ as an error, treating it like an outlier.

**Large $C$:** violations are expensive, so the SVM tries hard to reduce the error at $x = 9$ and becomes sensitive to this one household. Here, no cut-off can put 9 on the $-1$ side without also misclassifying the $+1$ point at 8, so a large $C$ cannot remove the error. It only makes the solution depend more on a single unusual point, which risks overfitting.

**Preference: a small $C$.** One unusual household should not decide the rule for everyone, and a wider margin usually generalises better to new households. A larger $C$ would make sense only if we knew that households like $x = 9$ form a real, important group.

---

# Section II: Hand Exercises

All six exercises are answered. Working is shown as *formula → substitution → result*, and final values are given to four decimal places unless exact. Every numerical result was cross-checked in Python with `scripts/verify_hand_calcs.py`.

## Exercise 1: Dot product and linear score

Given $\mathbf{w} = (1.5, -0.5, 2)$ and $b = -2$. Predict class 1 ("likely to default") if $s \geq 0$, otherwise class 0.

| Loan | $x_1$ (loan, INR ten-thousands) | $x_2$ (income, INR thousands) | $x_3$ (distance, km) |
|:---:|:---:|:---:|:---:|
| A | 3 | 4 | 2 |
| B | 1 | 5 | 0 |
| C | 2 | 6 | 1 |

### E1.1 Scores for A, B and C (1.5 marks)

Formula: $s = \mathbf{w}\cdot\mathbf{x} + b = w_1x_1 + w_2x_2 + w_3x_3 + b$

$$
\begin{aligned}
s_A &= (1.5)(3) + (-0.5)(4) + (2)(2) - 2 = 4.5 - 2 + 4 - 2 = \boxed{4.5} \\
s_B &= (1.5)(1) + (-0.5)(5) + (2)(0) - 2 = 1.5 - 2.5 + 0 - 2 = \boxed{-3} \\
s_C &= (1.5)(2) + (-0.5)(6) + (2)(1) - 2 = 3 - 3 + 2 - 2 = \boxed{0}
\end{aligned}
$$

### E1.2 Predicted classes (1 mark)

Rule: $\hat y = 1$ if $s \geq 0$, otherwise $\hat y = 0$.

| Loan | $s$ | Check | Predicted class |
|:---:|:---:|:---:|:---|
| A | $4.5$ | $4.5 \geq 0$ | $\hat y_A = 1$ (likely to default) |
| B | $-3$ | $-3 < 0$ | $\hat y_B = 0$ (repaid) |
| C | $0$ | $0 \geq 0$ | $\hat y_C = 1$ (likely to default) |

### E1.3 Loan on the boundary (0.5 marks)

**Loan C.** The decision boundary is the set of points where $\mathbf{w}\cdot\mathbf{x} + b = 0$, and loan C has $s_C = 0$ exactly. It is classified as 1 only because the convention uses $s \geq 0$.

## Exercise 2: A perceptron update

Start from $\mathbf{w}^{(0)} = (1.5, -0.5, 2)$ and $b^{(0)} = -2$. True labels: A $= 0$, B $= 1$, C $= 1$. Learning rate $\eta = 0.5$, with

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat y)\,\mathbf{x}, \qquad b \leftarrow b + \eta\,(y - \hat y)
$$

### E2.1 Misclassified loans (1 mark)

Using the predictions from E1.2:

| Loan | True $y$ | Predicted $\hat y$ | Result |
|:---:|:---:|:---:|:---|
| A | 0 | 1 | misclassified (false positive) |
| B | 1 | 0 | misclassified (false negative) |
| C | 1 | 1 | correct |

$$
\boxed{\text{Loans A and B are misclassified.}}
$$

### E2.2 Single update on loan A (1.5 marks)

For loan A: $\mathbf{x}_A = (3, 4, 2)$, $y_A = 0$, $\hat y_A = 1$, so $y_A - \hat y_A = 0 - 1 = -1$ and $\eta\,(y_A - \hat y_A) = 0.5 \times (-1) = -0.5$.

$$
\begin{aligned}
\mathbf{w}^{(1)} &= \mathbf{w}^{(0)} + \eta\,(y_A - \hat y_A)\,\mathbf{x}_A = (1.5, -0.5, 2) - 0.5\,(3, 4, 2) \\
&= (1.5, -0.5, 2) - (1.5, 2, 1) = \boxed{(0, -2.5, 1)} \\[6pt]
b^{(1)} &= b^{(0)} + \eta\,(y_A - \hat y_A) = -2 + (-0.5) = \boxed{-2.5}
\end{aligned}
$$

### E2.3 Recomputed scores and what they show (1.5 marks)

With $\mathbf{w}^{(1)} = (0, -2.5, 1)$ and $b^{(1)} = -2.5$:

$$
\begin{aligned}
s_A^{(1)} &= (0)(3) + (-2.5)(4) + (1)(2) - 2.5 = 0 - 10 + 2 - 2.5 = -10.5 \\
s_B^{(1)} &= (0)(1) + (-2.5)(5) + (1)(0) - 2.5 = 0 - 12.5 + 0 - 2.5 = -15 \\
s_C^{(1)} &= (0)(2) + (-2.5)(6) + (1)(1) - 2.5 = 0 - 15 + 1 - 2.5 = -16.5
\end{aligned}
$$

| Loan | True $y$ | $s^{(0)}$ | $\hat y^{(0)}$ | $s^{(1)}$ | $\hat y^{(1)}$ | Change |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| A | 0 | $4.5$ | 1 | $-10.5$ | 0 | now correct |
| B | 1 | $-3$ | 0 | $-15$ | 0 | still wrong |
| C | 1 | $0$ | 1 | $-16.5$ | 0 | now wrong |

**Observation.** The update fixed loan A, but loan B is still wrong and loan C, which was correct before, is now wrong. All three scores fell sharply because the update cut $w_2$ from $-0.5$ to $-2.5$ and lowered $b$.

**Lesson.** One perceptron update only looks at one loan. It moves the boundary to help that loan and can make other loans worse, so the number of errors can go up after an update. The perceptron needs many passes over all the loans. It is guaranteed to stop only if the data are linearly separable, and even then the path to that point is not smooth.

## Exercise 3: From score to probability

Log-odds $z = -2 + 1.5\,x_1 - 0.5\,x_2$ and probability $p = \sigma(z) = \dfrac{1}{1 + e^{-z}}$. Given $e^{-1} = 0.3679$, $e^{2} = 7.3891$ and $e^{0} = 1$.

| Borrower | $x_1$ (missed repayments) | $x_2$ (years in savings group) |
|:---:|:---:|:---:|
| P | 2 | 2 |
| Q | 2 | 0 |
| R | 0 | 0 |

### E3.1 Values of z and p (2 marks)

$$
\begin{aligned}
z_P &= -2 + 1.5(2) - 0.5(2) = -2 + 3 - 1 = 0, & p_P &= \frac{1}{1 + e^{0}} = \frac{1}{1 + 1} = \boxed{0.5000} \\[6pt]
z_Q &= -2 + 1.5(2) - 0.5(0) = -2 + 3 - 0 = 1, & p_Q &= \frac{1}{1 + e^{-1}} = \frac{1}{1 + 0.3679} = \frac{1}{1.3679} = \boxed{0.7311} \\[6pt]
z_R &= -2 + 1.5(0) - 0.5(0) = -2, & p_R &= \frac{1}{1 + e^{2}} = \frac{1}{1 + 7.3891} = \frac{1}{8.3891} = \boxed{0.1192}
\end{aligned}
$$

### E3.2 Predicted classes at thresholds 0.5 and 0.7 (1 mark)

Rule: predict default (1) if $p \geq t$, otherwise 0.

| Borrower | $p$ | $t = 0.5$ | $t = 0.7$ |
|:---:|:---:|:---:|:---:|
| P | $0.5000$ | 1 ($0.5000 \geq 0.5$) | 0 ($0.5000 < 0.7$) |
| Q | $0.7311$ | 1 ($0.7311 \geq 0.5$) | 1 ($0.7311 \geq 0.7$) |
| R | $0.1192$ | 0 ($0.1192 < 0.5$) | 0 ($0.1192 < 0.7$) |

Raising the threshold from 0.5 to 0.7 changes only P's prediction, from default to not default.

### E3.3 Meaning of z = 0 (1 mark)

**Why $p = 0.5$:** substituting $z = 0$ gives

$$
p = \frac{1}{1 + e^{-0}} = \frac{1}{1 + 1} = 0.5
$$

Since $z$ is the log-odds, $z = \ln\big(p/(1 - p)\big) = 0$ means the odds are $1$, so default and repayment are equally likely.

**Meaning for the boundary:** with the usual threshold 0.5, the points where $z = 0$ form the decision boundary:

$$
-2 + 1.5\,x_1 - 0.5\,x_2 = 0
$$

This is a straight line in the $(x_1, x_2)$ plane. Borrowers with $z > 0$ are predicted to default and those with $z < 0$ are not. Borrower P has $z_P = 0$, so P lies exactly on this boundary.

## Exercise 4: Metrics from a confusion matrix

| | Predicted default | Predicted repaid |
|:---|:---:|:---:|
| **Actually defaulted** | TP $= 24$ | FN $= 16$ |
| **Actually repaid** | FP $= 12$ | TN $= 148$ |

Total $= 24 + 16 + 12 + 148 = 200$. Actual defaults $= 24 + 16 = 40$, actual repaid $= 12 + 148 = 160$.

### E4.1 Accuracy and specificity (1 mark)

$$
\begin{aligned}
\text{Accuracy} &= \frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{24 + 148}{200} = \frac{172}{200} = \boxed{0.8600} \\[6pt]
\text{Specificity} &= \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{148}{148 + 12} = \frac{148}{160} = \boxed{0.9250}
\end{aligned}
$$

### E4.2 Precision and recall (1 mark)

$$
\begin{aligned}
\text{Precision} &= \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{24}{24 + 12} = \frac{24}{36} = \boxed{0.6667} \\[6pt]
\text{Recall} &= \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{24}{24 + 16} = \frac{24}{40} = \boxed{0.6000}
\end{aligned}
$$

### E4.3 F1-score (1 mark)

$$
F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.6667 \times 0.6000}{0.6667 + 0.6000} = \frac{0.8000}{1.2667} = \boxed{0.6316}
$$

Check with counts: $F_1 = \dfrac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}} = \dfrac{48}{48 + 12 + 16} = \dfrac{48}{76} = 0.6316$.

### E4.4 Majority-class baseline (2 marks)

The baseline predicts "repaid" for all 200 applicants, so it is right for the 160 who repaid:

$$
\text{Baseline accuracy} = \frac{160}{200} = \boxed{0.8000}, \qquad \text{Baseline recall} = \frac{0}{40} = 0
$$

The model's accuracy is $0.8600$, only $0.06$ higher than the baseline.

**What the comparison tells us:** the model does better than simply guessing the majority class, so it has learned something from the features.

**What it does not tell us:** accuracy hides which errors are made. The baseline catches none of the 40 defaulters, while the model catches 24 of them (recall 0.6000). Accuracy also does not show the 16 defaulters the model still misses, which are the costly errors for a lender.

## Exercise 5: KNN prediction

New applicant $(x_1, x_2) = (10, 6)$. Euclidean distance and majority vote.

### E5.1 Distances (2 marks)

Formula: $d = \sqrt{(x_1 - 10)^2 + (x_2 - 6)^2}$

| Borrower | $(x_1, x_2)$ | Label | Substitution | Exact | Distance |
|:---:|:---:|:---:|:---|:---:|:---:|
| 1 | $(10, 2)$ | 0 | $\sqrt{(10-10)^2 + (2-6)^2} = \sqrt{0 + 16}$ | $\sqrt{16}$ | $4.0000$ |
| 2 | $(12, 5)$ | 0 | $\sqrt{(12-10)^2 + (5-6)^2} = \sqrt{4 + 1}$ | $\sqrt{5}$ | $2.2361$ |
| 3 | $(8, 9)$ | 1 | $\sqrt{(8-10)^2 + (9-6)^2} = \sqrt{4 + 9}$ | $\sqrt{13}$ | $3.6056$ |
| 4 | $(14, 3)$ | 0 | $\sqrt{(14-10)^2 + (3-6)^2} = \sqrt{16 + 9}$ | $\sqrt{25}$ | $5.0000$ |
| 5 | $(9, 9)$ | 1 | $\sqrt{(9-10)^2 + (9-6)^2} = \sqrt{1 + 9}$ | $\sqrt{10}$ | $3.1623$ |
| 6 | $(6, 4)$ | 1 | $\sqrt{(6-10)^2 + (4-6)^2} = \sqrt{16 + 4}$ | $\sqrt{20}$ | $4.4721$ |

### E5.2 Predictions for K = 1, 3, 5 (1 mark)

Neighbours ordered by distance:

| Rank | Borrower | Distance | Label |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 2.2361 | 0 |
| 2 | 5 | 3.1623 | 1 |
| 3 | 3 | 3.6056 | 1 |
| 4 | 1 | 4.0000 | 0 |
| 5 | 6 | 4.4721 | 1 |
| 6 | 4 | 5.0000 | 0 |

| $K$ | Neighbours | Votes | Prediction |
|:---:|:---|:---|:---:|
| 1 | 2 | one vote for 0 | **0** (repaid) |
| 3 | 2, 5, 3 | two for 1, one for 0 | **1** (default) |
| 5 | 2, 5, 3, 1, 6 | three for 1, two for 0 | **1** (default) |

### E5.3 Units and standardising (1 mark)

In metres, a distance gap of 1 km becomes 1000, so its squared difference ($10^6$) swamps income gaps of a few thousand rupees (squared differences around 4 to 16), and KNN would pick neighbours almost only by distance to the branch. Standardising each feature, $z = (x - \mu)/\sigma$, puts both on the same scale with mean 0 and standard deviation 1, so income and distance count fairly in the distance.

## Exercise 6: Naive Bayes

Ten past loans with yes/no features SG (savings-group member) and MP (missed an earlier payment). New applicant: SG $=$ No and MP $=$ Yes. No smoothing.

| Loan | SG | MP | Default |
|:---:|:---:|:---:|:---:|
| 1 | No | Yes | Yes |
| 2 | No | Yes | Yes |
| 3 | No | No | Yes |
| 4 | Yes | Yes | Yes |
| 5 | Yes | No | No |
| 6 | Yes | No | No |
| 7 | Yes | Yes | No |
| 8 | No | No | No |
| 9 | Yes | No | No |
| 10 | No | No | No |

### E6.1 Priors and class-conditional probabilities (1.5 marks)

**Priors.** Default $=$ Yes for loans 1 to 4 (4 loans) and Default $=$ No for loans 5 to 10 (6 loans):

$$
P(\text{Yes}) = \frac{4}{10} = 0.4000, \qquad P(\text{No}) = \frac{6}{10} = 0.6000
$$

**Class-conditional probabilities.**

| Probability | Value | Loans counted |
|:---|:---:|:---|
| $P(\text{SG} = \text{No} \mid \text{Yes})$ | $3/4 = 0.7500$ | 1, 2, 3 out of 1 to 4 |
| $P(\text{MP} = \text{Yes} \mid \text{Yes})$ | $3/4 = 0.7500$ | 1, 2, 4 out of 1 to 4 |
| $P(\text{SG} = \text{No} \mid \text{No})$ | $2/6 = 0.3333$ | 8, 10 out of 5 to 10 |
| $P(\text{MP} = \text{Yes} \mid \text{No})$ | $1/6 = 0.1667$ | 7 out of 5 to 10 |

### E6.2 Posterior and predicted class (1.5 marks)

Let $X = (\text{SG} = \text{No}, \text{MP} = \text{Yes})$. Unnormalised scores (prior $\times$ likelihoods):

$$
\begin{aligned}
P(\text{Yes})\,P(X \mid \text{Yes}) &= 0.4 \times \tfrac{3}{4} \times \tfrac{3}{4} = 0.4 \times 0.5625 = 0.2250 \\[4pt]
P(\text{No})\,P(X \mid \text{No}) &= 0.6 \times \tfrac{2}{6} \times \tfrac{1}{6} = 0.6 \times 0.0556 = 0.0333
\end{aligned}
$$

Normalising by their sum, $0.2250 + 0.0333 = 0.2583$:

$$
\begin{aligned}
P(\text{Yes} \mid X) &= \frac{0.2250}{0.2583} = \boxed{0.8710} \\[6pt]
P(\text{No} \mid X) &= \frac{0.0333}{0.2583} = 0.1290
\end{aligned}
$$

Since $0.8710 > 0.5$, the predicted class is **Default $=$ Yes**.

### E6.3 Independence assumption (1 mark)

**Assumption:** given the class, the features are independent:

$$
P(\text{SG}, \text{MP} \mid \text{Default}) = P(\text{SG} \mid \text{Default}) \times P(\text{MP} \mid \text{Default})
$$

**Why it may not hold:** savings-group membership and missed payments are likely to be linked. Group members get peer support, regular saving habits and reminders from the group, so they are less likely to miss a payment. Knowing SG then tells us something about MP even within the same class, which breaks the assumption.

---

# Notation

| Symbol | Meaning |
|:---|:---|
| $\mathbf{x} \in \mathbb{R}^d$ | feature vector of one observation |
| $\mathbf{w},\ b$ | weight vector and bias (intercept) of a linear model |
| $s = \mathbf{w}\cdot\mathbf{x} + b$ | linear score, called $z$ when read as log-odds |
| $\sigma(z) = 1/(1 + e^{-z})$ | logistic (sigmoid) function |
| $y,\ \hat y$ | true and predicted class label |
| $t$ | classification threshold on $\hat p$ |
| $\eta$ | perceptron learning rate |
| TP, FP, FN, TN | confusion-matrix counts (positive class = default) |
| $K$ | number of neighbours in KNN |
| $C$ | SVM regularisation parameter (penalty on margin violations) |

# References

- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An introduction to statistical learning: With applications in Python*. Springer. https://doi.org/10.1007/978-3-031-38747-0
- Newman, D. A. (2014). Missing data: Five practical guidelines. *Organizational Research Methods*, *17*(4), 372–411. https://doi.org/10.1177/1094428114548590
