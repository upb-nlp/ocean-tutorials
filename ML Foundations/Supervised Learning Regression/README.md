# Supervised Learning: Regression

**Regression** is the supervised-learning task of mapping an input vector $x \in \mathbb{R}^d$ to a *continuous* target $y \in \mathbb{R}$. Given a labeled training set $D = \{(x_i, y_i)\}_{i=1}^n$, the goal is to learn a function $h: \mathbb{R}^d \rightarrow \mathbb{R}$ - the **hypothesis** - that generalizes to *unseen* inputs.

```
Input Features (x) → [ Regressor h ] → Predicted Value ŷ
   living area,            Linear / Ridge / Lasso         house price
   #rooms, distance…       Polynomial / Tree / RF / GB    fuel efficiency
```

Regression underlies house-price estimation, demand forecasting, dose–response modelling, sensor calibration, click-through-rate prediction and most "how-much / how-many" decisions taken by ML systems.

## 1. The Regression Setting

### 1.1 Definition

> Regression predicts a value of a given **continuous** variable based on the values of other variables, assuming a *linear* or *nonlinear* model of dependency.

It is used for **prediction** and **forecasting** - its application overlaps machine learning - and also to understand the *relationship* between variables.

Given a training set $X$ (e.g. living area) and corresponding targets $Y$ (e.g. house price), the goal is to learn a function $h: X \rightarrow Y$ such that $h(x)$ is a good predictor for the corresponding $y$. The function $h$ is also called the **hypothesis**.

### 1.2 A worked example - Portland house prices

Living areas (ft²) and selling prices for houses in Portland, Oregon can be summarised by a straight line:

$$
\hat{y} = 71.27 + 0.1345 \cdot x
$$

- The **bias** $b = 71.27$ is where the line intersects the y-axis.
- The **weight** $w = 0.1345$ is the slope of the line.

Using this model, a $2{,}000\ \text{ft}^2$ house has a predicted price of about **$310{,}000**.

### 1.3 Train / Validation / Test

A model that simply memorises the training set has zero training error but learns nothing useful. To estimate **generalisation**, the dataset is partitioned into three disjoint subsets:

```
┌───────────────────────────────────────────────────────────────┐
│           Full Labeled Dataset D                              │
├───────────────────┬────────────────┬──────────────────────────┤
│  Training (60%)   │ Validation 20% │       Test (20%)         │
│  fit parameters   │ tune hparams   │   final, untouched eval  │
└───────────────────┴────────────────┴──────────────────────────┘
```

- **Training set** - the model fits its parameters (weights $w$, tree splits, leaf values).
- **Validation set** - used to choose hyperparameters (polynomial degree, ridge $\alpha$, tree depth, $\eta$, $T$).
- **Test set** - touched **once**, at the very end, to report final, unbiased performance.

### 1.4 Regression Metrics

Let $y_i$ be the true value and $\hat{y}_i$ the prediction.

| Metric | Formula | Notes |
|---|---|---|
| **MAE** | $\tfrac{1}{n}\sum_i \lvert y_i - \hat{y}_i \rvert$ | Mean absolute error - robust to outliers, in target units |
| **MSE** | $\tfrac{1}{n}\sum_i (y_i - \hat{y}_i)^2$ | Mean squared error - penalises large errors more strongly |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as $y$; widely reported |
| **$R^2$** | $1 - \dfrac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$ | Fraction of variance explained; 1 = perfect, 0 = mean baseline |
| **MAPE** | $\tfrac{1}{n}\sum_i \left\lvert \tfrac{y_i - \hat{y}_i}{y_i} \right\rvert$ | Scale-free; breaks when $y_i \approx 0$ |

## 2. Linear Regression

### 2.1 The Model

With a single feature, the hypothesis is a straight line:

$$
\hat{y} = w x + b
$$

With multiple features, every feature gets its own weight:

$$
\hat{y} = b + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = b + w^\top x
$$

Example: a house-price model could use

- Number of rooms ($x_1$)
- Courtyard area size ($x_2$)
- Distance from the city centre ($x_3$)

each with a separate weight $w_j$.

### 2.2 Losses - L1 vs L2

Before learning we need a **loss**: a function that measures how far the model's predictions are from the actual values. Visually, it is the **vertical distance** from each data point to the regression line.

| Loss | Per-example | Aggregate |
|---|---|---|
| **L1 loss** | $\lvert y_i - \hat{y}_i \rvert$ | **MAE** - average L1 across the set |
| **L2 loss** | $(y_i - \hat{y}_i)^2$ | **MSE** - average L2 across the set |

The functional difference is **squaring**:

- When the error is large, squaring makes it *even larger* → MSE is sensitive to outliers.
- When the error is small (< 1), squaring makes it *even smaller* → MSE tolerates tiny errors.

| Residual $\lvert y - \hat{y} \rvert$ | L1 loss | L2 loss |
|---|---|---|
| 0.5 | 0.5 | 0.25 |
| 1 | 1 | 1 |
| 2 | 2 | **4** |
| 5 | 5 | **25** |
| 10 | 10 | **100** |

The L2 column explodes for large residuals -> one outlier dominates the optimisation. L1 grows linearly, so an outlier worth 10x the typical error contributes only 10x the typical loss instead of 100x.

### 2.3 Optimisation

Training is the **optimisation problem**: choose $w^*$ so that $E(w^*) = \min_w E(w)$.

For L2 loss with a linear model, the problem has a **closed-form** solution (the *normal equation*):

$$
w^* = (X^\top X)^{-1} X^\top y
$$

For large $d$ or non-convex losses, gradient descent is used instead.

### 2.4 Maximum Likelihood Justification

Assume the targets are generated by a deterministic linear function plus **Gaussian noise**:

$$
t = w^\top \phi(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \beta^{-1})
$$

The likelihood of the observed targets factorises across examples; taking the logarithm gives

$$
\log p(\mathbf{t} \mid X, w, \beta) = -\frac{\beta}{2} \sum_{i=1}^n (t_i - w^\top \phi(x_i))^2 + \text{const}
$$

Maximising the likelihood w.r.t. $w$ is therefore *equivalent* to **minimising the sum-of-squared errors**. Setting the gradient to zero recovers the closed-form $w^* = (\Phi^\top \Phi)^{-1} \Phi^\top \mathbf{t}$.

## 3. Polynomial Regression and Basis Functions

### 3.1 From Lines to Curves

A linear model in $w$ can still be highly **non-linear in $x$** if we first transform $x$ through fixed **basis functions** $\phi_j(x)$:

$$
\hat{y} = w_0 + \sum_{j=1}^{M-1} w_j \, \phi_j(x) = w^\top \phi(x)
$$

If $\phi_0(x) = 1$, then $w_0$ acts as a **bias**. In the simplest case the basis is the identity (plain linear regression). Three popular alternatives:

| Basis | $\phi_j(x)$ | Behaviour |
|---|---|---|
| **Polynomial** | $x^j$ | **Global** - a small change in $x$ affects all basis functions |
| **Gaussian** | $\exp\!\left(-\tfrac{(x - \mu_j)^2}{2 s^2}\right)$ | **Local** - only nearby basis functions react; $\mu_j, s$ control location and scale |
| **Sigmoidal** | $\sigma\!\left(\tfrac{x - \mu_j}{s}\right)$ | **Local** - slope-like activation; $\mu_j, s$ control location and slope |

### 3.2 Polynomial Curve Fitting

For a single feature, the polynomial model of degree $M$ is

$$
\hat{y}(x, w) = w_0 + w_1 x + w_2 x^2 + \dots + w_M x^M
$$

Increasing $M$ lets the curve bend more. The degrees commonly studied:

```
M = 0  ──→  horizontal line                  (severe underfit)
M = 1  ──→  best straight line               (still underfit on curvy data)
M = 3  ──→  smooth cubic                     (often a good compromise)
M = 9  ──→  passes through every point       (over-fit - wild oscillations)
```

### 3.3 Over-fitting and the RMS Curve

The Root-Mean-Square error

$$
E_{\text{RMS}} = \sqrt{\tfrac{1}{n}\sum_i (y_i - \hat{y}_i)^2}
$$

drops to zero on the **training set** as $M$ grows - the model memorises the points. On the **test set** it grows again past some sweet spot:

| Polynomial degree $M$ | Train RMSE | Test RMSE |
|---|---|---|
| Very low (under-fit) | high | high |
| Sweet spot | low | **lowest** |
| Very high (over-fit) | -> 0 | high |

The "U" shape of the test curve - low for moderate $M$, large at both ends - is the canonical **bias-variance trade-off** picture. See `polynomial_fits.png` in the tutorial outputs: degree 1 under-fits, degree 12 fits well, degree 16 oscillates wildly.

### 3.4 Remedies for Over-fitting

| Strategy | Idea |
|---|---|
| **More data** | A 9th-order polynomial fitted on 15 points oscillates wildly; fitted on 100 points it traces the true curve cleanly. |
| **Lower model complexity** | Cap $M$, prune trees, shrink network width. |
| **Regularisation** | Keep $M$ large but **penalise** large coefficients (see §4). |
| **Cross-validation** | Choose $M$, $\alpha$, depth etc. on a held-out set. |

The example from the companion script `supervised_regression.py` fits degrees $\{1, 4, 8, 12, 16\}$ to $\sqrt{x}\sin(x)$ with Gaussian noise. Low degrees underfit; very high degrees oscillate near the edges (Runge's phenomenon).

## 4. Regularisation

### 4.1 Why Penalise Coefficients?

A 9th-order polynomial that interpolates a noisy training set ends up with **enormous coefficients of alternating sign**. Regularisation augments the loss with a term that grows with the size of $w$:

$$
\mathcal{L}_{\text{reg}}(w) = \underbrace{\tfrac{1}{2}\sum_i (y_i - w^\top \phi(x_i))^2}_{\text{data fit}} + \underbrace{\alpha \cdot R(w)}_{\text{penalty}}
$$

The hyperparameter $\alpha \geq 0$ controls the trade-off: $\alpha = 0$ recovers ordinary least squares, $\alpha \to \infty$ forces all weights to zero.

### 4.2 Ridge Regression (L2 penalty)

$$
R(w) = \|w\|_2^2 = \sum_{j} w_j^2
$$

Ridge has a closed-form solution that is **always invertible** even when $\Phi^\top \Phi$ is singular:

$$
w_{\text{ridge}}^* = (\Phi^\top \Phi + \alpha I)^{-1} \Phi^\top \mathbf{t}
$$

Ridge **shrinks** coefficients smoothly toward zero but rarely sets any *exactly* to zero. Use it when many features each contribute a little.

**MAP interpretation.** Placing a zero-mean Gaussian prior $p(w) = \mathcal{N}(0, \alpha^{-1} I)$ on the weights and taking the maximum *a posteriori* (MAP) estimate gives exactly the ridge objective - the regularisation term is the negative log-prior. Hence "MAP = MLE + L2 regulariser".

### 4.3 Lasso Regression (L1 penalty)

$$
R(w) = \|w\|_1 = \sum_{j} |w_j|
$$

Lasso has no closed-form solution but is a convex problem solved by coordinate descent. Crucially, the L1 ball has **corners** on the axes, so the optimum often lands on a corner - pushing some weights to **exactly zero**. Lasso therefore performs **automatic feature selection** alongside fitting.

Geometrically: Ridge constrains $w$ to a **circle** ($w_1^2 + w_2^2 \le t$), Lasso constrains it to a **diamond** ($|w_1| + |w_2| \le t$). The diamond has corners exactly on the axes, so the unconstrained optimum often gets pulled to a corner where one coordinate is zero. The circle has no corners, so Ridge solutions almost never sit exactly on an axis.

| Constraint region | Shape | Optimum tends to land… |
|---|---|---|
| **Ridge** ($\lVert w \rVert_2 \le t$) | circle / sphere | anywhere on the boundary |
| **Lasso** ($\lVert w \rVert_1 \le t$) | diamond / cross-polytope | on a **corner** (axis) → sparse $w$ |

### 4.4 Elastic Net (L1 + L2)

When features are highly **correlated**, lasso picks an arbitrary one and drops the others; ridge keeps them all. Elastic Net blends both penalties:

$$
R(w) = \rho \|w\|_1 + (1 - \rho) \|w\|_2^2
$$

with $\rho \in [0, 1]$ controlling the mix. It retains lasso's sparsity while sharing weight across correlated features - a good default when $d > n$ or features form groups.

### 4.5 Summary

| Method | Penalty | Closed-form? | Sparse $w$? | Use when… |
|---|---|---|---|---|
| **OLS** | none | ✅ | ❌ | $n \gg d$, no correlated features |
| **Ridge** | $\lVert w \rVert_2^2$ | ✅ | ❌ | many small contributors; multicollinearity |
| **Lasso** | $\lVert w \rVert_1$ | ❌ (CD) | ✅ | many features, most irrelevant |
| **Elastic Net** | $\rho \lVert w \rVert_1 + (1-\rho)\lVert w \rVert_2^2$ | ❌ (CD) | ✅ (less aggressive) | correlated features; $d > n$ |

**Always standardise features** before regularised regression - the penalty would otherwise punish features with large numeric scale disproportionately.

## 5. Bayesian Linear Regression (a short detour)

Instead of choosing $w$ as fixed parameters, Bayesian regression computes a **probability distribution** over all possible values of $w$.

$$
\underbrace{p(w \mid \mathbf{t})}_{\text{posterior}} \propto \underbrace{p(\mathbf{t} \mid w)}_{\text{likelihood}} \cdot \underbrace{p(w)}_{\text{prior}}
$$

Predictions for a new point integrate over the posterior:

$$
p(t^* \mid x^*, \mathbf{t}) = \int p(t^* \mid x^*, w) \, p(w \mid \mathbf{t}) \, dw
$$

Three benefits:

- Escapes the over-fitting problem of plain ML.
- Yields a **predictive variance** that shrinks near observed points and grows in unexplored regions - exactly what you want for "how confident are we?".
- Determines model complexity **automatically** from the training data alone.

| Estimator | Optimises | Adds |
|---|---|---|
| **MLE** | $p(\mathbf{t} \mid X, w)$ | nothing → minimises MSE |
| **MAP** | $p(w \mid \mathbf{t})$ | Gaussian prior → minimises MSE + $\alpha \lVert w \rVert^2$ |
| **Bayes** | full posterior $p(w \mid \mathbf{t})$ | predictive distribution with calibrated uncertainty |

## 6. Tree-Based Regression

### 6.1 Regression Trees (CART)

A regression tree partitions feature space into axis-aligned rectangles. Each **leaf** stores the **mean** of the training targets that fall into it; prediction looks up the leaf for the query point.

```
                  living_area > 1500 ?
                /                       \
             No                          Yes
             /                            \
        rooms ≤ 2 ?                 city_centre_dist < 5 ?
        /     \                       /              \
     ŷ=145   ŷ=210                  ŷ=420           ŷ=320
```

**Split criterion.** A candidate split divides a node $S$ into $S_L$ and $S_R$. The reduction in **sum of squared errors** is

$$
\text{Gain} = \sum_{i \in S} (y_i - \bar{y}_S)^2 - \left[ \sum_{i \in S_L} (y_i - \bar{y}_{S_L})^2 + \sum_{i \in S_R} (y_i - \bar{y}_{S_R})^2 \right]
$$

The pair (feature, threshold) with the largest gain wins. (Some libraries use MAE instead of MSE; it is slower but more robust to outliers.)

### 6.2 Hyperparameters that Matter

| Hyperparameter | Effect |
|---|---|
| `max_depth` | Hard cap on tree depth. Shallow → underfit; deep → memorises. |
| `min_samples_leaf` | Refuses splits that leave fewer than this many points per leaf. Smooths predictions. |
| `min_samples_split` | Minimum size of a node before it is considered for splitting. |
| `max_features` | Subset of features considered at each split. Lower → more variance, more randomness. |
| `ccp_alpha` | Cost-complexity post-pruning strength. |

Trees produce **piecewise-constant** predictions - they cannot extrapolate beyond the training range and produce visible "stairs" on smooth functions.

### 6.3 Random Forest Regressor

A **random forest** averages many decorrelated trees:

1. **Bootstrap sample** the training set (sample $n$ examples with replacement).
2. At every split consider only $m$ randomly chosen features ($m \approx d/3$ is typical for regression).
3. Grow each tree to maximum depth.
4. **Average** the predictions of all $T$ trees.

Averaging reduces **variance** without inflating bias - the staircase smooths out without the individual trees overfitting any harder than they already do.

### 6.4 Gradient Boosting Regressor

Gradient boosting builds trees **sequentially**, each fitting the **residuals** of the current ensemble:

```
F₀(x) = ȳ                       ← start with the mean
For t = 1, …, T:
    rᵢ ← yᵢ − F_{t-1}(xᵢ)       ← residual (the negative gradient of squared loss)
    fit shallow tree h_t to (xᵢ, rᵢ)
    F_t(x) ← F_{t-1}(x) + η · h_t(x)
```

The **learning rate** $\eta$ (shrinkage) trades off the contribution of each tree against the total number of trees needed. Typical settings: $\eta \in [0.01, 0.1]$, $T \in [100, 1000]$, depth $3$–$8$.

**XGBoost** (and LightGBM, CatBoost) add a second-order Taylor expansion, leaf-weight regularisation, sparsity-aware splits and histogram bucketing - they dominate tabular regression leaderboards.

### 6.5 Strengths and Weaknesses

| ✅ Pros | ❌ Cons |
|---|---|
| Capture non-linear interactions out of the box | Piecewise-constant - cannot extrapolate |
| Handle mixed feature types, no need to scale | Many hyperparameters in boosted variants |
| Robust to monotone transforms of features | Single trees are unstable |
| Built-in feature importance | Less interpretable than a linear model |
| GBT often top-of-leaderboard on tabular data | Slow to train on very large datasets without GPU |

## 7. Choosing a Regressor

```
Tabular, moderate size, want a great default? → Gradient Boosting / XGBoost
Need interpretability above all?              → Linear / Ridge regression
High dimensional, many irrelevant features?   → Lasso or Elastic Net
Small dataset, simple curve?                  → Polynomial regression with CV
Smooth function, want extrapolation?          → Linear or basis-function model
Need calibrated uncertainty?                  → Bayesian regression / GP
Heavy outliers in the target?                 → MAE-based (Huber, MAE tree)
```

Like with classification, the *no free lunch* theorem applies: benchmark a handful of models on a held-out set, then pick.

## Tutorial

The companion script `supervised_regression.py` trains and benchmarks the regressors covered above on a shared dataset (California Housing from `sklearn.datasets`, 20 640 samples × 8 features - predict the median house value of a district). It also reproduces the polynomial-curve-fitting demo from the lecture's Colab notebook on the synthetic $\sqrt{x}\sin(x)$ signal.

### Pipeline

1. **Load and preprocess** - split into train / validation / test; standardise features.
2. **Lab 1 - Data, metrics, baselines** - sizes, mean-baseline RMSE/MAE/$R^2$.
3. **Lab 2 - Linear & Polynomial regression** - reproduces the Colab `np.vander` example (degrees 1, 4, 8, 12, 16) on the synthetic signal *and* fits linear regression on California Housing.
4. **Lab 3 - Regularisation** - Ridge, Lasso, Elastic Net; coefficient paths; $\alpha$ sweeps.
5. **Lab 4 - Tree-based regression** - single tree, Random Forest, Gradient Boosting (and XGBoost if installed).
6. **Lab 5 - Final benchmark + visualisations** - every regressor on the test set, 5-fold cross-validation, prediction-vs-actual scatter plots, residual plots, feature importance, learning curve, coefficient paths.

All plots are saved to `./outputs/regression/`.

### Installation

```bash
# Create a virtual environment
python -m venv regression-course
source regression-course/bin/activate  # Windows: regression-course\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run all labs
python supervised_regression.py

# Or run a specific lab
python supervised_regression.py --lab 1   # Data, splits, evaluation metrics
python supervised_regression.py --lab 2   # Linear & Polynomial regression
python supervised_regression.py --lab 3   # Ridge, Lasso, Elastic Net
python supervised_regression.py --lab 4   # Decision Tree, Random Forest, Gradient Boosting
python supervised_regression.py --lab 5   # Final benchmark + visualisations
```

### Outputs

```
outputs/regression/
├── polynomial_fits.png         ← Colab demo: degrees 1, 4, 8, 12, 16 on √x·sin(x)
├── regularization_paths.png    ← Ridge / Lasso coefficient paths vs α
├── alpha_sweep.png             ← validation RMSE vs α for Ridge / Lasso / ElasticNet
├── metrics_comparison.png      ← bar chart of RMSE / MAE / R² across regressors
├── predictions_vs_actual.png   ← scatter plots for every regressor
├── residuals.png               ← residual plots for every regressor
├── feature_importance.png      ← RF / GB / XGBoost feature importances
├── cv_results.png              ← box-plots of 5-fold CV R²
└── learning_curve.png          ← train/test RMSE vs training-set size
```
