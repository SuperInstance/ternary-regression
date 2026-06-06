# ternary-regression

Linear regression where every feature is {−1, 0, +1}.

Standard linear regression works fine with ternary features — the math doesn't care about the input range. But when you *know* your features are ternary, interesting things happen: the design matrix has bounded condition number, regularization is less critical, and the normal equation solver doesn't need preconditioning. This crate exploits those properties to give you clean, interpretable regression results with a minimal API.

It provides three solvers (OLS, Ridge, Lasso), full residual analysis, R² computation, and convenience functions that get you from data to results in one call.

## Why This Exists

When you regress continuous targets onto ternary features, you're essentially asking: "Given a pattern of activations and inhibitions, what's the expected output?" This comes up constantly in:

- **Quantized neural network analysis**: predict layer output statistics from ternary weight patterns
- **Ternary feature importance**: which {−1, 0, +1} features drive the target?
- **Calibration**: map ternary classifier scores to calibrated probabilities

The key insight: with ternary features, `XᵀX` has a predictable structure. Each diagonal entry counts feature variance (bounded by 1), and off-diagonal entries are correlation-weighted. This means OLS rarely needs regularization — the features are already well-conditioned by construction.

## Quick Start

```rust
use ternary_regression::{ols_regression, TernaryLinearRegression};

// Ternary features, continuous targets
let x = vec![
    vec![ 1,  1],   // both features positive
    vec![ 1, -1],   // mixed
    vec![-1,  1],   // mixed
    vec![-1, -1],   // both negative
    vec![ 0,  0],   // neutral
];
let y = vec![5.0, 1.0, -1.0, -5.0, 0.0]; // y = 2*x[0] + 3*x[1]

let result = ols_regression(&x, &y);

println!("Coefficients: {:?}", result.coefficients);  // [~2.0, ~3.0]
println!("Intercept:    {:.4}", result.intercept);     // ~0.0
println!("R²:           {:.4}", result.r_squared);     // ~1.0

// Predict on new data
let predictions = TernaryLinearRegression::predict(&result, &vec![
    vec![1, 0], vec![-1, 0],
]);
```

## The Three Solvers

### OLS — When You Don't Need Regularization

Solves `(XᵀX)β = Xᵀy` via Gaussian elimination with partial pivoting. Exact solution, no hyperparameters.

```rust
use ternary_regression::ols_regression;

let result = ols_regression(&x, &y);
```

Use this when you have more samples than features and no multicollinearity (common with ternary features).

### Ridge (L2) — Shrink Coefficients

Adds `λI` to `XᵀX` to handle multicollinearity and prevent overfitting:

```rust
use ternary_regression::ridge_regression;

let result = ridge_regression(&x, &y, 1.0); // alpha = 1.0
// Coefficients are smaller in magnitude than OLS
```

Use this when features are correlated or you have few samples.

### Lasso (L1) — Feature Selection

Uses proximal gradient descent with soft-thresholding. Produces sparse coefficients — exactly zero for irrelevant features:

```rust
use ternary_regression::lasso_regression;

let result = lasso_regression(&x, &y, 0.5);
// Some coefficients will be exactly 0.0
```

Use this when you want to know *which* ternary features matter.

## Architecture

```
┌────────────────────────────────────────────────────┐
│         TernaryLinearRegression                    │
│                                                    │
│  fit(x, y) ──→ l1 > 0? ──yes──→ fit_iterative()  │
│                    │               (proximal grad)  │
│                    no                               │
│                    │                               │
│                    └──→ fit_normal()               │
│                         (XᵀX + λI)β = Xᵀy        │
│                                                    │
│  predict(result, x_new) → ŷ                       │
├────────────────────────────────────────────────────┤
│  Utility Functions                                 │
│  mse(y_true, y_pred)                               │
│  mae(y_true, y_pred)                               │
│  analyze_residuals(residuals) → ResidualAnalysis   │
├────────────────────────────────────────────────────┤
│  Convenience Functions                             │
│  ols_regression(x, y)                              │
│  ridge_regression(x, y, alpha)                     │
│  lasso_regression(x, y, alpha)                     │
└────────────────────────────────────────────────────┘
```

### The Normal Equation Solver

For OLS and Ridge, the solve path is:

1. Compute `XᵀX` (d × d matrix) — for N samples with d ternary features
2. Add `λI` to diagonal (zero for OLS)
3. Compute `Xᵀy` (d × 1 vector)
4. Solve via Gaussian elimination with partial pivoting
5. Derive intercept: `b = ȳ − x̄·β`

The intercept isn't part of the normal equation — it's computed separately as the mean adjustment. This means `XᵀX` is always d × d, not (d+1) × (d+1), which matters for numerical stability.

### The Lasso Solver

Proximal gradient descent with soft-thresholding:

```
βⱼ ← sign(βⱼ − lr·∂L/∂βⱼ) · max(|βⱼ − lr·∂L/∂βⱼ| − lr·λ, 0)
```

The soft-thresholding operator is the proximal operator for the L1 norm. It drives small coefficients to exactly zero, producing sparse solutions.

## API Reference

### `RegressionResult`

| Field | Description |
|-------|-------------|
| `coefficients` | Fitted feature weights (Vec<f64>) |
| `intercept` | Bias term |
| `r_squared` | R² on training data |
| `residuals` | yᵢ − ŷᵢ for each training point |
| `iterations` | Iterations used (0 for direct solve) |

### `RegressionConfig`

| Field | Default | Purpose |
|-------|---------|---------|
| `l2_penalty` | 0.0 | Ridge strength |
| `l1_penalty` | 0.0 | Lasso strength |
| `learning_rate` | 0.01 | Step size (lasso only) |
| `max_iter` | 10000 | Max iterations (lasso only) |
| `tol` | 1e-10 | Convergence tolerance |

### `ResidualAnalysis`

Produced by `analyze_residuals(&result.residuals)`:

| Field | Description |
|-------|-------------|
| `mean` | Should be near 0 for a good fit |
| `std_dev` | Spread of errors |
| `min` / `max` | Error range |
| `mse` | Mean squared error |
| `mae` | Mean absolute error |

## Real-World Example: Predicting Quantized Layer Performance

```rust
use ternary_regression::{ols_regression, analyze_residuals, TernaryLinearRegression};

// Features: ternary weight statistics for each layer in a quantized network
// x[i] = [ratio_neg, ratio_zero, ratio_pos, sparsity_pattern]
let layer_features: Vec<Vec<i8>> = vec![
    vec![-1,  0,  1,  1],   // layer 0
    vec![ 0,  0,  0,  1],   // layer 1 (sparse)
    vec![-1, -1,  1, -1],   // layer 2
    vec![ 1,  0,  1,  0],   // layer 3
    vec![ 0,  1,  0, -1],   // layer 4
];

// Target: measured inference accuracy drop (percentage points)
let accuracy_drop = vec![0.5, 0.2, 1.8, 0.3, 0.1];

let result = ols_regression(&layer_features, &accuracy_drop);

// Which features predict accuracy drop?
for (i, coef) in result.coefficients.iter().enumerate() {
    println!("Feature {}: {:.3} ({})", 
        i, coef,
        if coef.abs() > 0.5 { "important" } else { "noise" });
}

// Residual analysis
let analysis = analyze_residuals(&result.residuals);
println!("Mean residual: {:.4} (should be ~0)", analysis.mean);
println!("R²: {:.4}", result.r_squared);

// Predict for a new layer
let new_layer = vec![1i8, -1, 1, 1];
let prediction = TernaryLinearRegression::predict(&result, &vec![new_layer]);
println!("Predicted accuracy drop: {:.2}%", prediction[0]);
```

## Ecosystem Connections

- **`ternary-logistic`** — Same features, categorical targets. Use regression for continuous targets, logistic for classification.
- **`ternary-em`** — EM can discover subpopulations in your data before regression. Run EM to split, then regress within each cluster.
- **`ternary-fence`** — Coordinate distributed regression training across workers.

## Performance Notes

- **OLS solve**: O(d³) for the Gaussian elimination, where d is the number of features. With ternary features, d is typically small (<100), so this is fast.
- **Lasso**: O(N × d × max_iter). Convergence depends on the learning rate and penalty strength. Start with defaults and tune.
- **Memory**: O(d²) for XᵀX. For d < 1000, fits in L1/L2 cache.
- **Ternary advantage**: The design matrix entries are −1, 0, +1, so XᵀX computation uses integer multiply-accumulate. Potential for SIMD optimization.

## Open Questions

- **Weighted least squares**: No support for sample weights. Would be useful for importance-weighted regression on imbalanced ternary datasets.
- **Robust regression**: OLS is sensitive to outliers. An M-estimator or RANSAC variant would be more robust.
- **Cross-validation**: No built-in k-fold CV. You'd need to implement it manually to compare OLS vs Ridge vs Lasso.
- **Elastic net**: Combines L1 and L2. Currently you get one or the other, not both.

## License

MIT
