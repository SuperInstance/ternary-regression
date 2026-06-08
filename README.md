# ternary-regression

Linear regression where every feature lives in {−1, 0, +1}.

## The Problem

You have a dataset where features are ternary — quantized neural network weights, balanced ternary encodings, hash codes — and you need to predict a continuous target. You *could* feed these into any regression library, but you'd be paying for generality you don't need: feature scaling is unnecessary, the design matrix is well-conditioned by construction, and the Gramian `XᵀX` has a structure you can reason about.

The real problem: general-purpose solvers don't exploit that structure. They preprocess, precondition, and regularize for the worst case. With ternary features, the worst case rarely arrives.

## The Insight

When features are {−1, 0, +1}, the design matrix `XᵀX` is tightly bounded. Each diagonal entry is at most N (if a feature is always ±1), and off-diagonal entries are correlation-weighted sums bounded by the same N. This means the condition number of `XᵀX` is predictable and usually moderate — OLS rarely needs regularization.

The gradient `∂L/∂wⱼ = 2(ŷ − y) · xⱼ/n` is also constrained: `xⱼ` is one of three values, so every gradient update is a discrete multiple of the residual. The optimizer can't oscillate in tiny increments — it moves in steps proportional to the error, scaled by {-1, 0, +1}.

This is why ternary regression works with a straightforward normal-equation solver and Gaussian elimination: the problem is already well-conditioned.

## How It Works

**Three solvers, one decision point:**

```
fit(x, y) → l1_penalty > 0?
                │
         yes ←──┴──→ no
          │           │
    fit_iterative   fit_normal
    (proximal       (XᵀX + λI)β = Xᵀy
     gradient        via Gaussian elimination
     + soft
     threshold)
```

### OLS and Ridge: The Normal Equation

1. Compute `XᵀX` — a d×d matrix where each entry is a sum of products of ternary values (integer arithmetic, no floating-point accumulation error in the products themselves).
2. Add `λI` to the diagonal (λ=0 for OLS, λ>0 for Ridge).
3. Compute `Xᵀy`.
4. Solve via Gaussian elimination with partial pivoting.
5. Derive the intercept separately: `b = ȳ − x̄·β`. This keeps `XᵀX` at d×d instead of (d+1)×(d+1).

### Lasso: Proximal Gradient Descent

Starts from the OLS/Ridge solution, then iterates:

```
βⱼ ← soft_threshold(βⱼ − lr · ∂L/∂βⱼ, lr · λ)
```

The `soft_threshold` operator is the proximal operator for the L1 norm: `sign(z) · max(|z| − λ, 0)`. It drives small coefficients to exactly zero, producing sparse solutions — a built-in feature selector that tells you *which* ternary features matter.

## Code Example

```rust
use ternary_regression::{ols_regression, ridge_regression, lasso_regression,
                         TernaryLinearRegression, RegressionConfig, analyze_residuals};

// Features: ternary patterns. Targets: continuous.
let x: Vec<Vec<i8>> = vec![
    vec![ 1,  1],
    vec![ 1, -1],
    vec![-1,  1],
    vec![-1, -1],
    vec![ 0,  0],
];
let y: Vec<f64> = vec![5.0, 1.0, -1.0, -5.0, 0.0]; // y ≈ 2·x₀ + 3·x₁

// OLS — exact solution, no hyperparameters
let ols = ols_regression(&x, &y);
println!("coefs: {:?}", ols.coefficients);  // [~2.0, ~3.0]
println!("R²:    {:.4}", ols.r_squared);     // ~1.0

// Ridge — shrinks coefficients toward zero
let ridge = ridge_regression(&x, &y, 10.0);
let ols_norm: f64 = ols.coefficients.iter().map(|c| c*c).sum::<f64>().sqrt();
let ridge_norm: f64 = ridge.coefficients.iter().map(|c| c*c).sum::<f64>().sqrt();
assert!(ridge_norm <= ols_norm); // always holds

// Lasso — drives irrelevant features to exactly zero
let x_sparse: Vec<Vec<i8>> = vec![
    vec![1, 1], vec![1, -1], vec![-1, 1], vec![-1, -1], vec![1, 0], vec![-1, 0],
];
let y_sparse: Vec<f64> = x_sparse.iter().map(|xi| 3.0 * xi[0] as f64).collect();
let lasso = lasso_regression(&x_sparse, &y_sparse, 0.5);
// lasso.coefficients[1] ≈ 0.0 (feature 1 is irrelevant)

// Predict on new data
let preds = TernaryLinearRegression::predict(&ols, &vec![vec![1, 0], vec![-1, 0]]);

// Residual analysis
let analysis = analyze_residuals(&ols.residuals);
println!("residual std: {:.4}", analysis.std_dev);
println!("MAE:          {:.4}", analysis.mae);
```

## Module Map

```
ternary_regression
├── TernaryLinearRegression
│   ├── new()                         — default config (OLS)
│   ├── with_config(config)           — custom config
│   ├── fit(x, y) → RegressionResult  — dispatches to normal or iterative
│   └── predict(result, x) → Vec<f64>
├── RegressionConfig
│   ├── l2_penalty: f64               — Ridge strength (0 = OLS)
│   ├── l1_penalty: f64               — Lasso strength (0 = no L1)
│   ├── learning_rate: f64            — step size for Lasso
│   ├── max_iter: usize               — cap for iterative solver
│   └── tol: f64                      — convergence threshold
├── RegressionResult
│   ├── coefficients: Vec<f64>
│   ├── intercept: f64
│   ├── r_squared: f64
│   ├── residuals: Vec<f64>
│   └── iterations: usize             — 0 for direct solve
├── Convenience functions
│   ├── ols_regression(x, y)
│   ├── ridge_regression(x, y, alpha)
│   └── lasso_regression(x, y, alpha)
├── Metrics
│   ├── mse(y_true, y_pred) → f64
│   ├── mae(y_true, y_pred) → f64
│   └── analyze_residuals(residuals) → ResidualAnalysis
└── Internal
    ├── solve_linear_system(A, b)     — Gaussian elimination + partial pivoting
    ├── compute_r_squared(y, ŷ)       — 1 − SS_res/SS_tot
    └── soft_threshold(z, λ)          — sign(z)·max(|z|−λ, 0)
```

## Design Decisions

**Intercept computed separately, not absorbed into the normal equation.** Adding a constant feature column would make `XᵀX` a (d+1)×(d+1) system and worsen conditioning. Computing `b = ȳ − x̄·β` after solving keeps the system smaller and better-conditioned.

**Gaussian elimination, not LU/Cholesky.** For the typical dimensionality of ternary feature spaces (d < 100), the O(d³) cost is negligible. Gaussian elimination with partial pivoting is simple, correct, and avoids the positive-definiteness requirement of Cholesky.

**Lasso uses proximal gradient, not coordinate descent.** Coordinate descent cycles through features one at a time. Proximal gradient updates all features simultaneously with soft-thresholding. The trade-off: proximal gradient needs a learning rate, but it parallelizes trivially if you later want to SIMD the gradient computation.

**i8 features, f64 arithmetic.** The input features are `i8` (compact, three-valued), but all matrix arithmetic is `f64`. The ternary structure helps the *conditioning* of the problem, but the solution itself is continuous — the coefficients are real numbers.

## Status

| Aspect | State |
|--------|-------|
| OLS | Stable, tested |
| Ridge (L2) | Stable, tested |
| Lasso (L1) | Works, fixed iteration count (no early stopping) |
| Elastic Net (L1+L2) | Not supported |
| Weighted least squares | Not supported |
| Cross-validation | Not built-in |
| SIMD / parallel | Not yet |
| MSRV | Edition 2024 |

**Known limitations:** The Lasso solver runs for a fixed `max_iter` iterations without checking the convergence tolerance. For high penalty values, the learning rate may need manual tuning. No elastic net (combined L1+L2) — you get one or the other.

## Related Crates

- **[ternary-logistic](https://github.com/SuperInstance/ternary-logistic)** — Same feature space, categorical targets
- **[ternary-em](https://github.com/SuperInstance/ternary-em)** — Discover subpopulations before regression
- **[ternary-fence](https://github.com/SuperInstance/ternary-fence)** — Distributed regression across workers

## License

MIT
