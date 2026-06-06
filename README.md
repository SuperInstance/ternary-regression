# ternary-regression

**Ternary Linear Regression**

A Rust library for linear regression with ternary `{-1, 0, +1}` features. Supports ordinary least squares (OLS), ridge regression (L2 penalty), lasso approximation (L1 via proximal gradient), residual analysis, and R² computation.

## Features

- **OLS Regression**: Normal equation solver with Gaussian elimination and partial pivoting
- **Ridge Regression**: L2-regularized least squares for multicollinearity and overfitting
- **Lasso Approximation**: L1-regularized regression via coordinate descent with soft-thresholding
- **Residual Analysis**: Mean, std dev, min/max, MSE, MAE of residuals
- **R² Computation**: Coefficient of determination for model quality
- **Prediction**: Out-of-sample prediction on new ternary features
- **Convenience Functions**: `ols_regression()`, `ridge_regression()`, `lasso_regression()`

## Quick Start

### Ordinary Least Squares

```rust
use ternary_regression::{ols_regression, TernaryLinearRegression};

let x = vec![
    vec![ 1,  1],
    vec![ 1, -1],
    vec![-1,  1],
    vec![-1, -1],
    vec![ 0,  0],
];
let y = vec![5.0, 1.0, -1.0, -5.0, 0.0]; // y = 2*x[0] + 3*x[1]

let result = ols_regression(&x, &y);
println!("Coefficients: {:?}", result.coefficients);
println!("Intercept: {:.4}", result.intercept);
println!("R²: {:.4}", result.r_squared);
```

### Ridge Regression

```rust
use ternary_regression::ridge_regression;

let result = ridge_regression(&x, &y, 1.0); // alpha = 1.0
```

### Lasso Regression

```rust
use ternary_regression::lasso_regression;

let result = lasso_regression(&x, &y, 0.5); // L1 penalty = 0.5
```

### Prediction on New Data

```rust
use ternary_regression::TernaryLinearRegression;

let new_data = vec![vec![1, 0], vec![-1, 1]];
let predictions = TernaryLinearRegression::predict(&result, &new_data);
```

### Residual Analysis

```rust
use ternary_regression::{ols_regression, analyze_residuals};

let result = ols_regression(&x, &y);
let analysis = analyze_residuals(&result.residuals);
println!("Mean residual: {:.6}", analysis.mean);
println!("Std dev: {:.6}", analysis.std_dev);
println!("MSE: {:.6}", analysis.mse);
println!("MAE: {:.6}", analysis.mae);
```

## API Overview

### `RegressionResult`

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `Vec<f64>` | Fitted feature weights |
| `intercept` | `f64` | Bias term |
| `r_squared` | `f64` | R² on training data |
| `residuals` | `Vec<f64>` | yᵢ - ŷᵢ for each training point |
| `iterations` | `usize` | Iterations used (0 for direct solve) |

### `RegressionConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `l2_penalty` | 0.0 | Ridge regularization strength |
| `l1_penalty` | 0.0 | Lasso regularization strength |
| `learning_rate` | 0.01 | Step size for iterative methods |
| `max_iter` | 10000 | Maximum iterations |
| `tol` | 1e-10 | Convergence tolerance |

### `TernaryLinearRegression`

| Method | Description |
|--------|-------------|
| `new()` | OLS config |
| `with_config(config)` | Custom config |
| `fit(x, y)` | Fit model, returns `RegressionResult` |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `ols_regression(x, y)` | Fit OLS |
| `ridge_regression(x, y, alpha)` | Fit ridge (L2) |
| `lasso_regression(x, y, alpha)` | Fit lasso (L1) |

### Error Metrics

| Function | Description |
|----------|-------------|
| `mse(y_true, y_pred)` | Mean squared error |
| `mae(y_true, y_pred)` | Mean absolute error |
| `analyze_residuals(residuals)` | Full residual analysis |

### `ResidualAnalysis`

| Field | Description |
|-------|-------------|
| `mean` | Mean of residuals |
| `std_dev` | Standard deviation |
| `min` / `max` | Range of residuals |
| `mse` | Mean squared error |
| `mae` | Mean absolute error |

## Mathematical Details

### OLS / Ridge: Normal Equation

Solves (XᵀX + λI)β = Xᵀy via Gaussian elimination with partial pivoting.

- λ = 0 → OLS (exact solution)
- λ > 0 → Ridge (shrinks coefficients toward zero)

### Lasso: Proximal Gradient Descent

Applies soft-thresholding: βⱼ ← sign(zⱼ) · max(|zⱼ| - α·lr, 0)

This is the proximal operator for the L1 norm, promoting sparsity in the coefficient vector.

### R² Score

R² = 1 - SS_res / SS_tot, where:
- SS_res = Σ(yᵢ - ŷᵢ)²
- SS_tot = Σ(yᵢ - ȳ)²

## Testing

```bash
cargo test
```

14 comprehensive tests covering:
- Perfect fit on linear data
- R² = 1 for noiseless data
- Residual computation and analysis
- Ridge shrinks coefficient magnitudes
- Lasso promotes sparsity on relevant features
- MSE/MAE correctness
- Linear system solver (identity and general)
- Soft-thresholding operator
- Prediction on new data
- Zero-variance edge case

## License

MIT
