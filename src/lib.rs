//! # ternary-regression
//!
//! Linear regression with ternary `{-1, 0, +1}` features.
//!
//! Supports ordinary least squares, ridge regression (L2), lasso approximation (L1),
//! residual analysis, and R² computation.

/// Result of fitting a linear regression model.
#[derive(Debug, Clone)]
pub struct RegressionResult {
    /// Fitted coefficients.
    pub coefficients: Vec<f64>,
    /// Intercept (bias) term.
    pub intercept: f64,
    /// R² score on training data.
    pub r_squared: f64,
    /// Residuals on training data.
    pub residuals: Vec<f64>,
    /// Number of iterations (for iterative methods).
    pub iterations: usize,
}

/// Configuration for regression training.
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// L2 regularization strength (ridge). 0 = OLS.
    pub l2_penalty: f64,
    /// L1 regularization strength (lasso approximation). 0 = no L1.
    pub l1_penalty: f64,
    /// Learning rate for iterative lasso.
    pub learning_rate: f64,
    /// Max iterations for iterative methods.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        RegressionConfig {
            l2_penalty: 0.0,
            l1_penalty: 0.0,
            learning_rate: 0.01,
            max_iter: 10000,
            tol: 1e-10,
        }
    }
}

/// Ternary linear regression model.
pub struct TernaryLinearRegression {
    config: RegressionConfig,
}

impl TernaryLinearRegression {
    /// Create with default config (OLS).
    pub fn new() -> Self {
        TernaryLinearRegression {
            config: RegressionConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: RegressionConfig) -> Self {
        TernaryLinearRegression { config }
    }

    /// Fit the model to ternary-feature data using the normal equation (OLS or ridge).
    ///
    /// - `x`: feature matrix (n × d), each entry in {-1, 0, +1}
    /// - `y`: target values
    pub fn fit(&self, x: &[Vec<i8>], y: &[f64]) -> RegressionResult {
        if self.config.l1_penalty > 0.0 {
            self.fit_iterative(x, y)
        } else {
            self.fit_normal(x, y)
        }
    }

    /// OLS or Ridge via the normal equation: (XᵀX + λI)β = Xᵀy
    fn fit_normal(&self, x: &[Vec<i8>], y: &[f64]) -> RegressionResult {
        let n = x.len();
        let d = x[0].len();
        assert!(n > 0 && d > 0);
        assert_eq!(x.len(), y.len());

        // XᵀX (d × d)
        let mut xtx = vec![vec![0.0; d]; d];
        for i in 0..n {
            for j in 0..d {
                for k in 0..d {
                    xtx[j][k] += x[i][j] as f64 * x[i][k] as f64;
                }
            }
        }

        // Add L2 penalty to diagonal
        for j in 0..d {
            xtx[j][j] += self.config.l2_penalty;
        }

        // Xᵀy (d × 1)
        let mut xty = vec![0.0; d];
        for i in 0..n {
            for j in 0..d {
                xty[j] += x[i][j] as f64 * y[i];
            }
        }

        // Solve via Gaussian elimination with partial pivoting
        let coefficients = solve_linear_system(&xtx, &xty);

        // Compute intercept as mean(y) - mean(X) · coefficients
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
        let mean_x: Vec<f64> = (0..d)
            .map(|j| x.iter().map(|row| row[j] as f64).sum::<f64>() / n as f64)
            .collect();
        let intercept = mean_y - mean_x.iter().zip(&coefficients).map(|(m, c)| m * c).sum::<f64>();

        // Residuals and R²
        let predicted: Vec<f64> = x.iter().map(|xi| Self::predict_with(&coefficients, intercept, xi)).collect();
        let residuals: Vec<f64> = y.iter().zip(&predicted).map(|(yi, pi)| yi - pi).collect();
        let r_squared = compute_r_squared(y, &predicted);

        RegressionResult {
            coefficients,
            intercept,
            r_squared,
            residuals,
            iterations: 0,
        }
    }

    /// Iterative fit (for lasso / L1 regularization via coordinate descent).
    fn fit_iterative(&self, x: &[Vec<i8>], y: &[f64]) -> RegressionResult {
        let n = x.len();
        let d = x[0].len();

        // Start with OLS solution
        let mut result = self.fit_normal(x, y);
        let mut beta = result.coefficients.clone();
        let mut intercept = result.intercept;
        let lr = self.config.learning_rate;
        let l1 = self.config.l1_penalty;

        // Soft-thresholding for L1 via proximal gradient
        for _ in 0..self.config.max_iter {
            let mut gradients = vec![0.0; d];
            let mut grad_b = 0.0;

            for (xi, &yi) in x.iter().zip(y.iter()) {
                let pred = Self::predict_with(&beta, intercept, xi);
                let residual = pred - yi;
                for (j, &xij) in xi.iter().enumerate() {
                    gradients[j] += 2.0 * residual * xij as f64 / n as f64;
                }
                grad_b += 2.0 * residual / n as f64;
            }

            // Gradient step
            for j in 0..d {
                let updated = beta[j] - lr * gradients[j];
                // Soft-threshold (proximal operator for L1)
                beta[j] = soft_threshold(updated, lr * l1);
            }
            intercept -= lr * grad_b;
        }

        let predicted: Vec<f64> = x.iter().map(|xi| Self::predict_with(&beta, intercept, xi)).collect();
        let residuals: Vec<f64> = y.iter().zip(&predicted).map(|(yi, pi)| yi - pi).collect();
        let r_squared = compute_r_squared(y, &predicted);

        RegressionResult {
            coefficients: beta,
            intercept,
            r_squared,
            residuals,
            iterations: self.config.max_iter,
        }
    }

    /// Predict using stored coefficients and intercept.
    fn predict_with(coefficients: &[f64], intercept: f64, x: &[i8]) -> f64 {
        coefficients
            .iter()
            .zip(x.iter())
            .map(|(c, &xi)| c * xi as f64)
            .sum::<f64>()
            + intercept
    }

    /// Predict target values for new data.
    pub fn predict(result: &RegressionResult, x: &[Vec<i8>]) -> Vec<f64> {
        x.iter()
            .map(|xi| Self::predict_with(&result.coefficients, result.intercept, xi))
            .collect()
    }
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    // Augmented matrix
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            continue; // Skip singular columns
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if aug[i][i].abs() < 1e-15 {
            x[i] = 0.0;
            continue;
        }
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

/// Compute R² = 1 - SS_res / SS_tot.
fn compute_r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|&yi| (yi - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yi, &pi)| (yi - pi).powi(2))
        .sum();
    if ss_tot < 1e-15 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Soft-thresholding operator: sign(z) * max(|z| - λ, 0).
fn soft_threshold(z: f64, lambda: f64) -> f64 {
    if z > lambda {
        z - lambda
    } else if z < -lambda {
        z + lambda
    } else {
        0.0
    }
}

/// Compute mean squared error.
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f64>()
        / n
}

/// Compute mean absolute error.
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .sum::<f64>()
        / n
}

/// Residual analysis summary.
#[derive(Debug, Clone)]
pub struct ResidualAnalysis {
    /// Mean of residuals.
    pub mean: f64,
    /// Standard deviation of residuals.
    pub std_dev: f64,
    /// Minimum residual.
    pub min: f64,
    /// Maximum residual.
    pub max: f64,
    /// Mean squared error.
    pub mse: f64,
    /// Mean absolute error.
    pub mae: f64,
}

/// Perform residual analysis.
pub fn analyze_residuals(residuals: &[f64]) -> ResidualAnalysis {
    let n = residuals.len() as f64;
    let mean = residuals.iter().sum::<f64>() / n;
    let variance = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min = residuals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = residuals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mse = residuals.iter().map(|r| r * r).sum::<f64>() / n;
    let mae = residuals.iter().map(|r| r.abs()).sum::<f64>() / n;

    ResidualAnalysis {
        mean,
        std_dev,
        min,
        max,
        mse,
        mae,
    }
}

/// Convenience: fit ridge regression.
pub fn ridge_regression(x: &[Vec<i8>], y: &[f64], alpha: f64) -> RegressionResult {
    let model = TernaryLinearRegression::with_config(RegressionConfig {
        l2_penalty: alpha,
        ..Default::default()
    });
    model.fit(x, y)
}

/// Convenience: fit lasso regression (iterative).
pub fn lasso_regression(x: &[Vec<i8>], y: &[f64], alpha: f64) -> RegressionResult {
    let model = TernaryLinearRegression::with_config(RegressionConfig {
        l1_penalty: alpha,
        learning_rate: 0.01,
        max_iter: 5000,
        tol: 1e-10,
        l2_penalty: 0.0,
    });
    model.fit(x, y)
}

/// Convenience: fit OLS.
pub fn ols_regression(x: &[Vec<i8>], y: &[f64]) -> RegressionResult {
    TernaryLinearRegression::new().fit(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_fit_linear_data() {
        // y = 2*x[0] + 3*x[1] + 1
        let x: Vec<Vec<i8>> = vec![
            vec![-1, -1],
            vec![-1, 0],
            vec![-1, 1],
            vec![0, -1],
            vec![0, 0],
            vec![0, 1],
            vec![1, -1],
            vec![1, 0],
            vec![1, 1],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi[0] as f64 + 3.0 * xi[1] as f64 + 1.0).collect();

        let result = ols_regression(&x, &y);
        assert!((result.coefficients[0] - 2.0).abs() < 0.1, "coef[0] = {}", result.coefficients[0]);
        assert!((result.coefficients[1] - 3.0).abs() < 0.1, "coef[1] = {}", result.coefficients[1]);
        assert!(
            (result.intercept - 1.0).abs() < 0.1,
            "intercept = {}",
            result.intercept
        );
    }

    #[test]
    fn test_r_squared_perfect_fit() {
        let x: Vec<Vec<i8>> = vec![
            vec![1], vec![-1], vec![0], vec![1], vec![-1],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 5.0 * xi[0] as f64).collect();

        let result = ols_regression(&x, &y);
        assert!(
            result.r_squared > 0.99,
            "R² should be ~1 for perfect linear data, got {}",
            result.r_squared
        );
    }

    #[test]
    fn test_residual_computation() {
        let x: Vec<Vec<i8>> = vec![vec![1], vec![-1], vec![0]];
        let y: Vec<f64> = vec![3.0, -3.0, 0.0]; // y = 3*x[0]

        let result = ols_regression(&x, &y);
        for r in &result.residuals {
            assert!(
                r.abs() < 0.01,
                "Residual should be near 0 for perfect data: {}",
                r
            );
        }
    }

    #[test]
    fn test_residual_analysis() {
        let residuals = vec![0.1, -0.2, 0.05, -0.1, 0.15];
        let analysis = analyze_residuals(&residuals);
        assert!(analysis.mean.abs() < 1.0);
        assert!(analysis.std_dev > 0.0);
        assert!(analysis.mse > 0.0);
        assert!(analysis.mae > 0.0);
        assert!(analysis.max >= analysis.min);
    }

    #[test]
    fn test_ridge_shrinks_coefficients() {
        let x: Vec<Vec<i8>> = vec![
            vec![1, 1],
            vec![1, -1],
            vec![-1, 1],
            vec![-1, -1],
            vec![0, 0],
        ];
        let y: Vec<f64> = vec![3.0, 1.0, -1.0, -3.0, 0.0];

        let ols = ols_regression(&x, &y);
        let ridge = ridge_regression(&x, &y, 10.0);

        let ols_norm: f64 = ols.coefficients.iter().map(|c| c * c).sum::<f64>().sqrt();
        let ridge_norm: f64 = ridge.coefficients.iter().map(|c| c * c).sum::<f64>().sqrt();

        assert!(
            ridge_norm <= ols_norm + 0.01,
            "Ridge coefficients ({}) should be smaller than OLS ({})",
            ridge_norm,
            ols_norm
        );
    }

    #[test]
    fn test_prediction_correctness() {
        let x: Vec<Vec<i8>> = vec![
            vec![1], vec![-1], vec![0], vec![1], vec![-1],
        ];
        let y: Vec<f64> = vec![2.0, -2.0, 0.0, 2.0, -2.0]; // y = 2*x[0]

        let result = ols_regression(&x, &y);
        let preds = TernaryLinearRegression::predict(&result, &x);

        for (pred, &true_y) in preds.iter().zip(y.iter()) {
            assert!(
                (pred - true_y).abs() < 0.1,
                "Prediction {} should be near {}",
                pred,
                true_y
            );
        }
    }

    #[test]
    fn test_mse() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        assert!((mse(&y_true, &y_pred)).abs() < 1e-10);

        let y_pred2 = vec![2.0, 3.0, 4.0];
        assert!((mse(&y_true, &y_pred2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mae() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![2.0, 3.0, 4.0];
        assert!((mae(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(0.5, 0.1) - 0.4).abs() < 1e-10);
        assert!((soft_threshold(-0.5, 0.1) + 0.4).abs() < 1e-10);
        assert!((soft_threshold(0.05, 0.1)).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 5.0];
        let x = solve_linear_system(&a, &b);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_general() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = solve_linear_system(&a, &b);
        // Verify Ax = b
        let r0 = 2.0 * x[0] + 1.0 * x[1];
        let r1 = 1.0 * x[0] + 3.0 * x[1];
        assert!((r0 - 5.0).abs() < 1e-10);
        assert!((r1 - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_lasso_sparsity() {
        // y = 3*x[0] + 0*x[1], lasso should shrink x[1] toward 0
        let x: Vec<Vec<i8>> = vec![
            vec![1, 1],
            vec![1, -1],
            vec![-1, 1],
            vec![-1, -1],
            vec![1, 0],
            vec![-1, 0],
        ];
        let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi[0] as f64).collect();

        let result = lasso_regression(&x, &y, 0.5);
        assert!(
            result.coefficients[0].abs() > result.coefficients[1].abs(),
            "Lasso should prefer the relevant feature: coefs = {:?}",
            result.coefficients
        );
    }

    #[test]
    fn test_r_squared_zero_variance() {
        // All y are the same → R² should be 1.0 (or handled gracefully)
        let y_true = vec![5.0, 5.0, 5.0];
        let y_pred = vec![5.0, 5.0, 5.0];
        let r2 = compute_r_squared(&y_true, &y_pred);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_new_data() {
        let x: Vec<Vec<i8>> = vec![vec![1], vec![-1], vec![0]];
        let y: Vec<f64> = vec![4.0, -4.0, 0.0];
        let result = ols_regression(&x, &y);

        let new_x = vec![vec![1], vec![-1]];
        let preds = TernaryLinearRegression::predict(&result, &new_x);
        assert!((preds[0] - 4.0).abs() < 0.1);
        assert!((preds[1] + 4.0).abs() < 0.1);
    }
}
