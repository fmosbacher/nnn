use std::collections::VecDeque;

use rand::{distributions::Uniform, Rng};

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
}

impl Matrix {
    pub fn from(rows: usize, cols: usize, values: &[f64]) -> Self {
        Self {
            rows,
            cols,
            values: values.to_vec(),
        }
    }

    pub fn rand(rows: usize, cols: usize) -> Self {
        let rng = rand::thread_rng();
        Self {
            rows,
            cols,
            values: rng
                .sample_iter(Uniform::new_inclusive(-1.0, 1.0))
                .take(rows * cols)
                .collect::<Vec<f64>>(),
        }
    }

    pub fn powi(&self, p: i32) -> Self {
        self.map(|x| x.powi(p))
    }

    pub fn mean(&self) -> f64 {
        self.values.iter().sum::<f64>() / (self.values.len() as f64)
    }

    pub fn dot(&self, rhs: &Self) -> Self {
        let mut values = vec![];
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * rhs.get(k, j);
                }
                values.push(sum);
            }
        }
        return Matrix::from(self.rows, rhs.cols, &values);
    }

    pub fn transpose(&self) -> Self {
        let mut values = vec![];
        for col in 0..self.cols {
            for row in 0..self.rows {
                values.push(self.get(row, col));
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            values,
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.broadcast_op(rhs, |a, b| a + b)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.broadcast_op(rhs, |a, b| a - b)
    }

    pub fn mult(&self, rhs: &Self) -> Self {
        self.broadcast_op(rhs, |a, b| a * b)
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            values: self.values.iter().map(|x| f(*x)).collect(),
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }

    fn broadcast_op(&self, rhs: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        let mut values = vec![];
        let rows = self.rows.max(rhs.rows);
        let cols = self.cols.max(rhs.cols);

        for row in 0..rows {
            for col in 0..cols {
                values.push(f(
                    self.get(row % self.rows, col % self.cols),
                    rhs.get(row % rhs.rows, col % rhs.cols),
                ));
            }
        }

        Self::from(rows, cols, &values)
    }
}

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    // neurons: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(neurons: &[usize]) -> Self {
        Self {
            // neurons: neurons.to_vec(),
            weights: neurons
                .iter()
                .zip(neurons.iter().skip(1))
                .map(|(n, m)| Matrix::rand(*n, *m))
                .collect(),
            biases: neurons
                .iter()
                .skip(1)
                .map(|m| Matrix::rand(1, *m))
                .collect(),
        }
    }

    pub fn forward(&self, xs: &Matrix) -> Matrix {
        let mut ys = xs.clone();
        self.weights
            .iter()
            .zip(self.biases.iter())
            .for_each(|(ws, bs)| ys = ys.dot(ws).add(bs).map(sigmoid));
        ys
    }

    pub fn update(&mut self, xs: &Matrix, ys: &Matrix, learning_rate: f64) {
        let mut zs = vec![xs.clone()];
        let mut acs = vec![xs.clone()];

        for (layer, (ws, bs)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let z = acs[layer].dot(ws).add(bs);
            zs.push(z.clone());
            acs.push(z.map(sigmoid))
        }

        let err = acs[acs.len() - 1].sub(&ys);
        let mut delta = err;
        let mut ws_updates = VecDeque::new();
        let mut bs_updates = VecDeque::new();

        for (layer, _) in self.weights.iter().enumerate().rev() {
            delta = if let Some(next_ws) = self.weights.get(layer + 1) {
                delta
                    .dot(&next_ws.transpose())
                    .mult(&zs[layer + 1].map(deriv_sigmoid))
            } else {
                delta.mult(&zs[layer + 1].map(deriv_sigmoid))
            };
            ws_updates.push_front(acs[layer].transpose().dot(&delta));
            bs_updates.push_front(
                Matrix::from(xs.dims().0, 1, &vec![1.0; xs.dims().0])
                    .transpose()
                    .dot(&delta),
            );
        }

        let lr = Matrix::from(1, 1, &[learning_rate]);

        for (ws, update) in self.weights.iter_mut().zip(ws_updates.iter()) {
            *ws = ws.sub(&update.mult(&lr))
        }

        for (bs, update) in self.biases.iter_mut().zip(bs_updates.iter()) {
            *bs = bs.sub(&update.mult(&lr));
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn deriv_sigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
