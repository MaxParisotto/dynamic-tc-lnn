// src/models/network.rs

use serde::{Deserialize, Serialize};
use rand::Rng;
use rand::thread_rng;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde_json;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Metrics {
    pub iteration: usize,
    pub mse_a: f64,
    pub mae_a: f64,
    pub mse_b: f64,
    pub mae_b: f64,
    pub mse_c: f64,
    pub mae_c: f64,
    pub mse_meta: f64,
    pub mae_meta: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LiquidNeuralNetwork {
    pub neurons: Vec<Neuron>,
    // Add other fields as necessary
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Neuron {
    pub state: f64,
    pub bias: f64,
    pub tau: f64,
    pub weights: Vec<f64>,
}

// src/models/network.rs

impl LiquidNeuralNetwork {
    pub fn train(&mut self, _input: &[f64], _target: f64, _dt: f64, _lr: f64) {
        // Implement your training logic here
        // For example, update neuron states, adjust weights, etc.
    }

    pub fn predict(&self) -> f64 {
        // Implement your prediction logic here
        // For example, average neuron outputs
        self.neurons.iter().map(|n| n.state).sum::<f64>() / self.neurons.len() as f64
    }
}

impl Neuron {
    pub fn new(input_size: usize) -> Self {
        let mut rng = thread_rng();
        let limit = (6.0 / input_size as f64).sqrt(); // Xavier initialization
        Neuron {
            state: 0.0,
            bias: rng.gen_range(-limit..limit),
            tau: rng.gen_range(0.1..1.0), // Avoid zero tau
            weights: (0..input_size).map(|_| rng.gen_range(-limit..limit)).collect(),
        }
    }
}