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

impl LiquidNeuralNetwork {
    /// Creates a new LiquidNeuralNetwork with the specified number of input features and neurons.
    pub fn new(input_size: usize, initial_neurons: usize) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..initial_neurons {
            neurons.push(Neuron::new(input_size));
        }
        LiquidNeuralNetwork { neurons }
    }

    /// Trains the network with the given input, target, time step, and learning rate.
    pub fn train(&mut self, _input: &[f64], _target: f64, _dt: f64, _lr: f64) {
        // Implement your training logic here
        // For example, update neuron states, adjust weights, etc.
        // Currently unused variables; implement as needed
        // _input, _target, _dt, _lr
    }

    /// Generates a prediction based on the current state of the network.
    pub fn predict(&self) -> f64 {
        // Implement your prediction logic here
        // For example, average neuron outputs
        self.neurons.iter().map(|n| n.state).sum::<f64>() / self.neurons.len() as f64
    }

    /// Saves the network state to a file in JSON format.
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Loads the network state from a JSON file.
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let network = serde_json::from_reader(reader)?;
        Ok(network)
    }

    // Add methods for adjusting neurons based on errors
}

impl Neuron {
    /// Creates a new Neuron with randomized weights and bias.
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