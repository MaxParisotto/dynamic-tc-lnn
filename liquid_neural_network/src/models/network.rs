// src/models/network.rs

use serde::{Deserialize, Serialize};
use rand::Rng;
use rand::thread_rng;

// Structure to hold metrics
#[derive(Serialize, Clone)]
pub struct Metrics {
    pub iteration: usize,
    pub mse: f64,
    pub mae: f64,
}

// Neuron structure
#[derive(Serialize, Deserialize, Clone)]
pub struct Neuron {
    pub state: f64,
    pub bias: f64,
    pub tau: f64,
    pub weights: Vec<f64>,
}

impl Neuron {
    pub fn new(input_size: usize) -> Self {
        let mut rng = thread_rng();
        let limit = (6.0 / input_size as f64).sqrt(); // Xavier initialization limit
        Neuron {
            state: 0.0,
            bias: rng.gen_range(-limit..limit),
            tau: rng.gen_range(0.1..1.0), // Avoid zero tau
            weights: (0..input_size)
                .map(|_| rng.gen_range(-limit..limit))
                .collect(),
        }
    }
}

// Liquid Neural Network structure
#[derive(Serialize, Deserialize, Clone)]
pub struct LiquidNeuralNetwork {
    pub neurons: Vec<Neuron>,
    pub input_size: usize,
}

impl LiquidNeuralNetwork {
    pub fn new(input_size: usize, initial_neurons: usize) -> Self {
        let neurons = (0..initial_neurons)
            .map(|_| Neuron::new(input_size))
            .collect();
        LiquidNeuralNetwork {
            neurons,
            input_size,
        }
    }

    pub fn activation(x: f64) -> f64 {
        x.tanh()
    }

    pub fn forward(&mut self, input: &[f64], dt: f64) {
        for neuron in &mut self.neurons {
            let weighted_sum: f64 = neuron
                .weights
                .iter()
                .zip(input)
                .map(|(w, &i)| w * i)
                .sum::<f64>()
                + neuron.bias;

            let dx = (-neuron.state + Self::activation(weighted_sum)) / neuron.tau;
            neuron.state += dx * dt;
        }
    }

    pub fn predict(&self) -> f64 {
        self.neurons
            .iter()
            .map(|n| n.state)
            .sum::<f64>()
            / self.neurons.len() as f64
    }

    pub fn train(&mut self, input: &[f64], target: f64, dt: f64, lr: f64) {
        self.forward(input, dt);
        let prediction = self.predict();
        let error = target - prediction;

        for neuron in &mut self.neurons {
            let activation_derivative = 1.0 - Self::activation(neuron.state).powi(2);
            let delta = error * activation_derivative;

            neuron.bias += lr * delta;
            for (w, &i) in neuron.weights.iter_mut().zip(input) {
                *w += lr * delta * i;
            }
        }
    }

    pub fn should_add_neuron(&self, recent_errors: &[f64], threshold: f64) -> bool {
        recent_errors.len() >= 5
            && (recent_errors.iter().sum::<f64>() / recent_errors.len() as f64) > threshold
    }

    pub fn add_neuron(&mut self) {
        let new_neuron = Neuron::new(self.input_size);
        self.neurons.push(new_neuron);
    }
}