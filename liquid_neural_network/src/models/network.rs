use serde::{Deserialize, Serialize};
use rand::Rng;
use rand::thread_rng;
use log::debug; // Imported the debug macro

/// Metrics to track during training
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

impl Metrics {
    /// Update metrics based on new values for each model
    #[allow(dead_code)]
    pub fn update_metrics(&mut self, mse_a: f64, mae_a: f64, mse_b: f64, mae_b: f64, mse_c: f64, mae_c: f64, mse_meta: f64, mae_meta: f64) {
        self.mse_a = mse_a;
        self.mae_a = mae_a;
        self.mse_b = mse_b;
        self.mae_b = mae_b;
        self.mse_c = mse_c;
        self.mae_c = mae_c;
        self.mse_meta = mse_meta;
        self.mae_meta = mae_meta;
    }
}

/// A simple neuron model
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Neuron {
    pub state: f64,
    pub bias: f64,
    pub tau: f64,
    pub weights: Vec<f64>,
}

/// Liquid Neural Network consisting of multiple neurons
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LiquidNeuralNetwork {
    pub neurons: Vec<Neuron>,
    // Add other fields as necessary
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
    pub fn train(&mut self, input: &[f64], target: f64, _dt: f64, lr: f64) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Calculate neuron's raw output
            let raw_output: f64 =
                input.iter().zip(neuron.weights.iter()).map(|(x, w)| x * w).sum::<f64>() + neuron.bias;
            // Apply activation function (ReLU)
            let output = relu(raw_output);

            // Calculate error
            let error = target - output;

            // Update weights and bias
            for (w, x) in neuron.weights.iter_mut().zip(input.iter()) {
                *w += lr * error * x;
            }
            neuron.bias += lr * error;

            // Update neuron state
            neuron.state = output;

            // Log updated weights and bias (debug level)
            debug!(
                "Neuron {} - Updated Weights: {:?}, Updated Bias: {:.6}",
                i, neuron.weights, neuron.bias
            );
        }
    }

    /// Adjusts the learning rate based on a decay factor and the current iteration.
    #[allow(dead_code)]
    pub fn adjust_learning_rate(&self, initial_lr: f64, decay: f64, iteration: usize) -> f64 {
        initial_lr * (1.0 / (1.0 + decay * iteration as f64))
    }

    /// Generates a prediction based on the current state of the network.
    pub fn predict(&self) -> f64 {
        // Average the states of all neurons as the prediction
        self.neurons.iter().map(|n| n.state).sum::<f64>() / self.neurons.len() as f64
    }

    /// Generates a prediction based on custom input features.
    pub fn predict_with_input(&self, input: &[f64]) -> f64 {
        let output: f64 = input
            .iter()
            .zip(self.neurons.iter())
            .map(|(x, neuron)| relu(x * neuron.weights[0] + neuron.bias)) // Simplified; adjust as needed
            .sum::<f64>()
            / self.neurons.len() as f64;
        output
    }

    /// Scales the number of neurons in the network dynamically.
    #[allow(dead_code)]
    pub fn scale_neurons(&mut self, new_size: usize, input_size: usize) {
        let current_size = self.neurons.len();

        if new_size > current_size {
            for _ in 0..(new_size - current_size) {
                self.neurons.push(Neuron::new(input_size));
            }
        } else if new_size < current_size {
            self.neurons.truncate(new_size);
        }
    }

    /// Saves the network state to a file in JSON format.
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serde_json::to_string(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Loads the network state from a JSON file.
    #[allow(dead_code)]
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let network = serde_json::from_str(&data)?;
        Ok(network)
    }
}

/// Activation functions

/// ReLU activation function
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

/// Sigmoid activation function
#[allow(dead_code)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Input normalization function to ensure training stability
#[allow(dead_code)]
pub fn normalize_input(input: &[f64]) -> Vec<f64> {
    let max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = input.iter().cloned().fold(f64::INFINITY, f64::min);
    input.iter().map(|x| (x - min) / (max - min)).collect()
}

/// A simple neuron implementation
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