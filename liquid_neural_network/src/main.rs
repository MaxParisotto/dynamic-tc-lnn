use serde::{Serialize, Deserialize};
use rand::Rng;
use rand::thread_rng;

#[derive(Serialize, Deserialize)]
struct Neuron {
    state: f64,
    bias: f64,
    tau: f64,
    weights: Vec<f64>,
}

impl Neuron {
    fn new(input_size: usize) -> Self {
        let mut rng = thread_rng();
        let limit = (6.0 / input_size as f64).sqrt(); // Xavier initialization limit
        Neuron {
            state: 0.0,
            bias: rng.gen_range(-limit..limit),
            tau: rng.gen_range(0.1..1.0), // Avoid zero tau
            weights: (0..input_size).map(|_| rng.gen_range(-limit..limit)).collect(),
        }
    }
}

struct LiquidNeuralNetwork {
    neurons: Vec<Neuron>,
    input_size: usize,
}

impl LiquidNeuralNetwork {
    fn new(input_size: usize, initial_neurons: usize) -> Self {
        let neurons = (0..initial_neurons)
            .map(|_| Neuron::new(input_size))
            .collect();
        LiquidNeuralNetwork { neurons, input_size }
    }

    fn activation(x: f64) -> f64 {
        x.tanh()
    }

    fn forward(&mut self, input: &[f64], dt: f64) {
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

    fn predict(&self) -> f64 {
        self.neurons.iter().map(|n| n.state).sum::<f64>() / self.neurons.len() as f64
    }

    fn train(&mut self, input: &[f64], target: f64, dt: f64, lr: f64) {
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

    fn should_add_neuron(&self, recent_errors: &[f64], threshold: f64) -> bool {
        recent_errors.len() >= 5 && (recent_errors.iter().sum::<f64>() / recent_errors.len() as f64) > threshold
    }

    fn add_neuron(&mut self) {
        let new_neuron = Neuron::new(self.input_size);
        self.neurons.push(new_neuron);
    }
}

fn average(errors: &[f64]) -> f64 {
    errors.iter().sum::<f64>() / errors.len() as f64
}

fn main() {
    // Initialize network parameters
    let input_size = 10;          // Example input size
    let initial_neurons = 5;      // Starting with 5 neurons
    let dt = 0.1;                 // Time step adjusted for better convergence
    let learning_rate = 0.01;     // Increased learning rate
    let error_threshold = 0.05;   // Threshold for adding neurons

    let mut network = LiquidNeuralNetwork::new(input_size, initial_neurons);

    // Expanded data stream with more data points
    let data_stream = vec![
        (vec![0.5; input_size], 1.0),
        (vec![0.2; input_size], 0.5),
        (vec![0.8; input_size], 1.5),
        (vec![0.1; input_size], 0.3),
        (vec![0.6; input_size], 1.2),
        (vec![0.3; input_size], 0.7),
        (vec![0.9; input_size], 1.8),
        (vec![0.4; input_size], 0.9),
        // Add more data tuples as needed
    ];

    let mut recent_errors = Vec::new();
    let mut iteration = 0;
    let max_iterations = 1000; // Set a maximum number of iterations if desired

    loop {
        iteration += 1;
        let mut epoch_errors = Vec::new();

        for (input, target) in &data_stream {
            network.train(input, *target, dt, learning_rate);

            let prediction = network.predict();
            let error = (*target - prediction).abs();
            epoch_errors.push(error);
            recent_errors.push(error);

            if recent_errors.len() > 5 {
                recent_errors.remove(0);

                if network.should_add_neuron(&recent_errors, error_threshold) {
                    network.add_neuron();
                    println!("Iteration {}: Added a neuron. Total neurons: {}", iteration, network.neurons.len());
                }
            }
        }

        let average_error = average(&epoch_errors);
        println!("Iteration {}: Average Error: {:.4}", iteration, average_error);

        // Break condition (optional)
        if average_error < 0.01 || iteration >= max_iterations {
            println!("Training completed after {} iterations.", iteration);
            break;
        }
    }
}