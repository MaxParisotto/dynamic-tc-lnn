use serde::{Deserialize, Serialize};
use reqwest::blocking::Client;
use std::collections::HashMap;
use std::env;
use dotenv::dotenv;
use chrono::NaiveDate;
use rand::Rng;
use rand::thread_rng;
use plotters::prelude::*;
use plotters::series::LineSeries; // Explicitly import LineSeries

#[derive(Debug, Deserialize)]
struct TimeSeriesEntry {
    #[serde(rename = "1. open")]
    open: String,
    #[serde(rename = "2. high")]
    high: String,
    #[serde(rename = "3. low")]
    low: String,
    #[serde(rename = "4. close")]
    close: String,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    #[serde(rename = "Time Series FX (Daily)")]
    time_series: HashMap<String, TimeSeriesEntry>,
}

#[derive(Debug)]
struct MarketData {
    date: NaiveDate,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

fn fetch_forex_data() -> Result<Vec<MarketData>, Box<dyn std::error::Error>> {
    dotenv().ok();
    let api_key = env::var("ALPHAVANTAGE_API_KEY")?;
    let url = format!(
        "https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&apikey={}&outputsize=full",
        api_key
    );

    let client = Client::new();
    let response = client.get(&url).send()?;
    let response_text = response.text()?;

    let api_response: ApiResponse = serde_json::from_str(&response_text)?;

    let mut market_data = Vec::new();

    for (date_str, entry) in api_response.time_series {
        let date = NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")?;
        let open = entry.open.parse::<f64>()?;
        let high = entry.high.parse::<f64>()?;
        let low = entry.low.parse::<f64>()?;
        let close = entry.close.parse::<f64>()?;

        market_data.push(MarketData {
            date,
            open,
            high,
            low,
            close,
        });
    }

    market_data.sort_by_key(|data| data.date);

    Ok(market_data)
}

fn calculate_features(data: &[MarketData]) -> Vec<(Vec<f64>, f64)> {
    let mut features_targets = Vec::new();

    // Ensure there is enough data
    if data.len() < 6 {
        return features_targets; // Not enough data to compute features
    }

    for i in 6..data.len() {
        let mut input_features = Vec::new();
        for j in (i - 5)..i {
            // Safely calculate percentage changes
            let pct_change_open = if data[j - 1].open != 0.0 {
                (data[j].open - data[j - 1].open) / data[j - 1].open
            } else {
                0.0
            };

            let pct_change_high = if data[j - 1].high != 0.0 {
                (data[j].high - data[j - 1].high) / data[j - 1].high
            } else {
                0.0
            };

            let pct_change_low = if data[j - 1].low != 0.0 {
                (data[j].low - data[j - 1].low) / data[j - 1].low
            } else {
                0.0
            };

            let pct_change_close = if data[j - 1].close != 0.0 {
                (data[j].close - data[j - 1].close) / data[j - 1].close
            } else {
                0.0
            };

            input_features.extend_from_slice(&[
                pct_change_open,
                pct_change_high,
                pct_change_low,
                pct_change_close,
            ]);
        }

        // Target: Next day's percentage change in close price
        let target_pct_change = if data[i - 1].close != 0.0 {
            (data[i].close - data[i - 1].close) / data[i - 1].close
        } else {
            0.0
        };
        let target = target_pct_change;

        features_targets.push((input_features, target));
    }

    features_targets
}

fn normalize_features_targets(
    features_targets: &[(Vec<f64>, f64)],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut features_matrix = Vec::new();
    let mut targets = Vec::new();

    for (features, target) in features_targets {
        features_matrix.push(features.clone());
        targets.push(*target);
    }

    // Flatten features for normalization
    let all_features: Vec<f64> = features_matrix.iter().flatten().cloned().collect();

    // Calculate mean and standard deviation
    let mean = all_features.iter().sum::<f64>() / all_features.len() as f64;
    let std_dev = (all_features
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / all_features.len() as f64)
        .sqrt();

    // Avoid division by zero
    let std_dev = if std_dev == 0.0 { 1.0 } else { std_dev };

    // Normalize features
    let normalized_features: Vec<Vec<f64>> = features_matrix
        .iter()
        .map(|features| {
            features
                .iter()
                .map(|x| (x - mean) / std_dev)
                .collect::<Vec<f64>>()
        })
        .collect();

    // Normalize targets
    let target_mean = targets.iter().sum::<f64>() / targets.len() as f64;
    let target_std_dev = (targets
        .iter()
        .map(|x| (x - target_mean).powi(2))
        .sum::<f64>()
        / targets.len() as f64)
        .sqrt();
    let target_std_dev = if target_std_dev == 0.0 { 1.0 } else { target_std_dev };
    let normalized_targets: Vec<f64> = targets
        .iter()
        .map(|x| (x - target_mean) / target_std_dev)
        .collect();

    (normalized_features, normalized_targets)
}

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
            weights: (0..input_size)
                .map(|_| rng.gen_range(-limit..limit))
                .collect(),
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
        LiquidNeuralNetwork {
            neurons,
            input_size,
        }
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
        self.neurons
            .iter()
            .map(|n| n.state)
            .sum::<f64>()
            / self.neurons.len() as f64
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
        recent_errors.len() >= 5
            && (recent_errors.iter().sum::<f64>() / recent_errors.len() as f64) > threshold
    }

    fn add_neuron(&mut self) {
        let new_neuron = Neuron::new(self.input_size);
        self.neurons.push(new_neuron);
    }
}

fn average(errors: &[f64]) -> f64 {
    errors.iter().sum::<f64>() / errors.len() as f64
}

fn calculate_metrics(predictions: &[f64], targets: &[f64]) -> (f64, f64) {
    let mse = predictions
        .iter()
        .zip(targets)
        .map(|(pred, target)| (pred - target).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;

    let mae = predictions
        .iter()
        .zip(targets)
        .map(|(pred, target)| (pred - target).abs())
        .sum::<f64>()
        / predictions.len() as f64;

    (mse, mae)
}

// Add the plot_metrics function
fn plot_metrics(
    metric_history: &[f64],
    metric_name: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Specify the PixelFormat as RGBPixel
    let root_area = SVGBackend::new(filename, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let max_value = metric_history.iter().cloned().fold(f64::MIN, f64::max);
    let min_value = metric_history.iter().cloned().fold(f64::MAX, f64::min);

    let y_range = if (max_value - min_value).abs() < std::f64::EPSILON {
        // If all values are the same, set a default range
        (min_value - 1.0)..(max_value + 1.0)
    } else {
        min_value..max_value
    };

    let mut chart = ChartBuilder::on(&root_area)
        .caption(format!("{} Over Iterations", metric_name), ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..metric_history.len(), y_range)?;
    chart.configure_mesh().draw()?;
    chart
        .draw_series(LineSeries::new(
            metric_history.iter().enumerate().map(|(x, y)| (x, *y)),
            &RED,
        ))?
        .label(metric_name)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart.configure_series_labels().draw()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch market data from Alpha Vantage
    let market_data = fetch_forex_data()?;
    let features_targets = calculate_features(&market_data);
    let (inputs, targets) = normalize_features_targets(&features_targets);

    // Split data into training and testing sets
    let split_index = (inputs.len() as f64 * 0.8) as usize;
    let (train_inputs, test_inputs) = inputs.split_at(split_index);
    let (train_targets, test_targets) = targets.split_at(split_index);

    // Initialize network parameters
    let input_size = train_inputs[0].len();
    let initial_neurons = 5;
    let dt = 0.1;
    let learning_rate = 0.01;
    let error_threshold = 0.05;
    let max_iterations = 100;

    let mut network = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut recent_errors = Vec::new();
    let mut iteration = 0;

    // Initialize metric history vectors
    let mut mse_history = Vec::new();
    let mut mae_history = Vec::new();

    loop {
        iteration += 1;
        let mut epoch_errors = Vec::new();
        let mut predictions = Vec::new();

        for (input, target) in train_inputs.iter().zip(train_targets.iter()) {
            network.train(input, *target, dt, learning_rate);

            let prediction = network.predict();
            predictions.push(prediction);
            let error = (*target - prediction).abs();
            epoch_errors.push(error);
            recent_errors.push(error);

            if recent_errors.len() > 5 {
                recent_errors.remove(0);

                if network.should_add_neuron(&recent_errors, error_threshold) {
                    network.add_neuron();
                    println!(
                        "Iteration {}: Added a neuron. Total neurons: {}",
                        iteration,
                        network.neurons.len()
                    );
                }
            }
        }

        let average_error = average(&epoch_errors);
        let (mse, mae) = calculate_metrics(&predictions, &train_targets[..predictions.len()]);
        println!(
            "Iteration {}: Training - Average Error: {:.6}, MSE: {:.6}, MAE: {:.6}",
            iteration, average_error, mse, mae
        );

        // Store metrics for plotting
        mse_history.push(mse);
        mae_history.push(mae);

        // Break condition
        if average_error < 0.01 || iteration >= max_iterations {
            println!("Training completed after {} iterations.", iteration);
            break;
        }
    }

    // Evaluate on test data
    let mut test_predictions = Vec::new();
    for input in test_inputs.iter() {
        network.forward(input, dt);
        let prediction = network.predict();
        test_predictions.push(prediction);
    }

    let (test_mse, test_mae) = calculate_metrics(&test_predictions, &test_targets);
    println!("Test Performance Metrics:");
    println!("MSE: {:.6}, MAE: {:.6}", test_mse, test_mae);

    // Plot metrics
    plot_metrics(&mse_history, "MSE", "mse_over_iterations.png")?;
    plot_metrics(&mae_history, "MAE", "mae_over_iterations.png")?;

    Ok(())
}