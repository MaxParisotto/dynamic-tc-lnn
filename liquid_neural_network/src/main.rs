// src/main.rs

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use std::sync::{Arc, Mutex};
use tokio::task;

// Import modules
mod api;
mod models;
mod utils;

use api::{AppStateStruct, get_metrics};
use models::{LiquidNeuralNetwork, Metrics};
use utils::{fetch_forex_data, calculate_features, normalize_features_targets, average, calculate_metrics};

#[actix_web::main]
async fn main() -> Result<(), std::io::Error> {
    // Initialize environment variables
    dotenv::dotenv().ok();

    // Fetch market data
    let market_data = match fetch_forex_data() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error fetching Forex data: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Failed to fetch Forex data"));
        }
    };
    let features_targets = calculate_features(&market_data);
    let (inputs, targets) = normalize_features_targets(&features_targets);

    // Split data into training and testing sets
    let split_index = (inputs.len() as f64 * 0.8) as usize;
    let train_inputs = inputs[..split_index].to_vec(); // Owned Vec<f64>
    let test_inputs = inputs[split_index..].to_vec();  // Owned Vec<f64>
    let train_targets = targets[..split_index].to_vec(); // Owned Vec<f64>
    let test_targets = targets[split_index..].to_vec();  // Owned Vec<f64>

    // Initialize network parameters
    let input_size = train_inputs[0].len();
    let initial_neurons = 5;
    let dt = 0.1;
    let learning_rate = 0.01;
    let error_threshold = 0.05;
    let max_iterations = 100;

    // Initialize the Liquid Neural Network
    let network = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut recent_errors = Vec::new();
    let mut iteration = 0;

    // Initialize metric history vectors
    let mut mse_history = Vec::new();
    let mut mae_history = Vec::new();

    // Initialize shared metrics state
    let metrics = Arc::new(Mutex::new(Metrics {
        iteration: 0,
        mse: 0.0,
        mae: 0.0,
    }));

    // Clone metrics and training data for the training loop
    let metrics_clone = metrics.clone();
    let train_inputs_clone = train_inputs.clone();
    let train_targets_clone = train_targets.clone();
    let test_inputs_clone = test_inputs.clone();
    let test_targets_clone = test_targets.clone();

    // Spawn the training loop in a separate blocking thread
    task::spawn_blocking(move || {
        let mut network = network;
        loop {
            iteration += 1;
            let mut epoch_errors = Vec::new();
            let mut predictions = Vec::new();

            for (input, target) in train_inputs_clone.iter().zip(train_targets_clone.iter()) {
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
            let (mse, mae) = calculate_metrics(&predictions, &train_targets_clone[..predictions.len()]);
            println!(
                "Iteration {}: Training - Average Error: {:.6}, MSE: {:.6}, MAE: {:.6}",
                iteration, average_error, mse, mae
            );

            // Store metrics for plotting
            mse_history.push(mse);
            mae_history.push(mae);

            // Update shared metrics state
            {
                let mut metrics_lock = metrics_clone.lock().unwrap();
                metrics_lock.iteration = iteration;
                metrics_lock.mse = mse;
                metrics_lock.mae = mae;
            }

            // Optional: Plot metrics at each iteration
            // Uncomment the following lines if you wish to generate plots in real-time
            /*
            if iteration % 10 == 0 {
                if let Err(e) = utils::plot_metrics(&mse_history, "MSE", "mse_over_iterations.svg") {
                    eprintln!("Error plotting MSE: {}", e);
                }
                if let Err(e) = utils::plot_metrics(&mae_history, "MAE", "mae_over_iterations.svg") {
                    eprintln!("Error plotting MAE: {}", e);
                }
            }
            */

            // Break condition
            if average_error < 0.01 || iteration >= max_iterations {
                println!("Training completed after {} iterations.", iteration);
                break;
            }
        }

        // Evaluate on test data
        let mut test_predictions = Vec::new();
        for input in test_inputs_clone.iter() {
            network.forward(input, dt);
            let prediction = network.predict();
            test_predictions.push(prediction);
        }

        let (test_mse, test_mae) = calculate_metrics(&test_predictions, &test_targets_clone);
        println!("Test Performance Metrics:");
        println!("MSE: {:.6}, MAE: {:.6}", test_mse, test_mae);

        // Optionally, update the metrics one last time
        {
            let mut metrics_lock = metrics_clone.lock().unwrap();
            metrics_lock.iteration = iteration;
            metrics_lock.mse = test_mse;
            metrics_lock.mae = test_mae;
        }

        // Optionally, plot final metrics
        /*
        if let Err(e) = utils::plot_metrics(&mse_history, "MSE", "mse_over_iterations_final.svg") {
            eprintln!("Error plotting final MSE: {}", e);
        }
        if let Err(e) = utils::plot_metrics(&mae_history, "MAE", "mae_over_iterations_final.svg") {
            eprintln!("Error plotting final MAE: {}", e);
        }
        */
    });

    // Set up Actix-web server
    let app_state = web::Data::new(AppStateStruct {
        metrics: metrics.clone(),
    });

    HttpServer::new(move || {
        // Configure CORS to allow requests from frontend
        let cors = Cors::permissive(); // For development; restrict in production

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .route("/metrics", web::get().to(get_metrics))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}