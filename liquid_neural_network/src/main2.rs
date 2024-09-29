// src/main.rs

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use std::sync::{Arc, Mutex};
use tokio::task;
use log::{info, error};
use env_logger;

// Import modules
mod api;
mod models;
mod utils;

use api::{AppStateStruct, get_metrics};
use models::{LiquidNeuralNetwork, Metrics};
use utils::{fetch_forex_data, calculate_features, normalize_features_targets, calculate_mse, calculate_mae};

#[actix_web::main]
async fn main() -> Result<(), std::io::Error> {
    // Initialize environment variables
    dotenv::dotenv().ok();

    // Initialize the logger
    env_logger::init();

    // Fetch market data asynchronously
    let market_data = match fetch_forex_data().await {
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
    let save_interval = 10; // Save models every 10 iterations

    // Initialize the Liquid Neural Networks
    let mut model_a = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut model_b = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut model_c = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut meta_model = LiquidNeuralNetwork::new(3, initial_neurons); // Meta-model with 3 inputs

    let mut iteration = 0;

    // Initialize shared metrics state
    let metrics = Arc::new(Mutex::new(Metrics {
        iteration: 0,
        mse_a: 0.0,
        mae_a: 0.0,
        mse_b: 0.0,
        mae_b: 0.0,
        mse_c: 0.0,
        mae_c: 0.0,
        mse_meta: 0.0,
        mae_meta: 0.0,
    }));

    // Clone metrics and training data for the training loop
    let metrics_clone = metrics.clone();
    let train_inputs_clone = train_inputs.clone();
    let train_targets_clone = train_targets.clone();
    let test_inputs_clone = test_inputs.clone();
    let test_targets_clone = test_targets.clone();

    // Spawn the training loop in a separate blocking thread
    task::spawn_blocking(move || {
        info!("Training loop started.");

        loop {
            iteration += 1;

            // Ensure we don't exceed the training data
            if iteration > train_inputs_clone.len() {
                info!("Reached end of training data.");
                break;
            }

            let input = &train_inputs_clone[iteration - 1];
            let target = train_targets_clone[iteration - 1];

            // Train base models
            model_a.train(input, target, dt, learning_rate);
            model_b.train(input, target, dt, learning_rate);
            model_c.train(input, target, dt, learning_rate);

            // Predict with base models
            let pred_a = model_a.predict();
            let pred_b = model_b.predict();
            let pred_c = model_c.predict();

            // Calculate errors
            let error_a = target - pred_a;
            let error_b = target - pred_b;
            let error_c = target - pred_c;

            // Collect errors for meta-model input
            let combined_errors = vec![error_a, error_b, error_c];

            // Train meta-model on the errors
            let meta_target = 0.0; // Define as needed
            meta_model.train(&combined_errors, meta_target, dt, learning_rate);

            // Calculate and record metrics
            let mse_a_val = calculate_mse(&pred_a, &target);
            let mae_a_val = calculate_mae(&pred_a, &target);
            let mse_b_val = calculate_mse(&pred_b, &target);
            let mae_b_val = calculate_mae(&pred_b, &target);
            let mse_c_val = calculate_mse(&pred_c, &target);
            let mae_c_val = calculate_mae(&pred_c, &target);
            let mse_meta_val = calculate_mse(&meta_model.predict(), &meta_target);
            let mae_meta_val = calculate_mae(&meta_model.predict(), &meta_target);

            // Update shared metrics state
            {
                let mut metrics_lock = metrics_clone.lock().unwrap();
                metrics_lock.iteration = iteration;
                metrics_lock.mse_a = mse_a_val;
                metrics_lock.mae_a = mae_a_val;
                metrics_lock.mse_b = mse_b_val;
                metrics_lock.mae_b = mae_b_val;
                metrics_lock.mse_c = mse_c_val;
                metrics_lock.mae_c = mae_c_val;
                metrics_lock.mse_meta = mse_meta_val;
                metrics_lock.mae_meta = mae_meta_val;
            }

            // Save models periodically
            if iteration % save_interval == 0 {
                if let Err(e) = model_a.save_to_file("model_a.json") {
                    eprintln!("Failed to save Model A: {}", e);
                }
                if let Err(e) = model_b.save_to_file("model_b.json") {
                    eprintln!("Failed to save Model B: {}", e);
                }
                if let Err(e) = model_c.save_to_file("model_c.json") {
                    eprintln!("Failed to save Model C: {}", e);
                }
                if let Err(e) = meta_model.save_to_file("meta_model.json") {
                    eprintln!("Failed to save Meta-Model: {}", e);
                }
                info!("Models saved at iteration {}", iteration);
            }

            // Logging
            info!(
                "Iteration {}: Training - MSE_A: {:.6}, MAE_A: {:.6}, MSE_B: {:.6}, MAE_B: {:.6}, MSE_C: {:.6}, MAE_C: {:.6}, MSE_Meta: {:.6}, MAE_Meta: {:.6}",
                iteration, mse_a_val, mae_a_val, mse_b_val, mae_b_val, mse_c_val, mae_c_val, mse_meta_val, mae_meta_val
            );

            // Break condition
            if iteration >= max_iterations {
                info!("Training completed after {} iterations.", iteration);
                break;
            }
        }
        // Optionally, evaluate on test data
        // Implement as needed
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