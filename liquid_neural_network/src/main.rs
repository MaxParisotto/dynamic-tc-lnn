// src/main.rs

use actix_web::{web, App, HttpServer};
use dotenv::dotenv;
use log::{error, info}; // Removed 'debug' as it's unused in this file
use std::sync::Mutex;
use utils::{
    calculate_features, calculate_mae, calculate_mse, fetch_forex_data, normalize_features_targets,
};
use models::{LiquidNeuralNetwork, Metrics};
use api::{get_metrics, AppStateStruct};
use rand::prelude::SliceRandom; // Imported SliceRandom for shuffle method

mod models;
mod api;
mod utils;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Load environment variables from .env file
    dotenv().ok();
    // Initialize the logger
    env_logger::init();

    // Initialize metrics
    let metrics = Metrics {
        iteration: 0,
        mse_a: 0.0,
        mae_a: 0.0,
        mse_b: 0.0,
        mae_b: 0.0,
        mse_c: 0.0,
        mae_c: 0.0,
        mse_meta: 0.0,
        mae_meta: 0.0,
    };

    // Instantiate AppStateStruct from the api module
    let app_state = web::Data::new(AppStateStruct {
        metrics: Mutex::new(metrics),
    });

    // Fetch Forex data
    let market_data = match fetch_forex_data().await {
        Ok(data) => {
            info!("Fetched {} market data points.", data.len());
            data
        }
        Err(e) => {
            error!("Error fetching Forex data: {}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to fetch Forex data",
            ));
        }
    };

    // Calculate features and normalize
    let features_targets = calculate_features(&market_data);
    let (mut normalized_features, normalized_targets) = normalize_features_targets(&features_targets);

    info!("Calculated and normalized features.");

    // Initialize models
    let input_size = normalized_features[0].len(); // Adjust based on actual features
    let initial_neurons = 10; // Example number of neurons

    let mut model_a = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut model_b = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut model_c = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut meta_model = LiquidNeuralNetwork::new(3, initial_neurons); // Meta-model with 3 inputs

    info!("Initialized all models.");

    // Initialize history for plotting or further analysis (optional)
    let mut metrics_history = Vec::new();

    // Training loop
    for iteration in 1..=100 {
        info!("Starting iteration {}", iteration);

        // Shuffle data each iteration to ensure random order
        let mut combined: Vec<(&Vec<f64>, &f64)> =
            normalized_features.iter().zip(normalized_targets.iter()).collect();
        combined.shuffle(&mut rand::thread_rng());

        let mut total_mse_a = 0.0;
        let mut total_mae_a = 0.0;
        let mut total_mse_b = 0.0;
        let mut total_mae_b = 0.0;
        let mut total_mse_c = 0.0;
        let mut total_mae_c = 0.0;
        let mut total_mse_meta = 0.0;
        let mut total_mae_meta = 0.0;

        for (features, target) in combined.iter() {
            // Train each model
            model_a.train(features, **target, 0.1, 0.01);
            model_b.train(features, **target, 0.1, 0.01);
            model_c.train(features, **target, 0.1, 0.01);

            // Generate predictions
            let pred_a = model_a.predict();
            let pred_b = model_b.predict();
            let pred_c = model_c.predict();

            // Aggregate predictions for meta-model (simple average)
            let meta_input = vec![pred_a, pred_b, pred_c];
            let pred_meta = meta_model.predict_with_input(&meta_input);

            // Calculate errors
            let mse_a = calculate_mse(&pred_a, target);
            let mae_a = calculate_mae(&pred_a, target);

            let mse_b = calculate_mse(&pred_b, target);
            let mae_b = calculate_mae(&pred_b, target);

            let mse_c = calculate_mse(&pred_c, target);
            let mae_c = calculate_mae(&pred_c, target);

            let mse_meta = calculate_mse(&pred_meta, target);
            let mae_meta = calculate_mae(&pred_meta, target);

            // Accumulate errors
            total_mse_a += mse_a;
            total_mae_a += mae_a;
            total_mse_b += mse_b;
            total_mae_b += mae_b;
            total_mse_c += mse_c;
            total_mae_c += mae_c;
            total_mse_meta += mse_meta;
            total_mae_meta += mae_meta;
        }

        // Calculate average errors for this iteration
        let data_len = normalized_features.len() as f64;
        let avg_mse_a = total_mse_a / data_len;
        let avg_mae_a = total_mae_a / data_len;
        let avg_mse_b = total_mse_b / data_len;
        let avg_mae_b = total_mae_b / data_len;
        let avg_mse_c = total_mse_c / data_len;
        let avg_mae_c = total_mae_c / data_len;
        let avg_mse_meta = total_mse_meta / data_len;
        let avg_mae_meta = total_mae_meta / data_len;

        // Update metrics
        {
            let mut metrics = app_state.metrics.lock().unwrap();
            metrics.iteration = iteration;
            metrics.mse_a = avg_mse_a;
            metrics.mae_a = avg_mae_a;
            metrics.mse_b = avg_mse_b;
            metrics.mae_b = avg_mae_b;
            metrics.mse_c = avg_mse_c;
            metrics.mae_c = avg_mae_c;
            metrics.mse_meta = avg_mse_meta;
            metrics.mae_meta = avg_mae_meta;
        }

        // Optionally, store metrics for plotting
        metrics_history.push(app_state.metrics.lock().unwrap().clone());

        info!(
            "Iteration {}: MSE_A={:.6}, MAE_A={:.6}, MSE_B={:.6}, MAE_B={:.6}, MSE_C={:.6}, MAE_C={:.6}, MSE_Meta={:.6}, MAE_Meta={:.6}",
            iteration, avg_mse_a, avg_mae_a, avg_mse_b, avg_mae_b, avg_mse_c, avg_mae_c, avg_mse_meta, avg_mae_meta
        );

        // Periodically save models
        if iteration % 10 == 0 {
            if let Err(e) = model_a.save_to_file("model_a.json") {
                error!("Failed to save Model A: {}", e);
            }
            if let Err(e) = model_b.save_to_file("model_b.json") {
                error!("Failed to save Model B: {}", e);
            }
            if let Err(e) = model_c.save_to_file("model_c.json") {
                error!("Failed to save Model C: {}", e);
            }
            if let Err(e) = meta_model.save_to_file("meta_model.json") {
                error!("Failed to save Meta Model: {}", e);
            }
            info!("Saved models at iteration {}", iteration);
        }
    }

    info!("Completed training loop.");

    // Start Actix-web server
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/metrics", web::get().to(get_metrics))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}