// src/main.rs

use actix_web::{web, App, HttpServer};
use dotenv::dotenv;
use log::{debug, error, info};
use std::sync::Mutex;
use utils::{fetch_forex_data, calculate_features, normalize_features_targets, calculate_mse, calculate_mae};
use models::{LiquidNeuralNetwork, Metrics};
use api::get_metrics;
use api::AppStateStruct;

mod models;
mod api;
mod utils;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
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
        },
        Err(e) => {
            error!("Error fetching Forex data: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Failed to fetch Forex data"));
        },
    };

    // Calculate features and normalize
    let features_targets = calculate_features(&market_data);
    let (normalized_features, normalized_targets) = normalize_features_targets(&features_targets);

    info!("Calculated and normalized features.");

    // Initialize models
    let input_size = normalized_features[0].len(); // Adjust based on actual features
    let initial_neurons = 10; // Example number of neurons

    let mut model_a = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut model_b = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut model_c = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut meta_model = LiquidNeuralNetwork::new(3, initial_neurons); // Meta-model with 3 inputs

    info!("Initialized all models.");

    // Training loop (start from 1 to avoid iteration = 0)
    for iteration in 1..=100 {
        // Example training logic
        // Update models, calculate errors, update metrics...

        // For demonstration, we'll skip actual training logic
        // Update metrics with dummy values
        {
            let mut metrics = app_state.metrics.lock().unwrap();
            metrics.iteration = iteration;
            metrics.mse_a += 0.001;
            metrics.mae_a += 0.0005;
            metrics.mse_b += 0.0012;
            metrics.mae_b += 0.0006;
            metrics.mse_c += 0.0008;
            metrics.mae_c += 0.0004;
            metrics.mse_meta += 0.001;
            metrics.mae_meta += 0.0005;
        }

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