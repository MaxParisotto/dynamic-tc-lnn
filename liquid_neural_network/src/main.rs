use dotenv::dotenv;
use log::{error, info};
use std::sync::Mutex;
use utils::{calculate_features, calculate_mae, calculate_mse, fetch_forex_data, normalize_features_targets};
use models::{LiquidNeuralNetwork, Metrics};
use rand::prelude::SliceRandom;
use actix_web::{web};
use crate::server::AppStateStruct;

mod ws;
mod models;
mod api;
mod utils;
mod server;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    env_logger::init();

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

    let app_state = web::Data::new(AppStateStruct {
        metrics: Mutex::new(metrics),
    });

    // Fetch and preprocess Forex data
    let market_data = match fetch_forex_data().await {
        Ok(data) => {
            info!("Fetched {} market data points.", data.len());
            data
        }
        Err(e) => {
            error!("Error fetching Forex data: {}", e);
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Failed to fetch Forex data"));
        }
    };

    // Feature normalization
    let features_targets = calculate_features(&market_data);
    let (normalized_features, normalized_targets) = normalize_features_targets(&features_targets);

    info!("Calculated and normalized features.");

    // Initialize the models
    let input_size = normalized_features[0].len();
    let initial_neurons = 10;
    let mut model_a = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut model_b = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let mut model_c = LiquidNeuralNetwork::new(input_size, initial_neurons);
    let meta_model = LiquidNeuralNetwork::new(3, initial_neurons);

    info!("Initialized all models.");

    // Training loop
    for iteration in 1..=100 {
        info!("Starting iteration {}", iteration);

        // Shuffle data
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

        // Iterate over data, train models, and collect errors
        for (features, target) in combined.iter() {
            model_a.train(features, **target, 0.1, 0.01);
            model_b.train(features, **target, 0.1, 0.01);
            model_c.train(features, **target, 0.1, 0.01);

            let pred_a = model_a.predict();
            let pred_b = model_b.predict();
            let pred_c = model_c.predict();

            let meta_input = vec![pred_a, pred_b, pred_c];
            let pred_meta = meta_model.predict_with_input(&meta_input);

            total_mse_a += calculate_mse(&pred_a, target);
            total_mae_a += calculate_mae(&pred_a, target);
            total_mse_b += calculate_mse(&pred_b, target);
            total_mae_b += calculate_mae(&pred_b, target);
            total_mse_c += calculate_mse(&pred_c, target);
            total_mae_c += calculate_mae(&pred_c, target);
            total_mse_meta += calculate_mse(&pred_meta, target);
            total_mae_meta += calculate_mae(&pred_meta, target);
        }

        // Update metrics for this iteration
        let data_len = normalized_features.len() as f64;
        {
            let mut metrics = app_state.metrics.lock().unwrap();
            metrics.iteration = iteration;
            metrics.mse_a = total_mse_a / data_len;
            metrics.mae_a = total_mae_a / data_len;
            metrics.mse_b = total_mse_b / data_len;
            metrics.mae_b = total_mae_b / data_len;
            metrics.mse_c = total_mse_c / data_len;
            metrics.mae_c = total_mae_c / data_len;
            metrics.mse_meta = total_mse_meta / data_len;
            metrics.mae_meta = total_mae_meta / data_len;
        }

        info!("Iteration {}: Metrics updated", iteration);

        if iteration % 10 == 0 {
            save_models(&model_a, &model_b, &model_c, &meta_model, iteration);
        }
    }

    info!("Training loop completed.");

    // Start the Actix-web server
    server::run_server(app_state).await
}

// Helper function to save models
fn save_models(model_a: &LiquidNeuralNetwork, model_b: &LiquidNeuralNetwork, model_c: &LiquidNeuralNetwork, meta_model: &LiquidNeuralNetwork, iteration: usize) {
    if let Err(e) = model_a.save_to_file("model_a.json") {
        error!("Failed to save Model A at iteration {}: {}", iteration, e);
    }
    if let Err(e) = model_b.save_to_file("model_b.json") {
        error!("Failed to save Model B at iteration {}: {}", iteration, e);
    }
    if let Err(e) = model_c.save_to_file("model_c.json") {
        error!("Failed to save Model C at iteration {}: {}", iteration, e);
    }
    if let Err(e) = meta_model.save_to_file("meta_model.json") {
        error!("Failed to save Meta Model at iteration {}: {}", iteration, e);
    }
    info!("Saved models at iteration {}", iteration);
}