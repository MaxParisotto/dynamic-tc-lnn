// src/utils/data_processing.rs

use serde::{Deserialize};
use reqwest::blocking::Client;
use std::collections::HashMap;
use chrono::NaiveDate;
use crate::models::{MarketData};
use plotters::drawing::IntoDrawingArea;

// Define the structure for time series entries from the API
#[derive(Debug, Deserialize)]
pub struct TimeSeriesEntry {
    #[serde(rename = "1. open")]
    pub open: String,
    #[serde(rename = "2. high")]
    pub high: String,
    #[serde(rename = "3. low")]
    pub low: String,
    #[serde(rename = "4. close")]
    pub close: String,
}

// Define the structure for the API response
#[derive(Debug, Deserialize)]
pub struct ApiResponse {
    #[serde(rename = "Time Series FX (Daily)")]
    pub time_series: HashMap<String, TimeSeriesEntry>,
}

// Function to fetch Forex data from Alpha Vantage
pub fn fetch_forex_data() -> Result<Vec<MarketData>, Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();
    let api_key = std::env::var("ALPHAVANTAGE_API_KEY")?;
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

// Function to calculate features and targets
pub fn calculate_features(data: &[MarketData]) -> Vec<(Vec<f64>, f64)> {
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

// Function to normalize features and targets
pub fn normalize_features_targets(
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

// Function to calculate the average of a slice of f64
pub fn average(errors: &[f64]) -> f64 {
    errors.iter().sum::<f64>() / errors.len() as f64
}

// Function to calculate MSE and MAE
pub fn calculate_metrics(predictions: &[f64], targets: &[f64]) -> (f64, f64) {
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

// Function to plot metrics (optional, can be removed if only using API)
#[allow(dead_code)]
pub fn plot_metrics(
    metric_history: &[f64],
    metric_name: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Specify the PixelFormat as RGBPixel
    let root_area = plotters::prelude::SVGBackend::new(filename, (800, 600)).into_drawing_area();
    root_area.fill(&plotters::prelude::WHITE)?;

    let max_value = metric_history.iter().cloned().fold(f64::MIN, f64::max);
    let min_value = metric_history.iter().cloned().fold(f64::MAX, f64::min);

    let y_range = if (max_value - min_value).abs() < std::f64::EPSILON {
        // If all values are the same, set a default range
        (min_value - 1.0)..(max_value + 1.0)
    } else {
        min_value..max_value
    };

    let mut chart = plotters::prelude::ChartBuilder::on(&root_area)
        .caption(format!("{} Over Iterations", metric_name), ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..metric_history.len(), y_range)?;
    chart.configure_mesh().draw()?;
    chart
        .draw_series(plotters::prelude::LineSeries::new(
            metric_history.iter().enumerate().map(|(x, y)| (x, *y)),
            &plotters::prelude::RED,
        ))?
        .label(metric_name)
        .legend(|(x, y)| plotters::prelude::PathElement::new(vec![(x, y), (x + 20, y)], &plotters::prelude::RED));
    chart.configure_series_labels().draw()?;
    Ok(())
}