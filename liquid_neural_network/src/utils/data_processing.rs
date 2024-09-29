// src/utils/data_processing.rs

use serde::Deserialize;
use reqwest::blocking::Client;
use std::collections::HashMap;
use chrono::NaiveDate;
use crate::models::MarketData;

// Existing structs and functions...

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

// **Missing Function Implementations**

/// Calculates features from market data.
/// Modify this function based on the specific features you want to extract.
pub fn calculate_features(market_data: &[MarketData]) -> Vec<(Vec<f64>, f64)> {
    let mut features_targets = Vec::new();

    for data in market_data.windows(2) {
        if let [prev, current] = data {
            // Example Feature: Price change
            let price_change = current.close - prev.close;
            // Features could include more indicators like moving averages, RSI, etc.
            let features = vec![price_change];
            let target = current.close;
            features_targets.push((features, target));
        }
    }

    features_targets
}

/// Normalizes features and targets.
/// This example normalizes features to have zero mean and unit variance.
pub fn normalize_features_targets(features_targets: &[(Vec<f64>, f64)]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let feature_length = features_targets[0].0.len();
    let mut means = vec![0.0; feature_length];
    let mut variances = vec![0.0; feature_length];

    // Calculate means
    for (features, _) in features_targets {
        for i in 0..feature_length {
            means[i] += features[i];
        }
    }
    for mean in &mut means {
        *mean /= features_targets.len() as f64;
    }

    // Calculate variances
    for (features, _) in features_targets {
        for i in 0..feature_length {
            variances[i] += (features[i] - means[i]).powi(2);
        }
    }
    for variance in &mut variances {
        *variance /= features_targets.len() as f64;
        if *variance == 0.0 {
            *variance = 1.0; // Prevent division by zero
        }
    }

    // Normalize features
    let mut normalized_features = Vec::new();
    let mut normalized_targets = Vec::new();

    for (features, target) in features_targets {
        let mut normalized = Vec::new();
        for i in 0..feature_length {
            normalized.push((features[i] - means[i]) / variances[i].sqrt());
        }
        normalized_features.push(normalized);
        normalized_targets.push(*target);
    }

    (normalized_features, normalized_targets)
}

/// Calculates the average of a slice of f64 values.
pub fn average(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculates Mean Squared Error and Mean Absolute Error.
pub fn calculate_metrics(predictions: &[f64], targets: &[f64]) -> (f64, f64) {
    let mse = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).powi(2))
        .sum::<f64>() / predictions.len() as f64;

    let mae = predictions.iter()
        .zip(targets.iter())
        .map(|(p, t)| (t - p).abs())
        .sum::<f64>() / predictions.len() as f64;

    (mse, mae)
}