use crate::models::MarketData;
use chrono::NaiveDate;
use std::env;
use log::{error, info};
use std::fs;

/// Fetches Forex data from the Alpha Vantage API or local fallback file
pub async fn fetch_forex_data() -> Result<Vec<MarketData>, Box<dyn std::error::Error>> {
    let api_key = env::var("ALPHAVANTAGE_API_KEY")
        .expect("ALPHAVANTAGE_API_KEY must be set in the environment or .env file");

    let url = format!(
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=EURUSD&outputsize=full&apikey={}",
        api_key
    );

    // Attempt to fetch data from the API
    match reqwest::get(&url).await {
        Ok(resp) => {
            let json: serde_json::Value = resp.json().await?;
            if let Some(time_series) = json.get("Time Series (Daily)") {
                let mut market_data = Vec::new();

                for (date_str, data) in time_series.as_object().unwrap() {
                    let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")?;
                    let open = data["1. open"].as_str().unwrap().parse::<f64>()?;
                    let high = data["2. high"].as_str().unwrap().parse::<f64>()?;
                    let low = data["3. low"].as_str().unwrap().parse::<f64>()?;
                    let close = data["4. close"].as_str().unwrap().parse::<f64>()?;

                    market_data.push(MarketData {
                        date,
                        open,
                        high,
                        low,
                        close,
                    });
                }

                // Sort data by date ascending
                market_data.sort_by_key(|d| d.date);

                // Save data to a local file for future use
                save_data_to_file(&market_data)?;
                Ok(market_data)
            } else {
                // Log error and use local data if available
                error!("Missing 'Time Series (Daily)' in response: {:?}", json);
                load_local_data()
            }
        }
        Err(err) => {
            error!("Failed to fetch Forex data: {:?}", err);
            load_local_data()
        }
    }
}

/// Load previously fetched Forex data from a local file
fn load_local_data() -> Result<Vec<MarketData>, Box<dyn std::error::Error>> {
    info!("Attempting to load Forex data from local file...");
    let data = fs::read_to_string("market_data.json")?;
    let market_data: Vec<MarketData> = serde_json::from_str(&data)?;
    Ok(market_data)
}

/// Save Forex data to a local file
fn save_data_to_file(data: &Vec<MarketData>) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = serde_json::to_string(data)?;
    fs::write("market_data.json", serialized)?;
    Ok(())
}

/// Calculates features and targets from market data
pub fn calculate_features(market_data: &[MarketData]) -> Vec<(Vec<f64>, f64)> {
    let mut features_targets = Vec::new();

    for window in market_data.windows(2) {
        if let [prev, current] = window {
            // Example Feature: Price change
            let price_change = current.close - prev.close;
            // Additional features can be added here
            let features = vec![price_change];
            let target = current.close;
            features_targets.push((features, target));
        }
    }

    features_targets
}

/// Normalizes features and targets using min-max scaling
pub fn normalize_features_targets(
    features_targets: &[(Vec<f64>, f64)],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut normalized_features = Vec::new();
    let mut normalized_targets = Vec::new();

    // For simplicity, perform min-max normalization on features and targets separately
    let feature_length = features_targets[0].0.len();

    // Initialize min and max vectors
    let mut feature_min = vec![f64::INFINITY; feature_length];
    let mut feature_max = vec![f64::NEG_INFINITY; feature_length];
    let mut target_min = f64::INFINITY;
    let mut target_max = f64::NEG_INFINITY;

    // Find min and max for each feature and target
    for (features, target) in features_targets {
        for (i, feature) in features.iter().enumerate() {
            if *feature < feature_min[i] {
                feature_min[i] = *feature;
            }
            if *feature > feature_max[i] {
                feature_max[i] = *feature;
            }
        }
        if *target < target_min {
            target_min = *target;
        }
        if *target > target_max {
            target_max = *target;
        }
    }

    // Normalize features and targets
    for (features, target) in features_targets {
        let mut normalized = Vec::with_capacity(feature_length);
        for (i, feature) in features.iter().enumerate() {
            if (feature_max[i] - feature_min[i]).abs() < std::f64::EPSILON {
                normalized.push(0.0); // Avoid division by zero
            } else {
                normalized.push((feature - feature_min[i]) / (feature_max[i] - feature_min[i]));
            }
        }
        normalized_features.push(normalized);

        if (target_max - target_min).abs() < std::f64::EPSILON {
            normalized_targets.push(0.0);
        } else {
            normalized_targets.push((target - target_min) / (target_max - target_min));
        }
    }

    (normalized_features, normalized_targets)
}

/// Calculates Mean Squared Error between prediction and target
pub fn calculate_mse(prediction: &f64, target: &f64) -> f64 {
    (*prediction - *target).powi(2)
}

/// Calculates Mean Absolute Error between prediction and target
pub fn calculate_mae(prediction: &f64, target: &f64) -> f64 {
    (*prediction - *target).abs()
}