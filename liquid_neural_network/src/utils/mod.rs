// src/utils/mod.rs

pub mod data_processing;

pub use data_processing::{
    calculate_features,
    calculate_mae,
    calculate_mse,
    fetch_forex_data,
    normalize_features_targets,
};