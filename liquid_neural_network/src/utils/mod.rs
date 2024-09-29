// src/utils/mod.rs

pub mod data_processing;

pub use data_processing::{
    fetch_forex_data,
    calculate_features,
    normalize_features_targets,
    average,
    calculate_metrics,
};