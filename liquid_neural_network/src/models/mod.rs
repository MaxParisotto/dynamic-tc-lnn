// src/models/mod.rs

pub mod network;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

// Structure to hold market data
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MarketData {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

// Remove this line to avoid duplicate definition
// pub use MarketData;

// Re-export network components
pub use network::{LiquidNeuralNetwork, Metrics};