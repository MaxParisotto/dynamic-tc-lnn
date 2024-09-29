// src/models/mod.rs

pub mod network;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MarketData {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

// Re-export network components
pub use network::{LiquidNeuralNetwork, Metrics};