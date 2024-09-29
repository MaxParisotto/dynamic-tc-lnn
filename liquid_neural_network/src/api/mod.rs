// src/api/mod.rs

use serde::Deserialize;
use serde::Serialize;
use std::sync::Mutex;
use crate::models::Metrics;

// Define AppStateStruct within the api module
pub struct AppStateStruct {
    pub metrics: Mutex<Metrics>,
}

// Re-export handlers
pub mod handlers;

pub use handlers::get_metrics;