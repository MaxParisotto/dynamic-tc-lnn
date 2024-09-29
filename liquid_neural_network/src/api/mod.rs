// src/api/mod.rs

pub mod handlers;

use std::sync::{Arc, Mutex};
use crate::models::Metrics;

// Structure to hold application state
pub struct AppStateStruct {
    pub metrics: Arc<Mutex<Metrics>>,
}

// Re-export handlers for easy access
pub use handlers::get_metrics;