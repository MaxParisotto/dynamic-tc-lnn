// src/api/mod.rs

use std::sync::{Arc, Mutex};
use crate::models::Metrics;

pub struct AppStateStruct {
    pub metrics: Arc<Mutex<Metrics>>,
}

pub mod handlers;

// Re-export handlers
pub use handlers::get_metrics;