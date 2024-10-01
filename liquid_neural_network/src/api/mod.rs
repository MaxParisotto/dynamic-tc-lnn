// src/api/mod.rs
use std::sync::Mutex;
use crate::models::Metrics;

/// Application state structure containing shared metrics
pub struct AppStateStruct {
    pub metrics: Mutex<Metrics>,
}

/// Re-export handlers
pub mod handlers;

// pub use handlers::get_metrics;