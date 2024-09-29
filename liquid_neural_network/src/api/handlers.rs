// src/api/handlers.rs

use actix_web::{web, Responder, HttpResponse};
use serde::Serialize;
use crate::api::AppStateStruct;
use crate::models::Metrics;

#[derive(Serialize)]
pub struct ApiResponse {
    pub iteration: usize,
    pub mse_a: f64,
    pub mae_a: f64,
    pub mse_b: f64,
    pub mae_b: f64,
    pub mse_c: f64,
    pub mae_c: f64,
    pub mse_meta: f64,
    pub mae_meta: f64,
}

pub async fn get_metrics(data: web::Data<AppStateStruct>) -> impl Responder {
    let metrics = data.metrics.lock().unwrap();

    // Check if at least one iteration has occurred
    if metrics.iteration < 1 {
        return HttpResponse::BadRequest()
            .body("Requested application data is not configured correctly. View/enable debug logs for more details.");
    }

    let response = ApiResponse {
        iteration: metrics.iteration,
        mse_a: metrics.mse_a,
        mae_a: metrics.mae_a,
        mse_b: metrics.mse_b,
        mae_b: metrics.mae_b,
        mse_c: metrics.mse_c,
        mae_c: metrics.mae_c,
        mse_meta: metrics.mse_meta,
        mae_meta: metrics.mae_meta,
    };

    HttpResponse::Ok().json(response)
}