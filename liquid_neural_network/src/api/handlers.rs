// src/api/handlers.rs

use actix_web::{web, Responder, HttpResponse};

// Import AppStateStruct from mod.rs
use crate::api::AppStateStruct;

pub async fn get_metrics(data: web::Data<AppStateStruct>) -> impl Responder {
    match data.metrics.lock() {
        Ok(metrics) => HttpResponse::Ok().json(&*metrics),
        Err(_) => HttpResponse::InternalServerError().body("Failed to retrieve metrics"),
    }
}