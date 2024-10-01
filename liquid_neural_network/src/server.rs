use actix::{Actor, StreamHandler};
use actix_web::{web, App, HttpServer, HttpRequest, Error, HttpResponse};
use actix_files as fs;
use actix_cors::Cors;
use actix_web_actors::ws;

// In server.rs or its appropriate module
use std::sync::Mutex;
use crate::models::Metrics;  // Assuming Metrics is in models

pub struct AppStateStruct {
    pub metrics: Mutex<Metrics>,  // Ensure this exists for real-time metrics
}

// Define the MetricsWebSocket struct
pub struct MetricsWebSocket;

// Implement the Actor trait for MetricsWebSocket
impl Actor for MetricsWebSocket {
    type Context = ws::WebsocketContext<Self>;
}

// Implement StreamHandler for MetricsWebSocket to handle incoming WebSocket messages
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MetricsWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => ctx.text(format!("Echo: {}", text)),
            Ok(ws::Message::Binary(bin)) => ctx.binary(bin),
            _ => (),
        }
    }
}

// WebSocket route handler
async fn metrics_ws(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, Error> {
    ws::start(MetricsWebSocket {}, &req, stream)
}

// Running the Actix web server
pub async fn run_server(app_state: web::Data<AppStateStruct>) -> std::io::Result<()> {
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())  // Enable CORS
            .app_data(app_state.clone())
            .route("/metrics_ws", web::get().to(metrics_ws))  // WebSocket route
            .service(fs::Files::new("/", "./frontend").index_file("index.html")) // Serve frontend
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}