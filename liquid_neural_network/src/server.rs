use actix::{Actor, StreamHandler, AsyncContext};  // Add AsyncContext here
use actix_web::{web, App, HttpServer, HttpRequest, Error, HttpResponse};
use actix_files as fs;
use actix_cors::Cors;
use actix_web_actors::ws;
use std::sync::Mutex;
use crate::models::Metrics;

// Application state that holds metrics inside a Mutex for thread safety
pub struct AppStateStruct {
    pub metrics: Mutex<Metrics>,  // Ensure this exists for real-time metrics
}

// WebSocket actor for handling real-time metrics updates
pub struct MetricsWebSocket {
    pub app_state: web::Data<AppStateStruct>,  // Add application state inside the WebSocket
}

impl Actor for MetricsWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Set up interval to send metrics every 5 seconds
        let app_state_clone = self.app_state.clone(); // Clone the app state for use inside the interval

        ctx.run_interval(std::time::Duration::from_secs(5), move |_, ctx| {
            let metrics = app_state_clone.metrics.lock().unwrap();  // Get actual metrics from app state

            let serialized_metrics = serde_json::to_string(&*metrics).unwrap();  // Serialize the metrics
            ctx.text(serialized_metrics);  // Send real-time metrics to the frontend
        });
    }
}

// Implement StreamHandler for WebSocket to manage incoming WebSocket messages
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MetricsWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),  // Handle ping
            Ok(ws::Message::Text(text)) => ctx.text(format!("Echo: {}", text)),  // Echo received text
            Ok(ws::Message::Binary(bin)) => ctx.binary(bin),  // Echo binary data
            _ => (),  // Ignore other message types
        }
    }
}

// WebSocket route handler to establish a WebSocket connection
async fn metrics_ws(req: HttpRequest, stream: web::Payload, app_state: web::Data<AppStateStruct>) -> Result<HttpResponse, Error> {
    ws::start(MetricsWebSocket { app_state }, &req, stream)
}

// Function to run the Actix web server
pub async fn run_server(app_state: web::Data<AppStateStruct>) -> std::io::Result<()> {
    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())  // Enable permissive CORS for frontend-backend interaction
            .app_data(app_state.clone())  // Pass application state
            .route("/metrics_ws", web::get().to(metrics_ws))  // Route for WebSocket connection
            .service(fs::Files::new("/", "./frontend").index_file("index.html"))  // Serve frontend files
    })
    .bind(("127.0.0.1", 8080))?  // Bind to localhost:8080
    .run()
    .await
}