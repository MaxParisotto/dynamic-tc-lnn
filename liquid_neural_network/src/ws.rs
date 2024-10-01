use actix::{Actor, StreamHandler};
use actix_web::{HttpRequest, HttpResponse, Error};
use actix_web_actors::ws;
use log::info;

// Define the WebSocket actor
pub struct MetricsWebSocket;

impl Actor for MetricsWebSocket {
    type Context = ws::WebsocketContext<Self>;
}

// Handle incoming WebSocket messages
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MetricsWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => {
                info!("Received message: {}", text);
                ctx.text(format!("Echo: {}", text));
            },
            _ => (),
        }
    }
}

// Function to start the WebSocket connection
pub async fn metrics_ws(req: HttpRequest, stream: actix_web::web::Payload) -> Result<HttpResponse, Error> {
    ws::start(MetricsWebSocket {}, &req, stream)
}