use actix::{Actor, StreamHandler};
use actix_web::{web, HttpRequest, HttpResponse, Error}; // Make sure to include HttpResponse
use actix_web_actors::ws;

pub struct MetricsWebSocket;

impl Actor for MetricsWebSocket {
    type Context = ws::WebsocketContext<Self>;
}

// Handle WebSocket messages
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

// The WebSocket route handler
pub async fn metrics_ws(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, Error> {
    ws::start(MetricsWebSocket {}, &req, stream)
}