//! Studio and the Yendory Network (docs/NETWORK.md on yendory-hub): the background service, installed
//! by Yendory Hub, that runs the Engine for every program of the suite.
//!
//! Step 1 of moving Studio onto it:
//! - Studio connects when it starts (starting the Network if the Hub installed it).
//! - The model Studio has open is opened in the Network too, so no other program renames or
//!   deletes it underneath Studio.
//! - Chat replies are written by the Network (`model.chat`, streamed token by token).
//! - Studio still trains on its own: after it saves a model, the Network reads it again
//!   (`models.reload`), so replies always come from the saved weights.
//!
//! Without the Network (not installed, not answering), Studio does all of this itself, as before,
//! and Global Settings ▸ Yendory Network says why.
use super::*;
use std::sync::Arc;
use std::time::Duration;
use yendory_client::Client;
use yendory_engine::learn::sampling::{Reply, Sampling};

/// The link to the Network, kept in Studio's state.
#[derive(Default)]
pub(crate) struct Link {
    client: Mutex<Option<Arc<Client>>>,
    status: Mutex<NetStatus>,
    /// The model Studio opened in the Network (Studio's open model).
    registered: Mutex<Option<String>>,
    /// The chat whose reply the Network is writing (its stream), for Stop.
    writing: Mutex<Option<String>>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NetStatus {
    pub connected: bool,
    /// Trying to connect (or start the Network) now.
    pub connecting: bool,
    pub network: Option<String>,
    pub engine: Option<String>,
    pub api: Option<u32>,
    /// Why Studio works without it.
    pub problem: Option<String>,
}

fn set_status(app: &AppHandle, status: NetStatus) {
    let studio = app.state::<Studio>();
    *studio.network.status.lock().unwrap() = status.clone();
    let _ = app.emit("network-status", status);
}

/// The connected Network, if any (a lost connection counts as none).
pub(crate) fn client(studio: &Studio) -> Option<Arc<Client>> {
    let c = studio.network.client.lock().unwrap().clone()?;
    c.is_alive().then_some(c)
}

/// Connects in the background (starting the Network the Hub installed when it isn't running).
pub(crate) fn connect(app: &AppHandle) {
    {
        let studio = app.state::<Studio>();
        let mut s = studio.network.status.lock().unwrap();
        if s.connecting {
            return;
        }
        s.connecting = true;
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let version = app.package_info().version.to_string();
        match Client::start_and_connect("Yendory Studio", &version, Duration::from_secs(15)) {
            Ok(c) => {
                let c = Arc::new(c);
                let hello = c.hello.clone();
                if let Some(events) = c.take_events() {
                    let app = app.clone();
                    std::thread::spawn(move || {
                        for e in events {
                            on_event(&app, &e.name, &e.data);
                        }
                        // The connection is gone (the Network stopped or was updated).
                        lost(&app, "The connection to the Yendory Network was lost");
                    });
                }
                let studio = app.state::<Studio>();
                *studio.network.client.lock().unwrap() = Some(c);
                *studio.network.registered.lock().unwrap() = None;
                set_status(
                    &app,
                    NetStatus {
                        connected: true,
                        network: Some(hello.network),
                        engine: Some(hello.engine),
                        api: Some(hello.api),
                        ..Default::default()
                    },
                );
                // The model already open in Studio is opened there too.
                let open = studio.active.lock().unwrap().as_ref().map(|a| a.meta.id.clone());
                if let Some(id) = open {
                    opened(&app, &id);
                }
            }
            Err(e) => set_status(
                &app,
                NetStatus {
                    problem: Some(format!("{e} Studio works on its own meanwhile.")),
                    ..Default::default()
                },
            ),
        }
    });
}

fn lost(app: &AppHandle, why: &str) {
    let studio = app.state::<Studio>();
    *studio.network.client.lock().unwrap() = None;
    *studio.network.registered.lock().unwrap() = None;
    set_status(
        app,
        NetStatus {
            problem: Some(format!("{why}. Studio works on its own meanwhile.")),
            ..Default::default()
        },
    );
}

#[derive(Clone, Serialize)]
struct Token<'a> {
    chat: &'a str,
    text: &'a str,
}

fn on_event(app: &AppHandle, name: &str, data: &serde_json::Value) {
    match name {
        // A token of the chat reply the Network is writing (its stream is the chat's id).
        "model.token" => {
            if let (Some(chat), Some(text)) = (data["stream"].as_str(), data["text"].as_str()) {
                let text = yendory_engine::nexus::readable(text);
                let _ = app.emit("chat-token", Token { chat, text: &text });
            }
        }
        "models.changed" => {
            let _ = app.emit("network-models-changed", data.clone());
        }
        "network.stopping" => lost(app, "The Yendory Network is stopping"),
        _ => {}
    }
}

/// Studio opened `id`: the Network opens it too (and closes the one Studio had before).
pub(crate) fn opened(app: &AppHandle, id: &str) {
    let studio = app.state::<Studio>();
    let Some(c) = client(&studio) else { return };
    let before = studio.network.registered.lock().unwrap().replace(id.to_string());
    if before.as_deref() == Some(id) {
        return;
    }
    let (app, id) = (app.clone(), id.to_string());
    std::thread::spawn(move || {
        if let Some(old) = before {
            let _ = c.call("models.close", serde_json::json!({"id": old}));
        }
        if let Err(e) = c.call("models.open", serde_json::json!({"id": id})) {
            let _ = app.emit(
                "device-notice",
                format!(
                    "The Yendory Network could not open this model ({e}); Studio writes its replies itself."
                ),
            );
            let studio = app.state::<Studio>();
            let mut r = studio.network.registered.lock().unwrap();
            if r.as_deref() == Some(id.as_str()) {
                *r = None;
            }
        }
    });
}

/// Studio is about to delete `id` (or closed it): the Network lets go of it first.
pub(crate) fn closed(studio: &Studio, id: &str) {
    let Some(c) = client(studio) else { return };
    let mut r = studio.network.registered.lock().unwrap();
    if r.as_deref() == Some(id) {
        *r = None;
        let _ = c.call("models.close", serde_json::json!({"id": id}));
    }
}

/// Studio wrote `id`'s files: the Network reads the model again, so its replies come from the
/// saved weights.
pub(crate) fn saved(app: &AppHandle, id: &str) {
    let studio = app.state::<Studio>();
    let Some(c) = client(&studio) else { return };
    if studio.network.registered.lock().unwrap().as_deref() != Some(id) {
        return;
    }
    if let Err(e) = c.call("models.reload", serde_json::json!({"id": id})) {
        // Replies must not come from old weights: this model's replies are written here instead.
        *studio.network.registered.lock().unwrap() = None;
        let _ = app.emit(
            "device-notice",
            format!(
                "The Yendory Network could not read the saved model ({e}); Studio writes its replies itself."
            ),
        );
    }
}

/// A chat reply written by the Network, when it is connected and has `id` open; `None` when Studio
/// should write it itself. Tokens arrive as "chat-token" events for `chat`.
pub(crate) fn chat(
    studio: &Studio,
    id: &str,
    chat: &str,
    prompt: &str,
    max_tokens: usize,
    sampling: &Sampling,
) -> Option<Result<(Reply, String), String>> {
    let c = client(studio)?;
    if studio.network.registered.lock().unwrap().as_deref() != Some(id) {
        return None;
    }
    *studio.network.writing.lock().unwrap() = Some(chat.to_string());
    let answer = c.call(
        "model.chat",
        serde_json::json!({"id": id, "prompt": prompt, "maxTokens": max_tokens, "sampling": sampling, "stream": chat}),
    );
    *studio.network.writing.lock().unwrap() = None;
    let by = format!("Yendory Network {}", c.hello.network);
    Some(match answer {
        Ok(r) => Ok((
            Reply {
                text: r["text"].as_str().unwrap_or_default().to_string(),
                tokens: r["tokens"].as_u64().unwrap_or(0) as usize,
                finished: r["finished"].as_bool().unwrap_or(false),
                stopped: r["stopped"].as_bool().unwrap_or(false),
            },
            by,
        )),
        Err(e) => Err(e.message),
    })
}

/// Stop: the reply the Network is writing stops after its current token.
pub(crate) fn stop(studio: &Studio) {
    let Some(stream) = studio.network.writing.lock().unwrap().clone() else { return };
    if let Some(c) = client(studio) {
        std::thread::spawn(move || {
            let _ = c.call("model.stop", serde_json::json!({"stream": stream}));
        });
    }
}

#[tauri::command]
pub(crate) fn network_status(studio: State<Studio>) -> NetStatus {
    let mut s = studio.network.status.lock().unwrap().clone();
    if s.connected && client(&studio).is_none() {
        s = NetStatus {
            problem: Some("The connection to the Yendory Network was lost.".into()),
            ..Default::default()
        };
    }
    s
}

/// Connects again (Global Settings ▸ Yendory Network).
#[tauri::command]
pub(crate) fn network_connect(app: AppHandle) {
    connect(&app);
}
