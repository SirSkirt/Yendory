//! The Network as programs see it: the real binary, started in a temporary data folder, driven
//! through `yendory-client`.
use serde_json::json;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use yendory_client::{paths, Client, ServiceInfo};

struct Network {
    child: Child,
    data: PathBuf,
}

impl Network {
    fn start(name: &str, idle: u64) -> Network {
        let data = std::env::temp_dir().join(format!("yendory-service-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data);
        std::fs::create_dir_all(&data).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_yendory-network"))
            .env("YENDORY_DATA_DIR", &data)
            .env("YENDORY_NETWORK_IDLE", idle.to_string())
            .spawn()
            .unwrap();
        let n = Network { child, data };
        n.wait_for(|| n.info().is_some(), "service.json");
        n
    }
    fn info(&self) -> Option<ServiceInfo> {
        let path = self.data.join(paths::NETWORK_ID).join(paths::SERVICE_FILE);
        serde_json::from_slice(&std::fs::read(path).ok()?).ok()
    }
    fn wait_for(&self, ok: impl Fn() -> bool, what: &str) {
        let until = Instant::now() + Duration::from_secs(20);
        while !ok() {
            assert!(Instant::now() < until, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    /// A client of this Network (the data folder is chosen by environment, so set it here).
    fn connect(&self, name: &str) -> Result<Client, yendory_client::Error> {
        std::env::set_var("YENDORY_DATA_DIR", &self.data);
        Client::connect(name, "test")
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.data);
    }
}

// The tests share YENDORY_DATA_DIR (process environment), so they run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn programs_share_one_network_and_its_models() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let n = Network::start("share", 60);
    let hub = n.connect("Yendory Hub").unwrap();
    let studio = n.connect("Yendory Studio").unwrap();
    assert_eq!(hub.hello.api, yendory_client::API);
    assert_ne!(hub.hello.client_id, studio.hello.client_id);
    let studio_events = studio.take_events().unwrap();

    // The Hub makes a model; Studio hears about it and sees it.
    let made = hub
        .call(
            "models.create",
            json!({"name": "Robin", "architecture": "bicameral", "preset": "nova2", "context": 256}),
        )
        .unwrap();
    let id = made["meta"]["id"].as_str().unwrap().to_string();
    let e = studio_events.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!((e.name.as_str(), e.data["id"].as_str()), ("models.changed", Some(id.as_str())));
    let list = studio.call("models.list", json!({})).unwrap();
    assert_eq!(list.as_array().unwrap().len(), 1);

    // Studio opens it and asks it something; the untrained model answers something.
    studio.call("models.open", json!({"id": id})).unwrap();
    let r = studio.call("model.reply", json!({"id": id, "prompt": "U: hello\nN: ", "maxTokens": 8})).unwrap();
    assert!(r["text"].is_string());
    // The Hub can't reply without opening, nor rename or delete what Studio has open.
    assert_eq!(hub.call("model.reply", json!({"id": id, "prompt": "x"})).unwrap_err().code, "refused");
    let err = hub.call("models.rename", json!({"id": id, "name": "Wren"})).unwrap_err();
    assert!(err.message.contains("Yendory Studio has this model open"), "{}", err.message);
    assert_eq!(hub.call("models.delete", json!({"id": id})).unwrap_err().code, "refused");
    let listed = hub.call("models.list", json!({})).unwrap();
    assert_eq!(listed[0]["openBy"], json!(["Yendory Studio"]));
    // The Hub can't stop the Network under Studio.
    assert_eq!(hub.call("shutdown", json!({})).unwrap_err().code, "refused");

    // Studio leaves: its model is closed, and the Hub may change it.
    drop(studio);
    n.wait_for(
        || hub.call("models.list", json!({})).unwrap()[0]["openBy"] == json!([]),
        "Studio's model to close",
    );
    hub.call("models.rename", json!({"id": id, "name": "Wren"})).unwrap();
    hub.call("models.delete", json!({"id": id})).unwrap();
    assert_eq!(hub.call("models.list", json!({})).unwrap(), json!([]));
    assert_eq!(hub.call("no.such", json!({})).unwrap_err().code, "unknown_method");
}

#[test]
fn a_wrong_token_is_refused() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let n = Network::start("token", 60);
    let mut info = n.info().unwrap();
    info.token = "0".repeat(64);
    let path = n.data.join(paths::NETWORK_ID).join(paths::SERVICE_FILE);
    let real = std::fs::read(&path).unwrap();
    std::fs::write(&path, serde_json::to_vec(&info).unwrap()).unwrap();
    let err = n.connect("Intruder").err().expect("refused");
    assert_eq!(err.code, "unauthorized");
    std::fs::write(&path, real).unwrap();
    assert!(n.connect("Yendory Hub").is_ok());
}

#[test]
fn one_copy_runs_and_it_stops_when_idle() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut n = Network::start("idle", 2);
    let first = n.info().unwrap();
    // A second copy sees the first and leaves at once, without touching service.json.
    let status = Command::new(env!("CARGO_BIN_EXE_yendory-network"))
        .env("YENDORY_DATA_DIR", &n.data)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(n.info().unwrap(), first);
    // A connected program keeps it running past the idle time.
    let hub = n.connect("Yendory Hub").unwrap();
    std::thread::sleep(Duration::from_secs(3));
    assert_eq!(hub.call("ping", json!({})).unwrap(), json!("pong"));
    // Alone and idle, it stops and removes service.json.
    drop(hub);
    n.wait_for(|| n.info().is_none(), "the idle Network to stop");
    let code = n.child.wait().unwrap();
    assert!(code.success());
}

#[test]
fn shutdown_when_asked_and_alone() {
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut n = Network::start("shutdown", 60);
    let hub = n.connect("Yendory Hub").unwrap();
    let events = hub.take_events().unwrap();
    assert_eq!(hub.call("shutdown", json!({})).unwrap(), json!(true));
    let e = events.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(e.name, "network.stopping");
    assert!(n.child.wait().unwrap().success());
    assert!(n.info().is_none());
}

#[test]
fn version_is_reported_without_starting() {
    let out = Command::new(env!("CARGO_BIN_EXE_yendory-network")).arg("--version").output().unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["network"], env!("CARGO_PKG_VERSION"));
    assert_eq!(v["api"], yendory_client::API);
}

/// Studio's chat through the Network: a sampled reply streamed token by token (the same reply
/// the Engine writes on its own with those settings), stopped on request, and new weights read
/// after Studio wrote the model file itself.
#[test]
fn chat_streams_stops_and_reloads() {
    use yendory_engine::learn::sampling::{reply_with, Sampling};
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let n = Network::start("chat", 60);
    let studio = n.connect("Yendory Studio").unwrap();
    let events = studio.take_events().unwrap();
    let make = |name: &str| {
        studio
            .call(
                "models.create",
                json!({"name": name, "architecture": "bicameral", "preset": "nova2", "context": 256}),
            )
            .unwrap()["meta"]["id"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let a = make("Robin");
    let b = make("Wren");
    // models.changed events of the two creations.
    while events.recv_timeout(Duration::from_millis(300)).is_ok() {}
    let models = n.data.join(paths::STUDIO_ID).join("models");
    let prompt = "U: hello, how are you?\nN: ";
    let sampling = Sampling { temperature: 0.9, top_k: 40, top_p: 0.95, repetition_penalty: 1.2, seed: 7 };

    // A chat needs the model open.
    let ask = json!({"id": a, "prompt": prompt, "maxTokens": 16, "sampling": sampling, "stream": "chat-1"});
    assert_eq!(studio.call("model.chat", ask.clone()).unwrap_err().code, "refused");
    studio.call("models.open", json!({"id": a})).unwrap();
    let r = studio.call("model.chat", ask).unwrap();
    let mut streamed = String::new();
    while let Ok(e) = events.recv_timeout(Duration::from_millis(300)) {
        assert_eq!((e.name.as_str(), e.data["stream"].as_str()), ("model.token", Some("chat-1")));
        streamed.push_str(e.data["text"].as_str().unwrap());
    }
    let local = yendory_engine::Bicameral::load(models.join(&a).join("model.nnova2")).unwrap();
    let mut expected = String::new();
    let reply = reply_with(&local, prompt, 16, &sampling, |t| {
        expected.push_str(t);
        true
    })
    .unwrap();
    assert_eq!(r["text"], json!(reply.text));
    assert_eq!(r["tokens"], json!(reply.tokens));
    assert_eq!(r["finished"], json!(reply.finished));
    assert_eq!(streamed, expected);

    // Stopping: false for no reply, and a long reply stops soon after it is asked to.
    assert_eq!(studio.call("model.stop", json!({"stream": "none"})).unwrap(), json!(false));
    let studio = std::sync::Arc::new(studio);
    let long = {
        let studio = studio.clone();
        let a = a.clone();
        std::thread::spawn(move || {
            studio.call(
                "model.chat",
                json!({"id": a, "prompt": prompt, "maxTokens": 4096, "stream": "chat-2",
                       "sampling": {"temperature": 1.5, "seed": 3}}),
            )
        })
    };
    let first = events.recv_timeout(Duration::from_secs(20)).unwrap();
    assert_eq!(first.data["stream"], json!("chat-2"));
    let found = studio.call("model.stop", json!({"stream": "chat-2"})).unwrap();
    let r = long.join().unwrap().unwrap();
    if found == json!(true) {
        assert_eq!(r["stopped"], json!(true), "{r}");
        assert!(r["tokens"].as_u64().unwrap() < 4096);
    } else {
        // It had already ended with a line break.
        assert_eq!(r["finished"], json!(true), "{r}");
    }
    while events.recv_timeout(Duration::from_millis(300)).is_ok() {}

    // Studio writes the model file itself (training), then asks the Network to read it again.
    std::fs::copy(models.join(&b).join("model.nnova2"), models.join(&a).join("model.nnova2")).unwrap();
    let greedy = json!({"id": a, "prompt": prompt, "maxTokens": 12});
    studio.call("models.reload", json!({"id": a})).unwrap();
    let after = studio.call("model.chat", greedy).unwrap();
    let wren = yendory_engine::Bicameral::load(models.join(&b).join("model.nnova2")).unwrap();
    let want = reply_with(&wren, prompt, 12, &Sampling::default(), |_| true).unwrap();
    assert_eq!(after["text"], json!(want.text));
    // Only a program that has the model open may reload it.
    assert_eq!(studio.call("models.reload", json!({"id": b})).unwrap_err().code, "refused");
}

#[test]
fn game_skills_are_saved_into_models_in_one_message_or_in_parts() {
    use yendory_client::floats::{decode, encode};
    use yendory_engine::skill::{check_arch, Lineage, Size, Skill, SkillSpec};
    let _one = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let n = Network::start("skills", 60);
    let hub = n.connect("Yendory Hub").unwrap();
    let tools = n.connect("Yendory Tools").unwrap();
    let make = |arch: &str, preset: &str| {
        let m = hub
            .call(
                "models.create",
                json!({"name": "Nova", "architecture": arch, "preset": preset, "context": 256}),
            )
            .unwrap();
        m["meta"]["id"].as_str().unwrap().to_string()
    };
    let neuro = make("neuro", "neuro-core-1m");
    let bica = make("bicameral", "nova2");
    tools.call("models.open", json!({"id": neuro})).unwrap();
    assert_eq!(tools.call("model.skills", json!({"id": neuro})).unwrap(), json!([]));

    let spec = SkillSpec {
        game: "yendory-range".into(),
        name: "Yendory Range".into(),
        size: Size::Small,
        seed: 3,
        arch: check_arch(),
        keys: vec!["w".into(), "a".into()],
        buttons: vec!["left".into()],
        lineage: Lineage { generation: 0, parent: None, method: "imitation".into() },
        steps: 40,
        accepted_step: 30,
        created: 1,
        notes: json!({"runs": 1}),
    };
    let mut skill = Skill::new(spec.clone()).unwrap();
    skill.latest[2] = 0.75;
    skill.adam.t = 40;
    let save = |client: &Client, id: &str, k: &Skill| {
        client.call(
            "model.skill.save",
            json!({"id": id, "spec": k.spec, "accepted": encode(&k.accepted), "latest": encode(&k.latest),
                   "m": encode(&k.adam.m), "v": encode(&k.adam.v), "t": k.adam.t}),
        )
    };
    let entry = save(&tools, &neuro, &skill).unwrap();
    assert_eq!(entry["meta"]["skills"][0]["game"], "yendory-range");
    assert_eq!(entry["meta"]["skills"][0]["steps"], 40);
    assert_eq!(entry["meta"]["name"], "Nova", "other fields kept");

    // Read back exactly, from memory and from the file.
    let back = tools.call("model.skill", json!({"id": neuro, "game": "yendory-range"})).unwrap();
    assert_eq!(decode(back["latest"].as_str().unwrap()).unwrap(), skill.latest);
    assert_eq!(back["t"], 40);
    let file = yendory_engine::Bicameral::load(
        n.data.join("org.yendory.studio/models").join(&neuro).join("model.nnova2"),
    )
    .unwrap();
    assert_eq!(file.skills[0].spec, spec);
    assert_eq!(file.skills[0].accepted, skill.accepted);

    // Not while another program has the model open (its next save would undo it).
    hub.call("models.open", json!({"id": neuro})).unwrap();
    let e = save(&tools, &neuro, &skill).unwrap_err();
    assert_eq!(e.code, "refused");
    hub.call("models.close", json!({"id": neuro})).unwrap();

    // Into a Bicameral model too (Network 0.1.3); not with weights that don't fit.
    tools.call("models.open", json!({"id": bica})).unwrap();
    save(&tools, &bica, &skill).unwrap();
    let file = yendory_engine::Bicameral::load(
        n.data.join("org.yendory.studio/models").join(&bica).join("model.nnova2"),
    )
    .unwrap();
    assert!(file.neuro.is_none());
    assert_eq!(file.skills[0].latest, skill.latest);
    let mut short = skill.clone();
    short.accepted.pop();
    assert_eq!(save(&tools, &neuro, &short).unwrap_err().code, "invalid");

    // Removed again: the file is the plain format again.
    tools.call("model.skill.delete", json!({"id": neuro, "game": "yendory-range"})).unwrap();
    let file = yendory_engine::Bicameral::load(
        n.data.join("org.yendory.studio/models").join(&neuro).join("model.nnova2"),
    )
    .unwrap();
    assert!(file.skills.is_empty());
    let listed = hub.call("models.list", json!({})).unwrap();
    let meta = listed.as_array().unwrap().iter().find(|m| m["meta"]["id"] == neuro.as_str()).unwrap();
    assert_eq!(meta["meta"]["skills"], json!([]));

    // A skill too large for one message (over 64 MB of base64) goes in parts, both ways.
    let mut big_spec = spec.clone();
    big_spec.game = "ravenfield".into();
    big_spec.arch = yendory_engine::skill::Arch::new(384, 216, 3, 2, 1, vec![-4.0, 0.0, 4.0], Size::Small);
    let mut big = Skill::new(big_spec).unwrap();
    let n_weights = big.accepted.len();
    assert!(n_weights * 16 * 4 / 3 > 64 << 20, "{n_weights} weights fit one message");
    big.latest[n_weights - 1] = 2.5;
    big.adam.v[7] = 0.125;
    big.adam.t = 9;
    let mut data = yendory_client::floats::to_bytes(&big.accepted);
    for a in [&big.latest, &big.adam.m, &big.adam.v] {
        data.extend_from_slice(&yendory_client::floats::to_bytes(a));
    }
    let up = tools.upload(&data).unwrap();
    tools
        .call("model.skill.save", json!({"id": bica, "spec": big.spec, "upload": up, "t": big.adam.t}))
        .unwrap();
    // The upload is used up.
    assert!(tools
        .call("model.skill.save", json!({"id": bica, "spec": big.spec, "upload": up, "t": 1}))
        .is_err());
    let head = tools.call("model.skill", json!({"id": bica, "game": "ravenfield", "parts": true})).unwrap();
    assert_eq!(head["bytes"].as_u64().unwrap() as usize, data.len());
    assert_eq!(head["t"], 9);
    let back =
        tools.read_parts("model.skill.read", json!({"id": bica, "game": "ravenfield"}), data.len()).unwrap();
    assert!(back == data, "the skill read in parts differs");
    // An upload that is not the skill's size is refused, and another program can't use it.
    let small = tools.upload(&data[..1000]).unwrap();
    assert!(tools
        .call("model.skill.save", json!({"id": bica, "spec": big.spec, "upload": small, "t": 1}))
        .is_err());
    let theirs = tools.upload(&data[..16]).unwrap();
    assert!(hub.call("upload.part", json!({"upload": theirs, "data": "AAAA"})).is_err());

    // The whole model, downloaded, trained by the program and sent back (Network 0.1.4).
    let bytes = tools.download("model.checkpoint", json!({"id": bica})).unwrap();
    let mut model = yendory_engine::Bicameral::from_bytes(&bytes).unwrap();
    assert_eq!(model.skills.len(), 2, "the skills come with it");
    model.weights[5] += 0.5;
    model.step = 77;
    let up = tools.upload(&model.to_bytes().unwrap()).unwrap();
    let entry = tools.call("model.checkpoint.save", json!({"id": bica, "upload": up})).unwrap();
    assert_eq!(entry["meta"]["step"], 77);
    let file = yendory_engine::Bicameral::load(
        n.data.join("org.yendory.studio/models").join(&bica).join("model.nnova2"),
    )
    .unwrap();
    assert_eq!(file.weights[5], model.weights[5]);
    assert_eq!(file.skills.len(), 2);
    // Not another model's shape; not while another program has it open.
    let other = yendory_engine::Bicameral::with_config(
        yendory_engine::model::Config { d: 64, heads: 2, ff: 128, ..Default::default() },
        1,
    );
    let up = tools.upload(&other.to_bytes().unwrap()).unwrap();
    assert!(tools.call("model.checkpoint.save", json!({"id": bica, "upload": up})).is_err());
    hub.call("models.open", json!({"id": bica})).unwrap();
    let up = tools.upload(&model.to_bytes().unwrap()).unwrap();
    assert_eq!(
        tools.call("model.checkpoint.save", json!({"id": bica, "upload": up})).unwrap_err().code,
        "refused"
    );
}
