use keyring::Entry;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::State;
use chrono::Utc;

const SERVICE_NAME: &str = "camouflage";
const SOCKET_PORT: u16 = 54465;

// Camera process state
pub struct CameraState {
    process: Mutex<Option<Child>>,
    socket: Arc<Mutex<Option<TcpStream>>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Widget {
    pub id: String,
    pub widget_type: String,
    pub label: String,
    pub icon: String,
    pub position: u8,
    pub enabled: bool,
    pub config: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PieMenuConfig {
    pub widgets: Vec<Widget>,
    pub activation_gesture: String,
    pub activation_delay_ms: u32,
    pub selection_delay_ms: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConnectedAccount {
    pub provider: String,
    pub email: Option<String>,
    pub connected: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub pie_menu: PieMenuConfig,
    pub connected_accounts: Vec<ConnectedAccount>,
    pub general: GeneralConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeneralConfig {
    pub use_depthai_hardware: bool,
    pub virtual_camera_enabled: bool,
    pub show_nerd_stats: bool,
    pub mirror_display: bool,
    #[serde(default = "default_mirror_virtual_ui")]
    pub mirror_virtual_ui: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LocationData {
    pub latitude: f64,
    pub longitude: f64,
    pub accuracy: Option<f64>,
    pub timestamp: i64, // Unix timestamp
    pub city: Option<String>,
}

fn default_mirror_virtual_ui() -> bool {
    true
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            pie_menu: PieMenuConfig {
                widgets: vec![
                    Widget {
                        id: "draw".to_string(),
                        widget_type: "toggle".to_string(),
                        label: "Draw".to_string(),
                        icon: "pencil".to_string(),
                        position: 0,
                        enabled: true,
                        config: serde_json::json!({}),
                    },
                    Widget {
                        id: "nerd_stats".to_string(),
                        widget_type: "toggle".to_string(),
                        label: "Nerd Stats".to_string(),
                        icon: "eye".to_string(),
                        position: 1,
                        enabled: true,
                        config: serde_json::json!({}),
                    },
                ],
                activation_gesture: "fist".to_string(),
                activation_delay_ms: 500,
                selection_delay_ms: 500,
            },
            connected_accounts: vec![],
            general: GeneralConfig {
                use_depthai_hardware: true,
                virtual_camera_enabled: true,
                show_nerd_stats: true,
                mirror_display: true,
                mirror_virtual_ui: true,
            },
        }
    }
}

fn get_config_path() -> PathBuf {
    let home = dirs::home_dir().expect("Could not find home directory");
    home.join(".camouflage").join("config.json")
}

#[tauri::command]
fn load_config() -> Result<AppConfig, String> {
    let config_path = get_config_path();

    if config_path.exists() {
        let content = fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse config: {}", e))
    } else {
        Ok(AppConfig::default())
    }
}

#[tauri::command]
fn save_config(config: AppConfig) -> Result<(), String> {
    let config_path = get_config_path();

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
    }

    let content = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;

    fs::write(&config_path, content)
        .map_err(|e| format!("Failed to write config: {}", e))
}

#[tauri::command]
fn store_token(provider: &str, token: &str) -> Result<(), String> {
    let entry = Entry::new(SERVICE_NAME, provider)
        .map_err(|e| format!("Failed to create keyring entry: {}", e))?;

    entry.set_password(token)
        .map_err(|e| format!("Failed to store token: {}", e))
}

#[tauri::command]
fn get_token(provider: &str) -> Result<Option<String>, String> {
    let entry = Entry::new(SERVICE_NAME, provider)
        .map_err(|e| format!("Failed to create keyring entry: {}", e))?;

    match entry.get_password() {
        Ok(token) => Ok(Some(token)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(format!("Failed to get token: {}", e)),
    }
}

#[tauri::command]
fn delete_token(provider: &str) -> Result<(), String> {
    let entry = Entry::new(SERVICE_NAME, provider)
        .map_err(|e| format!("Failed to create keyring entry: {}", e))?;

    match entry.delete_credential() {
        Ok(()) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(format!("Failed to delete token: {}", e)),
    }
}

#[tauri::command]
fn get_available_widgets() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "id": "draw",
            "type": "toggle",
            "label": "Draw Mode",
            "icon": "pencil",
            "description": "Toggle drawing mode with pinch gesture",
            "requires_auth": false
        }),
        serde_json::json!({
            "id": "nerd_stats",
            "type": "toggle",
            "label": "Nerd Stats",
            "icon": "eye",
            "description": "Toggle FPS, hand landmarks, and debug info overlay",
            "requires_auth": false
        }),
        serde_json::json!({
            "id": "calendar",
            "type": "display",
            "label": "Calendar",
            "icon": "calendar",
            "description": "Show upcoming events from Google Calendar",
            "requires_auth": true,
            "auth_provider": "google"
        }),
        serde_json::json!({
            "id": "weather",
            "type": "display",
            "label": "Weather",
            "icon": "cloud",
            "description": "Show current weather conditions",
            "requires_auth": false
        }),
        serde_json::json!({
            "id": "notifications",
            "type": "display",
            "label": "Notifications",
            "icon": "bell",
            "description": "Show recent notifications",
            "requires_auth": false
        }),
        serde_json::json!({
            "id": "timer",
            "type": "action",
            "label": "Timer",
            "icon": "clock",
            "description": "Start/stop a countdown timer",
            "requires_auth": false
        }),
        serde_json::json!({
            "id": "avoid_gestures",
            "type": "toggle",
            "label": "Pause Gestures",
            "icon": "pause",
            "description": "Pause all gestures until dual PEACE signs are shown",
            "requires_auth": false
        }),
    ]
}

// Camera control arguments
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CameraArgs {
    pub use_depthai: bool,
    pub virtual_cam: bool,
    pub hide_extras: bool,
    pub draw_mode: bool,
}

fn get_project_root() -> PathBuf {
    // Navigate from config-app to the parent camouflage directory
    // Current dir during dev is config-app/src-tauri, so we need to go up two levels
    let config_dir = std::env::current_dir().expect("Could not get current directory");

    // Check if we're in config-app or config-app/src-tauri
    if config_dir.ends_with("src-tauri") {
        // We're in src-tauri, go up two levels
        config_dir.parent()
            .and_then(|p| p.parent())
            .unwrap_or(&config_dir)
            .to_path_buf()
    } else if config_dir.ends_with("config-app") {
        // We're in config-app, go up one level
        config_dir.parent().unwrap_or(&config_dir).to_path_buf()
    } else {
        // Check if demo.py exists at current location
        if config_dir.join("demo.py").exists() {
            config_dir
        } else if config_dir.parent().map(|p| p.join("demo.py").exists()).unwrap_or(false) {
            config_dir.parent().unwrap().to_path_buf()
        } else {
            // Default to parent
            config_dir.parent().unwrap_or(&config_dir).to_path_buf()
        }
    }
}

fn find_python(project_root: &PathBuf) -> String {
    // First try the project's venv
    let venv_python = project_root.join("venv").join("bin").join("python");
    if venv_python.exists() {
        return venv_python.to_string_lossy().to_string();
    }

    // Fall back to system python
    let candidates = vec![
        "python3.11",
        "python3",
        "python",
    ];

    for candidate in candidates {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return candidate.to_string();
        }
    }

    "python3".to_string()
}

#[tauri::command]
fn start_camera(
    args: CameraArgs,
    state: State<CameraState>,
) -> Result<(), String> {
    let mut process_lock = state.process.lock().map_err(|e| e.to_string())?;

    // Check if already running
    if process_lock.is_some() {
        return Err("Camera is already running".to_string());
    }

    let project_root = get_project_root();
    let python = find_python(&project_root);
    let model_path = project_root.join("models").join("hand_landmark_full-2022-11-10_sh6.blob");

    let cwd = std::env::current_dir().unwrap_or_default();
    log::info!("Current working directory: {:?}", cwd);
    log::info!("Starting camera with python: {}", python);
    log::info!("Project root: {:?}", project_root);
    log::info!("Model path: {:?}", model_path);
    log::info!("Demo.py exists: {}", project_root.join("demo.py").exists());
    log::info!("Venv python exists: {}", project_root.join("venv/bin/python").exists());

    let mut command = Command::new(&python);
    command
        .current_dir(&project_root)
        .arg("demo.py")
        .arg("--gesture")
        .arg("--lm_model")
        .arg(model_path.to_string_lossy().to_string())
        .arg("--messages")
        .arg("-f")
        .arg("15")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    // Configure based on hardware selection
    if args.use_depthai {
        command.arg("--edge");
    } else {
        command.arg("-i").arg("0");
    }

    if args.virtual_cam {
        command.arg("--virtual_cam");
    }

    if args.hide_extras {
        command.arg("--hide");
    }

    if args.draw_mode {
        command.arg("--draw");
    }

    // Set environment for 3D rendering on macOS
    command.env("PYOPENGL_PLATFORM", "egl");

    let child = command
        .spawn()
        .map_err(|e| format!("Failed to start camera: {}", e))?;

    *process_lock = Some(child);

    // Try to connect to socket after a delay
    drop(process_lock);

    // Spawn a thread to connect to the socket after demo.py starts
    let socket_arc = state.socket.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(5));

        for attempt in 0..3 {
            match TcpStream::connect(format!("127.0.0.1:{}", SOCKET_PORT)) {
                Ok(stream) => {
                    if let Ok(mut socket_lock) = socket_arc.lock() {
                        *socket_lock = Some(stream);
                        log::info!("Connected to demo.py socket");
                    }
                    return;
                }
                Err(e) => {
                    log::warn!("Socket connection attempt {} failed: {}", attempt + 1, e);
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
        log::error!("Failed to connect to demo.py socket after 3 attempts");
    });

    Ok(())
}

#[tauri::command]
fn stop_camera(state: State<CameraState>) -> Result<(), String> {
    // Close socket first
    if let Ok(mut socket_lock) = state.socket.lock() {
        if let Some(socket) = socket_lock.take() {
            drop(socket);
        }
    }

    // Then stop the process
    let mut process_lock = state.process.lock().map_err(|e| e.to_string())?;

    if let Some(mut child) = process_lock.take() {
        // Try to terminate gracefully first
        let _ = child.kill();
        let _ = child.wait();
    }

    Ok(())
}

#[tauri::command]
fn force_stop_camera(state: State<CameraState>) -> Result<String, String> {
    // First try normal stop
    if let Ok(mut socket_lock) = state.socket.lock() {
        if let Some(socket) = socket_lock.take() {
            drop(socket);
        }
    }

    if let Ok(mut process_lock) = state.process.lock() {
        if let Some(mut child) = process_lock.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    // Then force kill any demo.py processes using pkill
    let output = Command::new("pkill")
        .arg("-f")
        .arg("demo.py")
        .output();

    match output {
        Ok(result) => {
            if result.status.success() {
                Ok("Killed demo.py processes".to_string())
            } else {
                Ok("No demo.py processes found".to_string())
            }
        }
        Err(e) => Err(format!("Failed to run pkill: {}", e))
    }
}

#[tauri::command]
fn get_camera_status(state: State<CameraState>) -> Result<bool, String> {
    let mut process_lock = state.process.lock().map_err(|e| e.to_string())?;

    if let Some(ref mut child) = *process_lock {
        // Check if process is still running
        match child.try_wait() {
            Ok(Some(_)) => {
                // Process has exited
                *process_lock = None;
                Ok(false)
            }
            Ok(None) => {
                // Process is still running
                Ok(true)
            }
            Err(_) => {
                *process_lock = None;
                Ok(false)
            }
        }
    } else {
        Ok(false)
    }
}

#[tauri::command]
fn send_camera_command(command: String, state: State<CameraState>) -> Result<(), String> {
    let mut socket_lock = state.socket.lock().map_err(|e| e.to_string())?;

    if let Some(ref mut stream) = *socket_lock {
        stream
            .write_all(command.as_bytes())
            .map_err(|e| format!("Failed to send command: {}", e))?;
        Ok(())
    } else {
        Err("Not connected to camera".to_string())
    }
}

fn get_location_path() -> PathBuf {
    let home = dirs::home_dir().expect("Could not find home directory");
    home.join(".camouflage").join("location.json")
}

#[tauri::command]
fn save_location(location: LocationData) -> Result<(), String> {
    let location_path = get_location_path();

    if let Some(parent) = location_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
    }

    let content = serde_json::to_string_pretty(&location)
        .map_err(|e| format!("Failed to serialize location: {}", e))?;

    fs::write(&location_path, content)
        .map_err(|e| format!("Failed to write location: {}", e))
}

#[tauri::command]
fn get_saved_location() -> Result<Option<LocationData>, String> {
    let location_path = get_location_path();

    if location_path.exists() {
        let content = fs::read_to_string(&location_path)
            .map_err(|e| format!("Failed to read location: {}", e))?;
        let location: LocationData = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse location: {}", e))?;
        Ok(Some(location))
    } else {
        Ok(None)
    }
}

#[cfg(target_os = "macos")]
#[tauri::command]
async fn request_location() -> Result<LocationData, String> {
    use objc2_core_location::CLLocationManager;
    use objc2_foundation::MainThreadMarker;
    use std::thread;
    use std::time::Duration;

    // CoreLocation needs to run on the main thread
    // We'll use a simple polling approach
    let (tx, rx) = std::sync::mpsc::channel();

    thread::spawn(move || {
        // This is a simplified approach - in production you'd want proper delegate handling
        // For now, we'll try to get the last known location
        unsafe {
            if let Some(_mtm) = MainThreadMarker::new() {
                let manager = CLLocationManager::new();

                // Request authorization
                manager.requestWhenInUseAuthorization();

                // Wait a bit for authorization
                thread::sleep(Duration::from_secs(1));

                // Try to get location
                manager.startUpdatingLocation();
                thread::sleep(Duration::from_secs(3));
                manager.stopUpdatingLocation();

                if let Some(location) = manager.location() {
                    let coord = location.coordinate();
                    let accuracy = location.horizontalAccuracy();

                    let _ = tx.send(Ok(LocationData {
                        latitude: coord.latitude,
                        longitude: coord.longitude,
                        accuracy: Some(accuracy),
                        timestamp: Utc::now().timestamp(),
                        city: None, // Will be filled by reverse geocoding
                    }));
                    return;
                }
            }
        }
        let _ = tx.send(Err("Could not get location".to_string()));
    });

    rx.recv_timeout(Duration::from_secs(10))
        .map_err(|_| "Location request timed out".to_string())?
}

#[cfg(not(target_os = "macos"))]
#[tauri::command]
async fn request_location() -> Result<LocationData, String> {
    Err("Location services not available on this platform".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_http::init())
        .manage(CameraState {
            process: Mutex::new(None),
            socket: Arc::new(Mutex::new(None)),
        })
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_config,
            save_config,
            store_token,
            get_token,
            delete_token,
            get_available_widgets,
            start_camera,
            stop_camera,
            force_stop_camera,
            get_camera_status,
            send_camera_command,
            save_location,
            get_saved_location,
            request_location,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
