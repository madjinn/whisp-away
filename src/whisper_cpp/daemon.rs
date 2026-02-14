use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::net::Shutdown;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use tracing::{error, info, warn};
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
#[cfg(feature = "openvino")]
use whisper_rs::WhisperState;
use crate::helpers::wav_to_samples;

const SOCKET_PATH: &str = "/tmp/whisp-away-daemon.sock";

#[tokio::main]
pub async fn run_daemon(model_path: &str) -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    // Create and run daemon
    let daemon = WhisperDaemon::new(model_path)?;
    daemon.run().await
}

#[derive(Debug, Serialize, Deserialize)]
struct TranscriptionRequest {
    audio_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TranscriptionResponse {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub struct WhisperDaemon {
    ctx: Arc<WhisperContext>,
    socket_path: String,
    // Single reusable state with OpenVINO initialized
    #[cfg(feature = "openvino")]
    state: Arc<tokio::sync::Mutex<WhisperState>>,
}

impl WhisperDaemon {
    pub fn new(model_path: &str) -> Result<Self> {
        // If model_path doesn't contain a path separator, treat it as a model name
        // and construct the full path
        let final_model_path = if !model_path.contains('/') {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/home/martin".to_string());
            let model_extension = if model_path.ends_with(".bin") { "" } else { ".bin" };
            format!("{}/.cache/whisper-cpp/models/ggml-{}{}", home, model_path, model_extension)
        } else {
            model_path.to_string()
        };
        
        info!("Loading whisper.cpp model from: {}", final_model_path);
        
        // Check if model file exists
        if !Path::new(&final_model_path).exists() {
            return Err(anyhow::anyhow!("Model file not found: {}", final_model_path));
        }
        
        // Create whisper context with GPU configuration
        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(true);  // Enable GPU acceleration
        ctx_params.gpu_device(0);   // Use GPU device 0
        
        // Don't configure OpenVINO at context level - we'll do it at state level
        // This avoids the systemd initialization issue
        
        info!("Initializing WhisperContext with configured acceleration");
        let t_ctx = std::time::Instant::now();
        let ctx = WhisperContext::new_with_params(&final_model_path, ctx_params)
            .context("Failed to create WhisperContext")?;
        eprintln!("DEBUG DAEMON: Context creation took {:?}", t_ctx.elapsed());
        
        info!("Model loaded successfully into memory");
        
        // Create a single state with OpenVINO initialized
        #[cfg(feature = "openvino")]
        let state = {
            eprintln!("DEBUG DAEMON: Creating reusable state with OpenVINO...");
            let t_state = std::time::Instant::now();
            let mut state = ctx.create_state()
                .context("Failed to create whisper state")?;
            eprintln!("DEBUG DAEMON: State creation took {:?}", t_state.elapsed());
            
            // Initialize OpenVINO at state level
            let model_base = final_model_path.trim_end_matches(".bin");
            let openvino_model = format!("{}-encoder-openvino.xml", model_base);
            if std::path::Path::new(&openvino_model).exists() {
                let t_ov = std::time::Instant::now();
                eprintln!("DEBUG DAEMON: Initializing OpenVINO at state level...");
                // Use RAM-based cache in /dev/shm for faster access
                // Extract model name from path (e.g., "base.en" from "/path/to/ggml-base.en.bin")
                // Set cache directory as subdirectory next to the model files
                let cache_dir = format!("{}-encoder-openvino-cache", model_base);
                // Ensure cache directory exists
                if let Err(e) = std::fs::create_dir_all(&cache_dir) {
                    eprintln!("DEBUG DAEMON: Warning: Could not create cache dir: {:?}", e);
                }
                eprintln!("DEBUG DAEMON: Using cache dir: {}", cache_dir);
                // Use AUTO to let OpenVINO choose the best device
                match state.init_openvino_encoder_state_level(None, "AUTO", Some(&cache_dir)) {
                    Ok(_) => eprintln!("DEBUG DAEMON: OpenVINO initialized with AUTO device selection in {:?}", t_ov.elapsed()),
                    Err(e) => {
                        eprintln!("DEBUG DAEMON: Failed to init OpenVINO: {:?}", e);
                        eprintln!("DEBUG DAEMON: Will use regular CPU inference");
                    }
                }
            }
            Arc::new(tokio::sync::Mutex::new(state))
        };
        
        Ok(Self {
            ctx: Arc::new(ctx),
            socket_path: SOCKET_PATH.to_string(),
            #[cfg(feature = "openvino")]
            state,
        })
    }
    
    pub async fn run(&self) -> Result<()> {
        // Remove existing socket if it exists
        if Path::new(&self.socket_path).exists() {
            fs::remove_file(&self.socket_path)?;
        }
        
        // Create Unix socket listener
        let listener = UnixListener::bind(&self.socket_path)
            .context("Failed to bind Unix socket")?;
        
        // Set socket permissions
        let mut perms = fs::metadata(&self.socket_path)?.permissions();
        perms.set_mode(0o666);
        fs::set_permissions(&self.socket_path, perms)?;
        
        info!("Daemon listening on {}", self.socket_path);
        
        // Accept connections in a loop
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    #[cfg(feature = "openvino")]
                    {
                        let state = Arc::clone(&self.state);
                        // Spawn a task to handle the connection
                        tokio::spawn(async move {
                            let result = handle_connection_with_state(stream, state).await;
                            
                            if let Err(e) = result {
                                error!("Error handling connection: {}", e);
                            }
                        });
                    }
                    #[cfg(not(feature = "openvino"))]
                    {
                        let ctx = Arc::clone(&self.ctx);
                        // Spawn a task to handle the connection
                        tokio::spawn(async move {
                            let result = handle_connection(stream, ctx).await;
                            
                            if let Err(e) = result {
                                error!("Error handling connection: {}", e);
                            }
                        });
                    }
                }
                Err(e) => {
                    error!("Error accepting connection: {}", e);
                }
            }
        }
        
        Ok(())
    }
}

async fn handle_connection(
    mut stream: UnixStream,
    ctx: Arc<WhisperContext>,
) -> Result<()> {
    // Read request
    let mut buffer = vec![0; 4096];
    let n = stream.read(&mut buffer)?;
    let request_str = String::from_utf8_lossy(&buffer[..n]);
    
    // Parse request
    let request: TranscriptionRequest = serde_json::from_str(&request_str)
        .context("Failed to parse request")?;
    
    info!("Processing audio file: {}", request.audio_path);
    
    // Check if file exists
    if !Path::new(&request.audio_path).exists() {
        let response = TranscriptionResponse {
            success: false,
            text: None,
            error: Some(format!("Audio file not found: {}", request.audio_path)),
        };
        let response_json = serde_json::to_string(&response)?;
        stream.write_all(response_json.as_bytes())?;
        stream.flush()?;
        stream.shutdown(Shutdown::Both)?;
        return Ok(());
    }
    
    // Check file size (WAV header is 44 bytes)
    let metadata = fs::metadata(&request.audio_path)?;
    if metadata.len() <= 44 {
        warn!("Audio file is empty (only header): {}", request.audio_path);
        let response = TranscriptionResponse {
            success: true,
            text: Some(String::new()),
            error: None,
        };
        let response_json = serde_json::to_string(&response)?;
        stream.write_all(response_json.as_bytes())?;
        stream.flush()?;
        stream.shutdown(Shutdown::Both)?;
        return Ok(());
    }
    
    // Transcribe using a fresh state for each request
    let text = transcribe_audio(&request.audio_path, ctx)?;
    
    // Send response
    let response = TranscriptionResponse {
        success: true,
        text: Some(text),
        error: None,
    };
    
    let response_json = serde_json::to_string(&response)?;
    stream.write_all(response_json.as_bytes())?;
    stream.flush()?;
    stream.shutdown(Shutdown::Both)?;

    Ok(())
}

#[cfg(feature = "openvino")]
async fn handle_connection_with_state(
    mut stream: UnixStream,
    state: Arc<tokio::sync::Mutex<WhisperState>>,
) -> Result<()> {
    // Read request
    let mut buffer = vec![0; 4096];
    let n = stream.read(&mut buffer)?;
    let request_str = String::from_utf8_lossy(&buffer[..n]);
    
    // Parse request
    let request: TranscriptionRequest = serde_json::from_str(&request_str)
        .context("Failed to parse request")?;
    
    info!("Processing audio file: {}", request.audio_path);
    
    // Check if file exists
    if !Path::new(&request.audio_path).exists() {
        let response = TranscriptionResponse {
            success: false,
            text: None,
            error: Some(format!("Audio file not found: {}", request.audio_path)),
        };
        let response_json = serde_json::to_string(&response)?;
        stream.write_all(response_json.as_bytes())?;
        stream.flush()?;
        stream.shutdown(Shutdown::Both)?;
        return Ok(());
    }
    
    // Check file size (WAV header is 44 bytes)
    let metadata = fs::metadata(&request.audio_path)?;
    if metadata.len() <= 44 {
        warn!("Audio file is empty (only header): {}", request.audio_path);
        let response = TranscriptionResponse {
            success: true,
            text: Some(String::new()),
            error: None,
        };
        let response_json = serde_json::to_string(&response)?;
        stream.write_all(response_json.as_bytes())?;
        stream.flush()?;
        stream.shutdown(Shutdown::Both)?;
        return Ok(());
    }
    
    // Transcribe using the reusable state
    let text = transcribe_with_state(&request.audio_path, state).await?;
    
    // Send response
    let response = TranscriptionResponse {
        success: true,
        text: Some(text),
        error: None,
    };
    
    let response_json = serde_json::to_string(&response)?;
    stream.write_all(response_json.as_bytes())?;
    stream.flush()?;
    stream.shutdown(Shutdown::Both)?;

    Ok(())
}

#[cfg(feature = "openvino")]
async fn transcribe_with_state(
    audio_path: &str,
    state: Arc<tokio::sync::Mutex<WhisperState>>,
) -> Result<String> {
    use std::time::Instant;
    let start = Instant::now();
    
    // Load and convert audio 
    let t1 = Instant::now();
    let audio_data = std::fs::read(audio_path)
        .context("Failed to read audio file")?;
    eprintln!("DEBUG DAEMON: File read took {:?}", t1.elapsed());
    
    let t2 = Instant::now();
    let samples = wav_to_samples(&audio_data)?;
    eprintln!("DEBUG DAEMON: WAV conversion took {:?}", t2.elapsed());
    
    // Lock the state for exclusive use
    let mut state = state.lock().await;
    eprintln!("DEBUG DAEMON: Using pre-initialized state with OpenVINO");
    
    // Set up parameters - optimized for speed
    let t4 = Instant::now();
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(8);
    params.set_n_threads(num_threads);
    params.set_translate(false);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_temperature(0.0);
    params.set_single_segment(false);
    params.set_no_context(true);
    eprintln!("DEBUG DAEMON: Params setup took {:?}", t4.elapsed());
    
    // Run transcription
    let t5 = Instant::now();
    eprintln!("DEBUG DAEMON: Starting whisper transcription with {} samples...", samples.len());
    state.full(params, &samples)
        .context("Failed to transcribe audio")?;
    eprintln!("DEBUG DAEMON: Whisper transcription completed in {:?}", t5.elapsed());
    
    // Get the transcribed text from segments
    let t6 = Instant::now();
    let mut text = String::new();
    let num_segments = state.full_n_segments();
    for i in 0..num_segments {
        let segment = state.get_segment(i)
            .ok_or_else(|| anyhow!("Failed to get segment {}", i))?;
        let segment_text = segment.to_str()?;
        text.push_str(segment_text);
        text.push(' ');
    }
    eprintln!("DEBUG DAEMON: Segment extraction took {:?}", t6.elapsed());
    
    eprintln!("DEBUG DAEMON: Total transcription time: {:?}", start.elapsed());
    
    Ok(text.trim().to_string())
}

fn transcribe_audio(
    audio_path: &str,
    ctx: Arc<WhisperContext>,
) -> Result<String> {
    use std::time::Instant;
    let start = Instant::now();
    
    // Load and convert audio 
    let t1 = Instant::now();
    let audio_data = std::fs::read(audio_path)
        .context("Failed to read audio file")?;
    eprintln!("DEBUG DAEMON: File read took {:?}", t1.elapsed());
    
    let t2 = Instant::now();
    let samples = wav_to_samples(&audio_data)?;
    eprintln!("DEBUG DAEMON: WAV conversion took {:?}", t2.elapsed());
    
    // Create a fresh state for this transcription
    let t3 = Instant::now();
    let mut state = ctx.create_state()
        .context("Failed to create whisper state")?;
    eprintln!("DEBUG DAEMON: State creation took {:?}", t3.elapsed());
    eprintln!("DEBUG DAEMON: OpenVINO (if configured) was initialized automatically at context creation");
    
    // Set up parameters - optimized for speed
    let t4 = Instant::now();
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(8);
    params.set_n_threads(num_threads);
    params.set_translate(false);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_temperature(0.0);
    params.set_single_segment(false);
    params.set_no_context(true);
    eprintln!("DEBUG DAEMON: Params setup took {:?}", t4.elapsed());
    
    // Run transcription
    let t5 = Instant::now();
    eprintln!("DEBUG DAEMON: Starting whisper transcription with {} samples...", samples.len());
    state.full(params, &samples)
        .context("Failed to transcribe audio")?;
    eprintln!("DEBUG DAEMON: Whisper transcription completed in {:?}", t5.elapsed());
    
    // Get the transcribed text from segments
    let t6 = Instant::now();
    let mut text = String::new();
    let num_segments = state.full_n_segments();
    for i in 0..num_segments {
        let segment = state.get_segment(i)
            .ok_or_else(|| anyhow!("Failed to get segment {}", i))?;
        let segment_text = segment.to_str()?;
        text.push_str(segment_text);
        text.push(' ');
    }
    eprintln!("DEBUG DAEMON: Segment extraction took {:?}", t6.elapsed());
    
    eprintln!("DEBUG DAEMON: Total transcription time: {:?}", start.elapsed());
    
    Ok(text.trim().to_string())
}

