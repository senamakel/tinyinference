//! Shared installer plumbing for the local voice stack (Piper TTS).
//!
//! Both installers need the same primitives:
//!
//! - Stream a URL to disk via `.part` suffix + atomic rename so a crash
//!   never leaves a corrupt artifact that downstream code (the STT/TTS
//!   factory) tries to load.
//! - Validate either a known SHA256 (when upstream publishes one) or a
//!   minimum size threshold so a truncated download doesn't masquerade as
//!   a finished install.
//! - Surface per-engine progress (downloading, extracting, idle, ready,
//!   error) on a polled status RPC — matches the existing
//!   `local_ai_downloads_progress` UX so the VoicePanel can reuse the
//!   same progress UI primitives without inventing a new event-bus channel.
//!
//! The Ollama installer fires-and-forgets a single PowerShell / sh block
//! and lets the OS owner that process. For Piper we need
//! finer-grained progress reporting (the GGML model file alone is up to
//! 1.6 GB and users absolutely will need a percentage indicator) so the
//! shared harness here streams the body chunks itself and updates a
//! singleton state map keyed by engine id.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Overall request timeout for a single install download. The idle timeout
/// catches dead connections; this generous cap permits multi-gigabyte models
/// on slow but active links.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);

/// Per-chunk idle timeout. If the body stream produces no bytes for this
/// long, treat the connection as dead and abort so the caller can retry
/// from a clean state. Without this guard, a half-open TCP connection (the
/// failure mode behind the "progress stuck at 18%" symptom) holds the
/// install task forever, defeating the polled-status UX.
const CHUNK_IDLE_TIMEOUT: Duration = Duration::from_secs(45);
use tokio::io::AsyncWriteExt;

/// Stable engine id for status tracking. The installer registers its progress
/// under this key; the status RPC reads it back.
pub const ENGINE_PIPER: &str = "piper";

/// Lifecycle state for a voice-engine install. Mirrors the state machine
/// the Ollama installer exposes via `LocalAiStatus.state`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VoiceInstallState {
    /// Nothing has happened — no binaries, no models. The default state
    /// when the workspace has never been touched.
    Missing,
    /// An install is in flight. `progress` and `downloaded_bytes` will be
    /// updated as chunks land.
    Installing,
    /// All required artifacts (binary + at least one default model) are
    /// present and pass validation.
    Installed,
    /// The expected install dir contains artifacts but they fail
    /// validation (e.g. size below threshold, hash mismatch, missing
    /// `.onnx.json` sidecar). The user should re-run install.
    Broken,
    /// The last install attempt errored. `error_detail` carries the
    /// human-readable reason.
    Error,
}

impl VoiceInstallState {
    /// Return the stable wire name for this lifecycle state.
    pub fn as_str(&self) -> &'static str {
        match self {
            VoiceInstallState::Missing => "missing",
            VoiceInstallState::Installing => "installing",
            VoiceInstallState::Installed => "installed",
            VoiceInstallState::Broken => "broken",
            VoiceInstallState::Error => "error",
        }
    }
}

/// Snapshot returned over JSON-RPC for one engine's installer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInstallStatus {
    /// Stable engine id (today only `"piper"`).
    pub engine: String,
    /// Current state — see [`VoiceInstallState`].
    pub state: VoiceInstallState,
    /// 0-100 percent for the in-flight download (`None` when state is not
    /// `Installing`).
    pub progress: Option<u8>,
    /// Bytes received so far.
    pub downloaded_bytes: Option<u64>,
    /// Total bytes expected (from `Content-Length` — may be `None` for
    /// chunked transfer encoding).
    pub total_bytes: Option<u64>,
    /// Free-text status line — what file we're downloading, what stage
    /// we're at. Useful for the UI to show "Downloading piper…" vs
    /// "Downloading ggml-large-v3-turbo.bin…".
    pub stage: Option<String>,
    /// Populated when `state == Error` — the user-facing failure reason.
    pub error_detail: Option<String>,
}

impl VoiceInstallStatus {
    fn missing(engine: &str) -> Self {
        Self {
            engine: engine.to_string(),
            state: VoiceInstallState::Missing,
            progress: None,
            downloaded_bytes: None,
            total_bytes: None,
            stage: None,
            error_detail: None,
        }
    }
}

/// In-memory status table — keyed by engine id. Both installers share
/// this so the status RPC can answer for either engine without a separate
/// store.
static STATUS_TABLE: once_cell::sync::Lazy<Mutex<HashMap<String, VoiceInstallStatus>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Fetch the current status snapshot for `engine`. Returns `Missing`
/// when the engine has never been touched.
pub fn read_status(engine: &str) -> VoiceInstallStatus {
    STATUS_TABLE
        .lock()
        .expect("voice install status lock poisoned")
        .get(engine)
        .cloned()
        .unwrap_or_else(|| VoiceInstallStatus::missing(engine))
}

/// Replace the snapshot for `engine`. Internal helper for the installer
/// flow — exposed at module scope so install_piper can
/// update progress without going through a public setter API.
pub fn write_status(status: VoiceInstallStatus) {
    tracing::debug!(
        "[voice-install] status update engine={} state={} progress={:?} stage={:?}",
        status.engine,
        status.state.as_str(),
        status.progress,
        status.stage,
    );
    let mut table = STATUS_TABLE
        .lock()
        .expect("voice install status lock poisoned");
    table.insert(status.engine.clone(), status);
}

/// Force a fresh missing state for `engine`. Used by tests and by the
/// "Reinstall" path before kicking off a new download.
#[cfg(test)]
pub fn reset_status(engine: &str) {
    let mut table = STATUS_TABLE
        .lock()
        .expect("voice install status lock poisoned");
    table.remove(engine);
}

/// Set of engines that currently have an install task in flight. Acts as a
/// true single-writer guard around the install-start critical section so
/// two concurrent RPC calls (a double-click, or the auto-install-on-change
/// firing in parallel with a manual button click) can't both pass an
/// `is-Installing?` snapshot check and then both spawn duplicate install
/// tasks that race on the same `.part` file inside `download_to_file`.
///
/// `STATUS_TABLE` advertises lifecycle state to the polling RPC; this set
/// owns the *start* decision. They are deliberately separate locks: the
/// status table is read on every status poll (cheap, frequent), while
/// this set is only touched on install start / end (rare).
static IN_FLIGHT: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();

fn in_flight() -> &'static Mutex<HashSet<&'static str>> {
    IN_FLIGHT.get_or_init(|| Mutex::new(HashSet::new()))
}

/// RAII guard returned by [`try_acquire_install_slot`]. Holding one of
/// these proves the caller has exclusive ownership of the install-start
/// slot for `engine`. Dropping it (including via panic unwind) releases
/// the slot so a subsequent install attempt can proceed.
///
/// The handler is expected to **move** the slot into the spawned tokio
/// task so the slot lives for the install's actual duration, not just
/// the RPC handler's lifetime. Releasing the slot when the handler
/// returns (instead of when the install finishes) would re-open the
/// race the slot was added to close.
#[derive(Debug)]
pub struct InstallSlot {
    engine: &'static str,
}

impl Drop for InstallSlot {
    fn drop(&mut self) {
        match in_flight().lock() {
            Ok(mut guard) => {
                let removed = guard.remove(self.engine);
                tracing::debug!(
                    "[voice-install] install slot released engine={} was_present={}",
                    self.engine,
                    removed
                );
            }
            Err(_) => {
                // Lock poisoned — another thread panicked while holding
                // it. The set is in an unknown state; the best we can do
                // is log and let the process continue. Subsequent
                // acquire attempts will also hit the poisoned lock and
                // surface the failure to the user.
                tracing::error!(
                    "[voice-install] install slot lock poisoned on drop for engine={}",
                    self.engine
                );
            }
        }
    }
}

/// Atomically try to claim the install-start slot for `engine`. Returns
/// `Some(InstallSlot)` if no install is currently in flight for this
/// engine, or `None` if one is already running.
///
/// This replaces the previous non-atomic `read_status` -> check ->
/// `write_status` sequence in the install RPC handlers. The
/// check-and-insert happens under a single mutex acquisition, so two
/// concurrent callers cannot both observe "not installing" and both
/// spawn tasks.
pub fn try_acquire_install_slot(engine: &'static str) -> Option<InstallSlot> {
    let mut guard = in_flight()
        .lock()
        .expect("voice install in-flight lock poisoned");
    if guard.contains(engine) {
        tracing::debug!(
            "[voice-install] install slot denied engine={} (already in flight)",
            engine
        );
        return None;
    }
    guard.insert(engine);
    tracing::debug!("[voice-install] install slot acquired engine={}", engine);
    Some(InstallSlot { engine })
}

/// Download `url` to `dest` with atomic rename. Streams bytes through
/// SHA256 if `expected_sha256` is provided, otherwise validates that the
/// final size is at least `min_bytes`.
///
/// The on-disk write goes to `<dest>.part` first and is `rename`d into
/// place only after all checks pass. If the function is interrupted
/// mid-stream (process killed, network drop) the `.part` file is the
/// only thing left behind; the next call detects and overwrites it so
/// we never read a half-written model.
///
/// Progress callbacks fire every chunk with `(downloaded_bytes,
/// total_bytes)`. Total may be `None` for chunked responses.
pub async fn download_to_file(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    min_bytes: u64,
    log_prefix: &str,
    on_progress: impl FnMut(u64, Option<u64>),
) -> crate::Result<()> {
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            crate::Error::DownloadIo(format!("{log_prefix} mkdir {}: {e}", parent.display()))
        })?;
    }

    let part_path = part_path(dest);
    // Always start from scratch — resumable HTTP Range support is
    // useful but not free (servers must return 206, hash state has to
    // restart on hash mismatch). For the MVP we restart cleanly and
    // ensure `.part` is removed first so we never accidentally append
    // to leftover bytes from an earlier failed attempt.
    if part_path.exists() {
        let _ = tokio::fs::remove_file(&part_path).await;
    }

    let safe_url = tinyinference_core::sanitize::redact_url(url);
    tracing::debug!("{log_prefix} GET {safe_url} -> {}", part_path.display());
    let client = reqwest::Client::builder()
        // 15s connect handshake; 30min overall request budget (covers 1.6 GB
        // GGML model on a 1 Mbps link). Per-chunk idle timeout is enforced
        // separately on each stream read below so half-open connections
        // fail fast instead of hanging the install task forever.
        .connect_timeout(Duration::from_secs(15))
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| crate::Error::DownloadHttp(format!("{log_prefix} build http client: {e}")))?;
    let started = Instant::now();
    let resp =
        client.get(url).send().await.map_err(|e| {
            crate::Error::DownloadHttp(format!("{log_prefix} request {safe_url}: {e}"))
        })?;
    if !resp.status().is_success() {
        return Err(crate::Error::DownloadHttp(format!(
            "{log_prefix} non-2xx response from {safe_url}: {}",
            resp.status()
        )));
    }
    let total = resp.content_length();
    tracing::debug!(
        "{log_prefix} response status={} content_length={:?}",
        resp.status(),
        total
    );

    let stream = resp.bytes_stream();
    write_download_stream(
        stream,
        DownloadTarget {
            total,
            part_path: &part_path,
            dest,
            expected_sha256,
            min_bytes,
            log_prefix,
        },
        on_progress,
    )
    .await?;
    tracing::debug!(
        "{log_prefix} downloaded -> {} elapsed_ms={}",
        dest.display(),
        started.elapsed().as_millis()
    );
    Ok(())
}

#[derive(Debug)]
struct DownloadTarget<'a> {
    total: Option<u64>,
    part_path: &'a Path,
    dest: &'a Path,
    expected_sha256: Option<&'a str>,
    min_bytes: u64,
    log_prefix: &'a str,
}

async fn write_download_stream<S, E>(
    stream: S,
    target: DownloadTarget<'_>,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> crate::Result<()>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, E>>,
    E: std::fmt::Display,
{
    let DownloadTarget {
        total,
        part_path,
        dest,
        expected_sha256,
        min_bytes,
        log_prefix,
    } = target;
    futures::pin_mut!(stream);
    let mut file = tokio::fs::File::create(part_path).await.map_err(|e| {
        crate::Error::DownloadIo(format!("{log_prefix} create {}: {e}", part_path.display()))
    })?;
    let mut hasher = expected_sha256.is_some().then(Sha256::new);
    let mut downloaded: u64 = 0;
    loop {
        // Per-chunk idle timeout — if no bytes arrive within CHUNK_IDLE_TIMEOUT,
        // bail out so a stalled half-open TCP connection doesn't hold the install
        // task forever. Clean up the .part on the way out so a retry starts fresh.
        let next = tokio::time::timeout(CHUNK_IDLE_TIMEOUT, stream.next()).await;
        let chunk = match next {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(_) => {
                drop(file);
                let _ = tokio::fs::remove_file(&part_path).await;
                return Err(crate::Error::DownloadTimeout(format!(
                    "{log_prefix} body stream idle for >{}s after {downloaded} bytes; aborting",
                    CHUNK_IDLE_TIMEOUT.as_secs()
                )));
            }
        };
        let bytes = match chunk {
            Ok(bytes) => bytes,
            Err(e) => {
                drop(file);
                let _ = tokio::fs::remove_file(&part_path).await;
                return Err(crate::Error::DownloadHttp(format!(
                    "{log_prefix} body stream: {e}"
                )));
            }
        };
        if let Some(h) = hasher.as_mut() {
            h.update(&bytes);
        }
        if let Err(e) = file.write_all(&bytes).await {
            drop(file);
            let _ = tokio::fs::remove_file(&part_path).await;
            return Err(crate::Error::DownloadIo(format!(
                "{log_prefix} write {}: {e}",
                part_path.display()
            )));
        }
        downloaded = downloaded.saturating_add(bytes.len() as u64);
        on_progress(downloaded, total);
    }
    file.flush().await.map_err(|e| {
        crate::Error::DownloadIo(format!("{log_prefix} flush {}: {e}", part_path.display()))
    })?;
    drop(file);

    if downloaded < min_bytes {
        let _ = tokio::fs::remove_file(&part_path).await;
        return Err(crate::Error::DownloadIntegrity(format!(
            "{log_prefix} downloaded payload too small: {downloaded} bytes < min {min_bytes}"
        )));
    }
    if let (Some(expected), Some(hasher)) = (expected_sha256, hasher) {
        let got = hex::encode(hasher.finalize());
        let expected_norm = expected.trim().to_ascii_lowercase();
        if got != expected_norm {
            // Never log the full file contents on mismatch — just the hashes.
            tracing::warn!(
                "{log_prefix} sha256 mismatch expected={} got={}",
                expected_norm,
                got
            );
            let _ = tokio::fs::remove_file(&part_path).await;
            return Err(crate::Error::DownloadIntegrity(format!(
                "{log_prefix} sha256 mismatch (expected {expected_norm}, got {got})"
            )));
        }
    }

    commit_download(part_path, dest, log_prefix).await?;
    Ok(())
}

async fn commit_download(part_path: &Path, dest: &Path, log_prefix: &str) -> crate::Result<()> {
    match tokio::fs::rename(part_path, dest).await {
        Ok(()) => return Ok(()),
        Err(error) if !dest.exists() => {
            return Err(crate::Error::DownloadIo(format!(
                "{log_prefix} rename {} -> {}: {error}",
                part_path.display(),
                dest.display()
            )));
        }
        Err(_) => {}
    }

    // Windows cannot rename over an existing file. Move the old artifact to a
    // sibling backup and restore it if committing the validated replacement
    // fails, so a failed reinstall never destroys the working copy.
    let backup = dest.with_extension("replace-backup");
    if backup.exists() {
        tokio::fs::remove_file(&backup).await.map_err(|e| {
            crate::Error::DownloadIo(format!(
                "{log_prefix} remove stale backup {}: {e}",
                backup.display()
            ))
        })?;
    }
    tokio::fs::rename(dest, &backup).await.map_err(|e| {
        crate::Error::DownloadIo(format!(
            "{log_prefix} stage existing {} -> {}: {e}",
            dest.display(),
            backup.display()
        ))
    })?;
    if let Err(error) = tokio::fs::rename(part_path, dest).await {
        let restore = tokio::fs::rename(&backup, dest).await;
        return Err(crate::Error::DownloadIo(format!(
            "{log_prefix} commit replacement {} -> {}: {error}; restore={restore:?}",
            part_path.display(),
            dest.display()
        )));
    }
    if let Err(error) = tokio::fs::remove_file(&backup).await {
        tracing::warn!(
            "{log_prefix} committed replacement but could not remove backup {}: {error}",
            backup.display()
        );
    }
    Ok(())
}

/// Produce the `.part` sibling of `dest`. Helper kept testable.
pub fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

#[cfg(test)]
#[path = "download_test.rs"]
mod tests;
