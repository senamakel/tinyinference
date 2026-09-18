//! Spawn marker for host-owned `ollama serve` processes.
//!
//! Every time `start_and_wait_for_server` actually spawns an Ollama daemon
//! (i.e. didn't adopt a healthy external one), we write a small JSON file
//! recording the PID, the binary we launched, and the openhuman process
//! that owned it. On graceful shutdown the marker is cleared. If openhuman
//! crashes before its shutdown hook fires, the marker survives — and on
//! next launch we can reclaim the orphaned daemon (kill + respawn fresh)
//! instead of either leaking it forever or running blanket `taskkill /IM
//! ollama.exe` and hitting daemons we don't own.
//!
//! Liveness is checked via `sysinfo` (already a workspace dep) to match
//! the cross-platform pattern in `install::is_ollama_installer_running`.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// On-disk record of a host-spawned `ollama serve` process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OllamaSpawnMarker {
    /// Child process identifier.
    pub pid: u32,
    /// Unix timestamp when the process was started.
    pub started_at_unix: u64,
    /// Binary path used to launch the process.
    pub binary_path: String,
    /// Process identifier of the owning host.
    pub openhuman_pid: u32,
}

impl OllamaSpawnMarker {
    /// Creates a marker owned by the current process.
    pub fn new(pid: u32, binary_path: &Path) -> Self {
        let started_at_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            pid,
            started_at_unix,
            binary_path: binary_path.display().to_string(),
            openhuman_pid: std::process::id(),
        }
    }
}

// ---- Path-keyed helpers (testable without touching ~/.openhuman/) -------

/// Write `marker` to `path`, replacing any existing file. Creates the
/// parent directory if needed. Uses a tmp-and-rename so a crash mid-write
/// can't leave truncated JSON.
pub fn write_marker_at(path: &Path, marker: &OllamaSpawnMarker) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create marker dir {}: {e}", parent.display()))?;
    }
    let json =
        serde_json::to_string_pretty(marker).map_err(|e| format!("serialize spawn marker: {e}"))?;

    let tmp = path.with_extension("spawn.tmp");
    std::fs::write(&tmp, json).map_err(|e| format!("write marker tmp {}: {e}", tmp.display()))?;
    // Rust's Windows implementation uses MoveFileExW with replacement
    // semantics. Do not fall back to remove-then-rename: that creates a crash
    // window with no ownership marker.
    if let Err(error) = std::fs::rename(&tmp, path) {
        return Err(format!(
            "rename marker {} -> {}: {error}",
            tmp.display(),
            path.display()
        ));
    }
    Ok(())
}

/// Reads a marker, returning `None` when it is missing or malformed.
pub fn read_marker_at(path: &Path) -> Option<OllamaSpawnMarker> {
    let content = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str::<OllamaSpawnMarker>(&content) {
        Ok(m) => Some(m),
        Err(e) => {
            tracing::warn!(
                "[local_ai] ollama spawn marker at {} is unparseable, ignoring: {e}",
                path.display()
            );
            None
        }
    }
}

/// Removes a marker if present.
pub fn clear_marker_at(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => tracing::debug!(
            "[local_ai] cleared ollama spawn marker at {}",
            path.display()
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(
            "[local_ai] failed to clear ollama spawn marker {}: {e}",
            path.display()
        ),
    }
}

/// True iff `pid` corresponds to a live process on this machine.
///
/// Uses `sysinfo` rather than libc/Win32 directly to stay consistent with
/// the rest of the local_ai module (see `install::is_ollama_installer_running`).
pub fn pid_is_alive(pid: u32) -> bool {
    use sysinfo::{Pid, ProcessesToUpdate, System};
    let mut sys = System::new();
    let target = Pid::from_u32(pid);
    // Refresh just the one PID we care about; cheap on Windows where a full
    // refresh can take ~tens of ms on a loaded machine.
    sys.refresh_processes(ProcessesToUpdate::Some(&[target]), true);
    sys.process(target).is_some()
}

#[cfg(test)]
#[path = "spawn_marker_test.rs"]
mod tests;
