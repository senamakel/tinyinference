//! Piper installer — downloads the platform-specific Piper binary
//! archive and the bundled `en_US-lessac-medium` voice (`.onnx` +
//! `.onnx.json` sidecar) into the workspace.
//!
//! Voice IDs other than the bundled default are intentionally out of
//! scope; the VoicePanel exposes a free-text `tts_voice_id` input so
//! advanced users can manually drop in additional `.onnx` files alongside
//! the bundled one (see Voice TTS factory docs).

use std::path::PathBuf;

use crate::download::{
    ENGINE_PIPER, VoiceInstallState, VoiceInstallStatus, download_to_file, read_status,
    try_acquire_install_slot, write_status,
};
use crate::{Error, Result};

const LOG_PREFIX: &str = "[voice-install:piper]";

/// Filesystem layout used by the Piper installer.
#[derive(Debug, Clone)]
pub struct PiperInstall {
    root: PathBuf,
}

impl PiperInstall {
    /// Creates an installer rooted at the host-selected Piper directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Returns the installation root.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// Returns the model and JSON sidecar paths for a voice identifier.
    pub fn voice_paths(&self, voice_id: &str) -> Option<(PathBuf, PathBuf)> {
        let trimmed = voice_id.trim();
        if !voice_id_is_safe(trimmed) {
            return None;
        }
        let base = self.root.join("voices").join(trimmed);
        Some((
            base.with_extension("onnx"),
            base.with_extension("onnx.json"),
        ))
    }

    /// Returns candidate paths for the extracted Piper executable.
    pub fn binary_candidates(&self) -> Vec<PathBuf> {
        let binary = if cfg!(windows) { "piper.exe" } else { "piper" };
        vec![
            self.root.join(binary),
            self.root.join("piper").join(binary),
            self.root.join("bin").join(binary),
        ]
    }
}

fn voice_id_is_safe(voice_id: &str) -> bool {
    !voice_id.is_empty()
        && voice_id != "."
        && voice_id != ".."
        && !voice_id.contains("..")
        && voice_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

/// Default voice identifier shipped with the installer.
pub const DEFAULT_PIPER_VOICE: &str = "en_US-lessac-medium";

/// Minimum bytes for the Piper release archive. The smallest historical
/// build is ~7 MB; below 1 MB is almost certainly an error response.
const MIN_BINARY_ARCHIVE_BYTES: u64 = 1024 * 1024;

/// Minimum bytes for the voice `.onnx` model. `en_US-lessac-medium.onnx`
/// is ~60 MB; allow some slack for CDN compression differences.
const MIN_VOICE_BYTES: u64 = 30 * 1024 * 1024;

/// Minimum bytes for the `.onnx.json` sidecar. The file is human-readable
/// JSON, typically a few KB; anything below 256 bytes is almost certainly
/// a 404 HTML response masquerading as JSON.
const MIN_VOICE_JSON_BYTES: u64 = 256;
const MAX_ARCHIVE_COMPRESSED_BYTES: u64 = 256 * 1024 * 1024;
const MAX_ARCHIVE_EXPANDED_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_ARCHIVE_ENTRY_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 20_000;

/// Binaries shipped inside the Piper release archive that must carry the
/// executable bit for the engine to start.
///
/// The upstream macOS tarballs ship `espeak-ng` with mode `0644` (#5045).
/// `tar::Archive::unpack` faithfully reproduces the archived mode, so a
/// plain extraction leaves the file non-executable and Piper fails when it
/// shells out to phonemize. Extraction must therefore repair the bit
/// rather than trust the archive.
///
/// Unix-only: Windows has no executable bit, so the const would be dead
/// code there.
#[cfg(unix)]
const PIPER_EXECUTABLES: [&str; 3] = ["piper", "piper_phonemize", "espeak-ng"];

/// Result of resolving the Piper binary archive URL for the host OS.
struct BinaryAsset {
    url: String,
    /// Archive shape — drives the extraction strategy.
    kind: ArchiveKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchiveKind {
    Zip,
    TarGz,
}

/// Per-OS Piper release asset URL. The Piper project publishes one
/// archive per OS/architecture under the `latest` release alias. Names
/// have been stable across recent releases.
/// Read a Piper base-URL override, treating empty / whitespace-only values as
/// unset and stripping surrounding whitespace + trailing slashes so the asset
/// paths concatenated onto it never produce a doubled slash.
fn piper_base_override(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty())
}

fn binary_download_asset() -> Option<BinaryAsset> {
    let base = piper_base_override("OPENHUMAN_PIPER_RELEASE_BASE_URL")
        .unwrap_or_else(|| "https://github.com/rhasspy/piper/releases/latest/download".to_string());
    binary_asset_for(std::env::consts::OS, std::env::consts::ARCH, &base)
}

fn binary_asset_for(os: &str, arch: &str, base: &str) -> Option<BinaryAsset> {
    let (suffix, kind) = match (os, arch) {
        ("windows", "x86_64") => ("windows_amd64", ArchiveKind::Zip),
        ("macos", "x86_64") => ("macos_x64", ArchiveKind::TarGz),
        ("macos", "aarch64" | "arm64") => ("macos_aarch64", ArchiveKind::TarGz),
        ("linux", "x86_64") => ("linux_x86_64", ArchiveKind::TarGz),
        ("linux", "aarch64" | "arm64") => ("linux_aarch64", ArchiveKind::TarGz),
        ("linux", "armv7" | "arm") => ("linux_armv7", ArchiveKind::TarGz),
        _ => return None,
    };
    tracing::info!("{LOG_PREFIX} host os={os} arch={arch} selecting asset=piper_{suffix}");
    let extension = if kind == ArchiveKind::Zip {
        "zip"
    } else {
        "tar.gz"
    };
    Some(BinaryAsset {
        url: format!("{base}/piper_{suffix}.{extension}"),
        kind,
    })
}

/// Voice file URLs on HuggingFace. Returns `(onnx_url, onnx_json_url)`.
fn voice_download_urls(voice_id: &str) -> Option<(String, String)> {
    // The Piper voices repo uses the structure:
    //   en/en_US/lessac/medium/en_US-lessac-medium.onnx
    //   en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
    // We only support the bundled default — multi-voice support is
    // tracked separately. The path components mirror the voice id.
    let (lang_short, locale, name, quality) = decode_voice_id(voice_id)?;
    let base = match piper_base_override("OPENHUMAN_PIPER_VOICES_BASE_URL") {
        Some(root) => format!("{root}/{lang_short}/{locale}/{name}/{quality}"),
        None => format!(
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_short}/{locale}/{name}/{quality}"
        ),
    };
    let stem = format!("{locale}-{name}-{quality}");
    Some((
        format!("{base}/{stem}.onnx"),
        format!("{base}/{stem}.onnx.json"),
    ))
}

/// Decompose `en_US-lessac-medium` into its repo-path pieces.
///
/// Returns `(short_lang, locale, voice_name, quality)`.
fn decode_voice_id(voice_id: &str) -> Option<(String, String, String, String)> {
    let trimmed = voice_id.trim();
    let id = if trimmed.is_empty() {
        DEFAULT_PIPER_VOICE
    } else {
        trimmed
    };
    let parts: Vec<&str> = id.split('-').collect();
    if parts.len() < 3 {
        return None;
    }
    let locale = parts[0].to_string();
    let name = parts[1].to_string();
    let quality = parts[2..].join("-");
    let short_lang = locale.split('_').next().unwrap_or("en").to_string();
    Some((short_lang, locale, name, quality))
}

/// Convenience: read the current installer status snapshot, falling back
/// to "installed" when on-disk artifacts pass validation.
pub fn status(install: &PiperInstall, voice_id: &str) -> VoiceInstallStatus {
    let mut snapshot = read_status(ENGINE_PIPER);
    let configured_voice = voice_id.trim_end_matches(".onnx");
    if installed_artifacts_ok(install, configured_voice) {
        snapshot.state = VoiceInstallState::Installed;
        snapshot.stage = Some("binary and voice present".to_string());
    } else if matches!(snapshot.state, VoiceInstallState::Installed) {
        snapshot.state = VoiceInstallState::Missing;
        snapshot.progress = None;
        snapshot.stage = None;
    }
    snapshot
}

fn installed_artifacts_ok(install: &PiperInstall, voice_id: &str) -> bool {
    // Check the SPECIFIC requested voice, not the hard-coded default.
    // Without this, switching voice via the dropdown would short-circuit
    // with "already installed" and never fetch the new `.onnx`.
    let voice_ok = install
        .voice_paths(voice_id)
        .map(|(onnx, json)| {
            let onnx_ok = std::fs::metadata(&onnx)
                .map(|m| m.is_file() && m.len() >= MIN_VOICE_BYTES)
                .unwrap_or(false);
            let json_ok = std::fs::metadata(&json)
                .map(|m| m.is_file() && m.len() >= MIN_VOICE_JSON_BYTES)
                .unwrap_or(false);
            tracing::debug!(
                "{LOG_PREFIX} install check onnx={} onnx_ok={} json={} json_ok={}",
                onnx.display(),
                onnx_ok,
                json.display(),
                json_ok
            );
            onnx_ok && json_ok
        })
        .unwrap_or(false);
    let binary_ok = find_workspace_piper_binary(install).is_some();
    tracing::debug!(
        "{LOG_PREFIX} install check binary_ok={} voice_ok={}",
        binary_ok,
        voice_ok
    );
    binary_ok && voice_ok
}

/// Kick off (or re-kick) a Piper install. `force_reinstall = true`
/// removes any existing voice file first; otherwise an already-installed
/// engine returns immediately with a no-op success.
pub async fn install_piper(
    install: &PiperInstall,
    voice_id: Option<String>,
    force_reinstall: bool,
) -> Result<VoiceInstallStatus> {
    let voice = voice_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_PIPER_VOICE)
        .to_string();
    if install.voice_paths(&voice).is_none() {
        return Err(Error::InvalidInput(
            "Piper voice ID must be a locale-name-quality ASCII filename component".to_string(),
        ));
    }
    if decode_voice_id(&voice).is_none() {
        return Err(Error::InvalidInput(
            "Piper voice ID must have locale-name-quality segments".to_string(),
        ));
    }
    let _slot = try_acquire_install_slot(ENGINE_PIPER)
        .ok_or_else(|| Error::InstallInProgress(ENGINE_PIPER.to_string()))?;
    tracing::debug!(
        "{LOG_PREFIX} install requested voice={voice} force_reinstall={force_reinstall}"
    );

    if !force_reinstall {
        // Repair archives installed by older versions before deciding whether
        // the existing layout is runnable. Status checks remain read-only.
        ensure_executable_bits(install.root());
    }
    if !force_reinstall && installed_artifacts_ok(install, &voice) {
        tracing::debug!("{LOG_PREFIX} short-circuit: artifacts already present");
        // Repair permissions on the EXISTING install before reporting success.
        // Users hit by #5045 already extracted the affected archive, so they
        // reach this branch on every launch and would never run the repair that
        // only lived on the fresh-install path — they would keep seeing TTS
        // failures until they manually forced a reinstall. `ensure_executable_bits`
        // is a no-op when the bits are already set, so this costs a stat per
        // executable on the happy path.
        if find_workspace_piper_binary(install).is_some() {
            let snapshot = VoiceInstallStatus {
                engine: ENGINE_PIPER.to_string(),
                state: VoiceInstallState::Installed,
                progress: Some(100),
                downloaded_bytes: None,
                total_bytes: None,
                stage: Some("already installed".to_string()),
                error_detail: None,
            };
            write_status(snapshot.clone());
            return Ok(snapshot);
        }
        tracing::warn!("{LOG_PREFIX} existing artifacts have no usable executable; reinstalling");
    }

    write_status(VoiceInstallStatus {
        engine: ENGINE_PIPER.to_string(),
        state: VoiceInstallState::Installing,
        progress: Some(0),
        downloaded_bytes: Some(0),
        total_bytes: None,
        stage: Some(format!("starting piper install ({voice})")),
        error_detail: None,
    });

    let result = run_install(install, &voice).await;
    match result {
        Ok(()) => {
            let snapshot = VoiceInstallStatus {
                engine: ENGINE_PIPER.to_string(),
                state: VoiceInstallState::Installed,
                progress: Some(100),
                downloaded_bytes: None,
                total_bytes: None,
                stage: Some("install complete".to_string()),
                error_detail: None,
            };
            write_status(snapshot.clone());
            Ok(snapshot)
        }
        Err(msg) => {
            let snapshot = VoiceInstallStatus {
                engine: ENGINE_PIPER.to_string(),
                state: VoiceInstallState::Error,
                progress: None,
                downloaded_bytes: None,
                total_bytes: None,
                stage: None,
                error_detail: Some(msg.to_string()),
            };
            write_status(snapshot.clone());
            Err(msg)
        }
    }
}

async fn run_install(install: &PiperInstall, voice: &str) -> Result<()> {
    let root = install.root();
    let parent = root
        .parent()
        .ok_or_else(|| Error::Install(format!("{LOG_PREFIX} install root has no parent")))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| Error::Install(format!("{LOG_PREFIX} create install parent: {e}")))?;
    static STAGE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let sequence = STAGE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stage = parent.join(format!(
        ".piper-installing-{}-{sequence}",
        std::process::id()
    ));
    if stage.exists() {
        std::fs::remove_dir_all(&stage).map_err(|e| {
            Error::Install(format!("{LOG_PREFIX} remove stale staging directory: {e}"))
        })?;
    }
    if root.exists() {
        copy_directory(root, &stage).map_err(Error::Install)?;
    } else {
        std::fs::create_dir_all(&stage)
            .map_err(|e| Error::Install(format!("{LOG_PREFIX} create staging directory: {e}")))?;
    }
    let staged_install = PiperInstall::new(&stage);
    if let Err(error) = run_install_into(&staged_install, voice).await {
        let _ = std::fs::remove_dir_all(&stage);
        return Err(error);
    }
    if !installed_artifacts_ok(&staged_install, voice)
        || find_workspace_piper_binary(&staged_install).is_none()
    {
        let _ = std::fs::remove_dir_all(&stage);
        return Err(Error::Install(format!(
            "{LOG_PREFIX} staged installation has no usable Piper executable or voice"
        )));
    }
    commit_staged_directory(&stage, root).map_err(Error::Install)?;
    Ok(())
}

async fn run_install_into(install: &PiperInstall, voice: &str) -> Result<()> {
    // 1) Voice files: `.onnx` (heavy) + `.onnx.json` (small sidecar).
    let (onnx_url, json_url) = voice_download_urls(voice).ok_or_else(|| {
        Error::InvalidInput("Piper voice ID must have locale-name-quality segments".to_string())
    })?;
    let (onnx_path, json_path) = install
        .voice_paths(voice)
        .ok_or_else(|| Error::InvalidInput(format!("could not resolve Piper voice '{voice}'")))?;

    tracing::debug!(
        "{LOG_PREFIX} downloading voice url={}",
        tinyinference_core::sanitize::redact_url(&onnx_url)
    );
    update_stage(format!("downloading {voice}.onnx"));
    download_to_file(
        &onnx_url,
        &onnx_path,
        None,
        MIN_VOICE_BYTES,
        LOG_PREFIX,
        |downloaded, total| {
            let progress = total
                .filter(|t| *t > 0)
                .map(|t| ((downloaded * 100) / t).min(100) as u8);
            write_status(VoiceInstallStatus {
                engine: ENGINE_PIPER.to_string(),
                state: VoiceInstallState::Installing,
                progress,
                downloaded_bytes: Some(downloaded),
                total_bytes: total,
                stage: Some("downloading voice (.onnx)".to_string()),
                error_detail: None,
            });
        },
    )
    .await?;
    tracing::debug!("{LOG_PREFIX} voice .onnx staged at {}", onnx_path.display());

    tracing::debug!(
        "{LOG_PREFIX} downloading voice json url={}",
        tinyinference_core::sanitize::redact_url(&json_url)
    );
    update_stage(format!("downloading {voice}.onnx.json"));
    download_to_file(
        &json_url,
        &json_path,
        None,
        MIN_VOICE_JSON_BYTES,
        LOG_PREFIX,
        |downloaded, total| {
            let progress = total
                .filter(|t| *t > 0)
                .map(|t| ((downloaded * 100) / t).min(100) as u8);
            write_status(VoiceInstallStatus {
                engine: ENGINE_PIPER.to_string(),
                state: VoiceInstallState::Installing,
                progress,
                downloaded_bytes: Some(downloaded),
                total_bytes: total,
                stage: Some("downloading voice (.onnx.json)".to_string()),
                error_detail: None,
            });
        },
    )
    .await?;

    // 2) Binary archive.
    let asset = binary_download_asset().ok_or_else(|| {
        Error::Install(format!(
            "{LOG_PREFIX} no piper binary release for this OS/arch"
        ))
    })?;
    let archive_name = asset
        .url
        .rsplit('/')
        .next()
        .unwrap_or("piper_archive")
        .to_string();
    let archive_path = install.root().join(&archive_name);
    tracing::debug!(
        "{LOG_PREFIX} downloading binary url={}",
        tinyinference_core::sanitize::redact_url(&asset.url)
    );
    update_stage("downloading piper binary".to_string());
    download_to_file(
        &asset.url,
        &archive_path,
        None,
        MIN_BINARY_ARCHIVE_BYTES,
        LOG_PREFIX,
        |downloaded, total| {
            let progress = total
                .filter(|t| *t > 0)
                .map(|t| ((downloaded * 100) / t).min(100) as u8);
            write_status(VoiceInstallStatus {
                engine: ENGINE_PIPER.to_string(),
                state: VoiceInstallState::Installing,
                progress,
                downloaded_bytes: Some(downloaded),
                total_bytes: total,
                stage: Some("downloading binary".to_string()),
                error_detail: None,
            });
        },
    )
    .await?;
    update_stage("extracting piper binary".to_string());
    let dest = install.root().to_path_buf();
    match asset.kind {
        ArchiveKind::Zip => extract_zip(&archive_path, &dest).map_err(Error::Install)?,
        ArchiveKind::TarGz => extract_tar_gz(&archive_path, &dest).map_err(Error::Install)?,
    }
    ensure_executable_bits(&dest);
    if find_workspace_piper_binary(install).is_none() {
        return Err(Error::Install(format!(
            "{LOG_PREFIX} extracted archive contains no usable Piper executable"
        )));
    }
    if let Err(e) = std::fs::remove_file(&archive_path) {
        tracing::warn!(
            "{LOG_PREFIX} could not remove archive {}: {e}",
            archive_path.display()
        );
    }

    Ok(())
}

fn copy_directory(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> std::result::Result<(), String> {
    std::fs::create_dir_all(destination)
        .map_err(|e| format!("{LOG_PREFIX} create {}: {e}", destination.display()))?;
    for entry in std::fs::read_dir(source)
        .map_err(|e| format!("{LOG_PREFIX} read {}: {e}", source.display()))?
    {
        let entry = entry.map_err(|e| format!("{LOG_PREFIX} read directory entry: {e}"))?;
        let kind = entry
            .file_type()
            .map_err(|e| format!("{LOG_PREFIX} inspect {}: {e}", entry.path().display()))?;
        let target = destination.join(entry.file_name());
        if kind.is_dir() {
            copy_directory(&entry.path(), &target)?;
        } else if kind.is_file() {
            std::fs::copy(entry.path(), &target)
                .map_err(|e| format!("{LOG_PREFIX} copy {}: {e}", entry.path().display()))?;
        } else {
            return Err(format!(
                "{LOG_PREFIX} refusing to stage non-file entry {}",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

fn commit_staged_directory(
    stage: &std::path::Path,
    destination: &std::path::Path,
) -> std::result::Result<(), String> {
    let backup = destination.with_extension("install-backup");
    if backup.exists() {
        std::fs::remove_dir_all(&backup)
            .map_err(|e| format!("{LOG_PREFIX} remove stale backup: {e}"))?;
    }
    if destination.exists() {
        std::fs::rename(destination, &backup)
            .map_err(|e| format!("{LOG_PREFIX} stage previous installation: {e}"))?;
    }
    if let Err(error) = std::fs::rename(stage, destination) {
        let restore = if backup.exists() {
            std::fs::rename(&backup, destination)
        } else {
            Ok(())
        };
        return Err(format!(
            "{LOG_PREFIX} commit staged installation: {error}; restore={restore:?}"
        ));
    }
    if backup.exists()
        && let Err(error) = std::fs::remove_dir_all(&backup)
    {
        tracing::warn!(
            "{LOG_PREFIX} committed installation but could not remove backup {}: {error}",
            backup.display()
        );
    }
    Ok(())
}

/// Directories under the install root where an extracted Piper binary can
/// land. Mirrors the layouts probed by
/// [`PiperInstall::binary_candidates`]: the macOS/Linux tarballs
/// nest under `piper/`, some builds flatten to the root, and a few use
/// `bin/`. Bounded on purpose — a recursive walk would descend into
/// `espeak-ng-data/` and `voices/` (which holds a ~60 MB `.onnx`) for no
/// benefit.
#[cfg(unix)]
fn executable_search_dirs(dest_dir: &std::path::Path) -> [PathBuf; 3] {
    [
        dest_dir.to_path_buf(),
        dest_dir.join("piper"),
        dest_dir.join("bin"),
    ]
}

/// Repair the executable bit on the binaries extracted from the Piper
/// archive.
///
/// Upstream ships `espeak-ng` as mode `0644` inside both macOS tarballs
/// (#5045) and `tar::Archive::unpack` reproduces the archived mode
/// verbatim, so the extracted tree is left with a non-executable
/// `espeak-ng`. Rather than trust archive metadata, set `0o755` on every
/// binary we know Piper needs.
///
/// Best-effort by design: a `chmod` failure is logged but does not fail
/// the install, since the caller may still have a working engine on
/// `PATH` (`PIPER_BIN`) and a hard error here would regress that path.
#[cfg(unix)]
fn ensure_executable_bits(dest_dir: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;

    for dir in executable_search_dirs(dest_dir) {
        for name in PIPER_EXECUTABLES {
            let path = dir.join(name);
            let Ok(meta) = std::fs::metadata(&path) else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            let mode = meta.permissions().mode();
            // Any of user/group/other execute already set → leave it be.
            if mode & 0o111 != 0 {
                tracing::debug!(
                    "{LOG_PREFIX} {} already executable (mode {:o})",
                    path.display(),
                    mode & 0o777
                );
                continue;
            }
            match std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)) {
                Ok(()) => tracing::info!(
                    "{LOG_PREFIX} repaired execute bit on {} (was mode {:o})",
                    path.display(),
                    mode & 0o777
                ),
                Err(e) => tracing::warn!("{LOG_PREFIX} could not chmod +x {}: {e}", path.display()),
            }
        }
    }
}

/// Windows has no executable bit — extraction is sufficient there.
#[cfg(not(unix))]
fn ensure_executable_bits(_dest_dir: &std::path::Path) {}

fn update_stage(stage: String) {
    let mut current = read_status(ENGINE_PIPER);
    current.stage = Some(stage);
    write_status(current);
}

fn extract_zip(
    zip_path: &std::path::Path,
    dest_dir: &std::path::Path,
) -> std::result::Result<(), String> {
    tracing::debug!(
        "{LOG_PREFIX} extract_zip {} -> {}",
        zip_path.display(),
        dest_dir.display()
    );
    let file = std::fs::File::open(zip_path).map_err(|e| format!("{LOG_PREFIX} open zip: {e}"))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("{LOG_PREFIX} parse zip: {e}"))?;
    if archive.len() > MAX_ARCHIVE_ENTRIES {
        return Err(format!("{LOG_PREFIX} zip contains too many entries"));
    }
    std::fs::create_dir_all(dest_dir).map_err(|e| format!("{LOG_PREFIX} mkdir dest: {e}"))?;
    let mut expanded = 0u64;
    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("{LOG_PREFIX} zip entry {i}: {e}"))?;
        let Some(rel) = entry.enclosed_name() else {
            continue;
        };
        let rel = rel.to_path_buf();
        if entry.size() > MAX_ARCHIVE_ENTRY_BYTES {
            return Err(format!("{LOG_PREFIX} zip entry {i} exceeds size limit"));
        }
        expanded = expanded
            .checked_add(entry.size())
            .filter(|size| *size <= MAX_ARCHIVE_EXPANDED_BYTES)
            .ok_or_else(|| format!("{LOG_PREFIX} zip expanded size exceeds limit"))?;
        let out_path = dest_dir.join(&rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)
                .map_err(|e| format!("{LOG_PREFIX} mkdir {}: {e}", out_path.display()))?;
        } else {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{LOG_PREFIX} mkdir {}: {e}", parent.display()))?;
            }
            let mut out = std::fs::File::create(&out_path)
                .map_err(|e| format!("{LOG_PREFIX} create {}: {e}", out_path.display()))?;
            std::io::copy(&mut entry, &mut out)
                .map_err(|e| format!("{LOG_PREFIX} copy {}: {e}", out_path.display()))?;
        }
    }
    Ok(())
}

fn extract_tar_gz(
    archive: &std::path::Path,
    dest_dir: &std::path::Path,
) -> std::result::Result<(), String> {
    tracing::debug!(
        "{LOG_PREFIX} extract_tar_gz {} -> {}",
        archive.display(),
        dest_dir.display()
    );
    std::fs::create_dir_all(dest_dir).map_err(|e| format!("{LOG_PREFIX} mkdir dest: {e}"))?;
    let metadata =
        std::fs::metadata(archive).map_err(|e| format!("{LOG_PREFIX} inspect tar.gz: {e}"))?;
    if metadata.len() > MAX_ARCHIVE_COMPRESSED_BYTES {
        return Err(format!(
            "{LOG_PREFIX} compressed archive exceeds size limit"
        ));
    }
    let file =
        std::fs::File::open(archive).map_err(|e| format!("{LOG_PREFIX} open tar.gz: {e}"))?;
    let decoder = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
    let mut tar = tar::Archive::new(decoder);
    let entries = tar
        .entries()
        .map_err(|e| format!("{LOG_PREFIX} read tar entries: {e}"))?;
    let mut count = 0usize;
    let mut expanded = 0u64;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("{LOG_PREFIX} read tar entry: {e}"))?;
        count += 1;
        if count > MAX_ARCHIVE_ENTRIES {
            return Err(format!("{LOG_PREFIX} tar contains too many entries"));
        }
        let size = entry.size();
        if size > MAX_ARCHIVE_ENTRY_BYTES {
            return Err(format!("{LOG_PREFIX} tar entry exceeds size limit"));
        }
        expanded = expanded
            .checked_add(size)
            .filter(|total| *total <= MAX_ARCHIVE_EXPANDED_BYTES)
            .ok_or_else(|| format!("{LOG_PREFIX} tar expanded size exceeds limit"))?;
        if !entry
            .unpack_in(dest_dir)
            .map_err(|e| format!("{LOG_PREFIX} unpack tar entry: {e}"))?
        {
            return Err(format!("{LOG_PREFIX} tar entry escapes destination"));
        }
    }
    Ok(())
}

/// Returns the workspace-installed Piper binary path if one exists.
/// Hosts can use this result before falling back to `PIPER_BIN` or `PATH`.
pub fn find_workspace_piper_binary(install: &PiperInstall) -> Option<PathBuf> {
    let candidates = install.binary_candidates();
    for candidate in candidates {
        if !candidate.is_file() {
            continue;
        }
        // A file we cannot execute is not a usable candidate. Returning it
        // anyway would pin resolution to the broken workspace copy and make the
        // `PIPER_BIN` / PATH fallback unreachable, which is exactly what
        // happens when the `chmod` repair fails (denied, or a filesystem with
        // no execute bit). Skipping lets resolution continue to a working
        // engine instead of failing at launch.
        if !is_executable_file(&candidate) {
            tracing::warn!(
                "{LOG_PREFIX} skipping non-executable workspace piper binary {} — falling back to PIPER_BIN/PATH",
                candidate.display()
            );
            continue;
        }
        tracing::debug!(
            "{LOG_PREFIX} found workspace piper binary at {}",
            candidate.display()
        );
        return Some(candidate);
    }
    None
}

/// Whether `path` carries an execute bit for anybody.
///
/// Windows has no execute bit, so every regular file qualifies there.
#[cfg(unix)]
fn is_executable_file(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable_file(_path: &std::path::Path) -> bool {
    true
}

#[cfg(test)]
#[path = "piper_test.rs"]
mod tests;
