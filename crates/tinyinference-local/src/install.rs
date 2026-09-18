//! Automatic Ollama installer and system binary discovery.

use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// Name of the Inno Setup installer process. On Windows the installer is
/// spawned via PowerShell's `Start-Process`, which creates a top-level
/// process — it survives the parent OpenHuman process dying. If OpenHuman
/// is killed mid-install (or the user closes the app and reopens it before
/// install completes) we need to detect the in-flight installer instead
/// of launching a second one that would race on the same install dir.
#[cfg(windows)]
const OLLAMA_INSTALLER_PROCESS_NAME: &str = "OllamaSetup.exe";

/// Returns `true` when a Windows OllamaSetup.exe process is currently
/// running anywhere on the machine. macOS / Linux installs are spawned
/// as `sh` children of the Rust core and cannot orphan past it; the
/// non-Windows branch is therefore a constant `false`.
#[cfg(windows)]
pub fn is_ollama_installer_running() -> bool {
    use sysinfo::{ProcessesToUpdate, System};
    let mut sys = System::new();
    // refresh_processes is sufficient — we only need the process list, not
    // CPU/memory/disk/network. refresh_all() adds dozens of milliseconds of
    // blocking I/O on a loaded Windows machine and this function runs on the
    // async executor inside download_and_install_ollama.
    // sysinfo 0.33 added a second `remove_dead_processes` parameter.
    sys.refresh_processes(ProcessesToUpdate::All, true);
    sys.processes().values().any(|p| {
        p.name()
            .to_string_lossy()
            .eq_ignore_ascii_case(OLLAMA_INSTALLER_PROCESS_NAME)
    })
}

#[cfg(not(windows))]
/// Return whether a detached Ollama installer is currently running.
pub fn is_ollama_installer_running() -> bool {
    false
}

/// Captured output from the Ollama install script.
#[derive(Debug)]
/// Captured output from a platform Ollama installation command.
pub struct InstallResult {
    /// Installer process exit status.
    pub exit_status: std::process::ExitStatus,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
}

/// Run the platform-specific Ollama install into the workspace and capture stdout/stderr.
pub async fn run_ollama_install_script(install_dir: &Path) -> Result<InstallResult> {
    let mut cmd = build_install_command(install_dir).map_err(Error::Install)?;

    let output = cmd
        .output()
        .await
        .map_err(|e| Error::Install(format!("failed to execute Ollama installer: {e}")))?;

    tracing::debug!(
        "[local_ai] Ollama install script finished (dir={} exit={}) stdout={} stderr={}",
        install_dir.display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    Ok(InstallResult {
        exit_status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

#[cfg(target_os = "windows")]
pub(crate) fn resolve_powershell_executable() -> std::ffi::OsString {
    // `Command::new("powershell")` relies on PATH. When OpenHuman.exe is
    // spawned by `cargo tauri dev` (or similar dev harnesses) the inherited
    // PATH can be sanitized down to a subset that excludes the
    // `WindowsPowerShell\v1.0` dir, and the spawn fails with `program not
    // found`. Probe known absolute locations first and fall back to the
    // bare name if none are present.
    //
    // %SystemRoot% defaults to `C:\Windows` and is always set on Windows
    // sessions.
    let system_root =
        std::env::var_os("SystemRoot").unwrap_or_else(|| std::ffi::OsString::from("C:\\Windows"));
    for relative in [
        "System32\\WindowsPowerShell\\v1.0\\powershell.exe",
        "SysWOW64\\WindowsPowerShell\\v1.0\\powershell.exe",
    ] {
        let mut candidate = std::path::PathBuf::from(&system_root);
        candidate.push(relative);
        if candidate.is_file() {
            return candidate.into_os_string();
        }
    }
    // PowerShell 7 (`pwsh.exe`) is another viable substitute when present.
    if let Ok(pf) = std::env::var("ProgramFiles") {
        let p7 = std::path::PathBuf::from(pf)
            .join("PowerShell")
            .join("7")
            .join("pwsh.exe");
        if p7.is_file() {
            return p7.into_os_string();
        }
    }
    std::ffi::OsString::from("powershell")
}

fn build_install_command(
    install_dir: &Path,
) -> std::result::Result<tokio::process::Command, String> {
    #[cfg(target_os = "windows")]
    {
        let powershell_exe = resolve_powershell_executable();
        tracing::debug!(
            "[local_ai] resolved powershell for installer: {}",
            powershell_exe.to_string_lossy()
        );
        let mut cmd = tokio::process::Command::new(&powershell_exe);
        // Kill the PowerShell child if the spawning future is dropped — e.g.
        // the in-process tokio runtime shuts down because the user closed
        // OpenHuman mid-install. Without this, `cmd.output().await` keeps
        // OpenHuman.exe alive (and port 7788 bound) for the full 60–120s of
        // the install download, producing zombie processes and
        // "port in use" errors on the next launch. Note: OllamaSetup.exe is
        // spawned by PowerShell as a TOP-LEVEL process (via `Start-Process`),
        // so it survives PowerShell's death. That's intentional — the
        // crash-resume detection in `is_ollama_installer_running` picks it
        // up on the next OpenHuman launch and waits.
        cmd.kill_on_drop(true);
        super::process::apply_no_window(&mut cmd);
        cmd.env("OPENHUMAN_OLLAMA_INSTALL_DIR", install_dir);
        cmd.args([
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            r#"
            $ErrorActionPreference = "Stop"
            $ProgressPreference = "SilentlyContinue"
            $installDir = $env:OPENHUMAN_OLLAMA_INSTALL_DIR
            New-Item -ItemType Directory -Path $installDir -Force | Out-Null
            $installerUrl = "https://ollama.com/download/OllamaSetup.exe"
            $tempInstaller = Join-Path ([IO.Path]::GetTempPath()) ("OpenHuman-Ollama-{0}.exe" -f [Guid]::NewGuid())
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $installerUrl -OutFile $tempInstaller
                $signature = Get-AuthenticodeSignature -FilePath $tempInstaller
                if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch "Ollama") {
                    throw "Downloaded Ollama installer has an invalid or unexpected Authenticode signature"
                }
                # /SILENT (not /VERYSILENT) keeps the OS-owned progress dialog visible.
                $args = "/SILENT /NORESTART /SUPPRESSMSGBOXES /CURRENTUSER /DIR=""$installDir"""
                $proc = Start-Process -FilePath $tempInstaller -ArgumentList $args -PassThru
                $proc.WaitForExit()
                if ($proc.ExitCode -ne 0) {
                    throw "Installation failed with exit code $($proc.ExitCode)"
                }
            } finally {
                Remove-Item $tempInstaller -Force -ErrorAction SilentlyContinue
            }
            "#,
        ]);
        return Ok(cmd);
    }

    #[cfg(target_os = "macos")]
    {
        let mut cmd = tokio::process::Command::new("sh");
        // Same rationale as the Windows branch: kill the child if the
        // spawning task is dropped (e.g. app shutdown mid-install) so the
        // tokio runtime can exit cleanly instead of waiting for curl to
        // finish downloading the full Ollama.app bundle.
        cmd.kill_on_drop(true);
        cmd.env("OPENHUMAN_OLLAMA_INSTALL_DIR", install_dir);
        cmd.arg("-lc")
            .arg(
                r#"
                set -eu
                for tool in curl unzip mktemp rm cp chmod mkdir mv; do
                  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
                done
                dest="$OPENHUMAN_OLLAMA_INSTALL_DIR"
                tmp_dir="$(mktemp -d)"
                stage="${dest}.installing.$$"
                backup="${dest}.backup.$$"
                cleanup() {
                  rm -rf "$tmp_dir" "$stage"
                  if [ -e "$backup" ] && [ ! -e "$dest" ]; then mv "$backup" "$dest"; fi
                }
                trap cleanup EXIT
                archive="$tmp_dir/Ollama-darwin.zip"
                rm -rf "$stage" "$backup"
                echo ">>> Downloading Ollama for macOS into $dest" >&2
                curl --fail --show-error --location --progress-bar -o "$archive" "https://ollama.com/download/Ollama-darwin.zip"
                unzip -q "$archive" -d "$tmp_dir"
                mkdir -p "$stage"
                cp -R "$tmp_dir/Ollama.app/Contents/Resources/." "$stage/"
                chmod 755 "$stage/ollama"
                test -x "$stage/ollama"
                if [ -e "$dest" ]; then mv "$dest" "$backup"; fi
                if mv "$stage" "$dest"; then
                  rm -rf "$backup"
                else
                  if [ -e "$backup" ]; then mv "$backup" "$dest"; fi
                  exit 1
                fi
                "#,
            );
        return Ok(cmd);
    }

    #[cfg(target_os = "linux")]
    {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.kill_on_drop(true); // see Windows-branch comment above
        cmd.env("OPENHUMAN_OLLAMA_INSTALL_DIR", install_dir);
        cmd.arg("-lc")
            .arg(
                r#"
                set -eu
                for tool in curl tar uname rm mkdir mv; do
                  command -v "$tool" >/dev/null 2>&1 || { echo "missing required tool: $tool" >&2; exit 1; }
                done
                arch="$(uname -m)"
                case "$arch" in
                  x86_64) arch="amd64" ;;
                  aarch64|arm64) arch="arm64" ;;
                  *) echo "Unsupported architecture: $arch" >&2; exit 1 ;;
                esac
                dest="$OPENHUMAN_OLLAMA_INSTALL_DIR"
                stage="${dest}.installing.$$"
                backup="${dest}.backup.$$"
                cleanup() {
                  rm -rf "$stage"
                  if [ -e "$backup" ] && [ ! -e "$dest" ]; then mv "$backup" "$dest"; fi
                }
                trap cleanup EXIT
                archive_url="https://ollama.com/download/ollama-linux-${arch}.tar.zst"
                if ! command -v unzstd >/dev/null 2>&1; then
                  echo "missing required tool: unzstd (zstd package)" >&2
                  exit 1
                fi
                rm -rf "$stage" "$backup"
                mkdir -p "$stage"
                echo ">>> Downloading Ollama for Linux into $dest" >&2
                curl --fail --show-error --location --progress-bar "$archive_url" | tar --use-compress-program=unzstd -xf - -C "$stage"
                chmod 755 "$stage/bin/ollama"
                test -x "$stage/bin/ollama"
                if [ -e "$dest" ]; then mv "$dest" "$backup"; fi
                if mv "$stage" "$dest"; then
                  rm -rf "$backup"
                else
                  if [ -e "$backup" ]; then mv "$backup" "$dest"; fi
                  exit 1
                fi
                "#,
            );
        return Ok(cmd);
    }

    #[allow(unreachable_code)]
    Err(format!(
        "Unsupported platform for automatic Ollama install: {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    ))
}

/// Locate an existing Ollama executable from overrides and platform defaults.
pub fn find_system_ollama_binary() -> Option<PathBuf> {
    if let Some(from_env) = std::env::var("OLLAMA_BIN")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        let path = PathBuf::from(from_env);
        if is_executable_file(&path) {
            return Some(path);
        }
    }

    let binary_name = if cfg!(windows) {
        "ollama.exe"
    } else {
        "ollama"
    };
    if let Some(path_var) = std::env::var_os("PATH") {
        for entry in std::env::split_paths(&path_var) {
            let candidate = entry.join(binary_name);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }

    if cfg!(windows) {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            candidates.push(
                PathBuf::from(&local_app_data)
                    .join("Programs")
                    .join("Ollama")
                    .join("ollama.exe"),
            );
            candidates.push(
                PathBuf::from(&local_app_data)
                    .join("Ollama")
                    .join("ollama.exe"),
            );
        }
        if let Ok(program_files) = std::env::var("PROGRAMFILES") {
            candidates.push(
                PathBuf::from(&program_files)
                    .join("Ollama")
                    .join("ollama.exe"),
            );
        }
        for candidate in candidates {
            if is_executable_file(&candidate) {
                tracing::debug!(
                    "[local_ai] found system Ollama at common Windows path: {}",
                    candidate.display()
                );
                return Some(candidate);
            }
        }
    }

    if cfg!(target_os = "macos") {
        let mut candidates = vec![
            PathBuf::from("/usr/local/bin/ollama"),
            PathBuf::from("/opt/homebrew/bin/ollama"),
        ];
        // Ollama.app installed in /Applications or ~/Applications ships its
        // CLI binary inside the app bundle resources directory.
        let bundle_rel = std::path::Path::new("Applications")
            .join("Ollama.app")
            .join("Contents")
            .join("Resources")
            .join("ollama");
        candidates.push(PathBuf::from("/").join(&bundle_rel));
        if let Some(home) = std::env::var_os("HOME") {
            candidates.push(PathBuf::from(home).join(&bundle_rel));
        }
        for candidate in candidates {
            if is_executable_file(&candidate) {
                tracing::debug!(
                    "[local_ai] found system Ollama at macOS path: {}",
                    candidate.display()
                );
                return Some(candidate);
            }
        }
    }

    if cfg!(target_os = "linux") {
        let common = [
            PathBuf::from("/usr/local/bin/ollama"),
            PathBuf::from("/usr/bin/ollama"),
        ];
        for candidate in common {
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }

    None
}

#[cfg(unix)]
fn is_executable_file(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    std::fs::metadata(path)
        .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable_file(path: &std::path::Path) -> bool {
    path.is_file()
}

#[cfg(test)]
#[path = "install_test.rs"]
mod tests;
