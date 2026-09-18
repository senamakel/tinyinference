use super::*;

#[cfg(unix)]
#[test]
fn executable_check_rejects_regular_file_without_execute_bits() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let binary = temp.path().join("ollama");
    std::fs::write(&binary, b"stub").unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(!is_executable_file(&binary));

    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(is_executable_file(&binary));
}

#[test]
fn build_install_command_on_supported_platform_returns_ok() {
    let directory = tempfile::tempdir().expect("temporary install directory");
    let result = build_install_command(directory.path());
    if cfg!(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows"
    )) {
        assert!(result.is_ok(), "supported platform: {result:?}");
    } else {
        assert!(result.is_err(), "unsupported platform");
    }
}
