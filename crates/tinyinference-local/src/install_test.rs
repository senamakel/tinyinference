use super::*;

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
