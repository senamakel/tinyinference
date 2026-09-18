use super::*;
use tempfile::TempDir;

fn tmp_marker_path() -> (TempDir, std::path::PathBuf) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("local-ai").join("ollama.spawn");
    (tmp, path)
}

#[test]
fn marker_round_trips_through_disk() {
    let (_tmp, path) = tmp_marker_path();
    let m = OllamaSpawnMarker {
        pid: 4242,
        started_at_unix: 1_700_000_000,
        binary_path: "C:\\fake\\ollama.exe".to_string(),
        openhuman_pid: 9001,
    };

    write_marker_at(&path, &m).expect("write marker");
    let loaded = read_marker_at(&path).expect("read marker");
    assert_eq!(loaded, m);

    clear_marker_at(&path);
    assert!(
        read_marker_at(&path).is_none(),
        "marker must be gone after clear"
    );
}

#[test]
fn read_marker_returns_none_when_file_missing() {
    let (_tmp, path) = tmp_marker_path();
    assert!(read_marker_at(&path).is_none());
}

#[test]
fn read_marker_returns_none_on_corrupt_json() {
    let (_tmp, path) = tmp_marker_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"{ not valid json").unwrap();

    assert!(
        read_marker_at(&path).is_none(),
        "corrupt marker must be treated as absent"
    );
}

#[test]
fn clear_marker_is_idempotent() {
    let (_tmp, path) = tmp_marker_path();
    clear_marker_at(&path);
    clear_marker_at(&path);
}

#[test]
fn write_marker_creates_missing_parent_dir() {
    let (_tmp, path) = tmp_marker_path();
    // path.parent() does NOT exist yet — write should create it.
    assert!(!path.parent().unwrap().exists());
    let m = OllamaSpawnMarker::new(1234, std::path::Path::new("ollama"));
    write_marker_at(&path, &m).expect("write");
    assert!(path.exists());
}

#[test]
fn write_marker_replaces_an_existing_marker() {
    let (_tmp, path) = tmp_marker_path();
    let first = OllamaSpawnMarker::new(1234, std::path::Path::new("ollama-old"));
    let second = OllamaSpawnMarker::new(5678, std::path::Path::new("ollama-new"));
    write_marker_at(&path, &first).expect("write first marker");
    write_marker_at(&path, &second).expect("replace marker");
    assert_eq!(read_marker_at(&path), Some(second));
}

#[test]
fn new_marker_captures_current_process_id() {
    let m = OllamaSpawnMarker::new(4242, std::path::Path::new("ollama"));
    assert_eq!(m.openhuman_pid, std::process::id());
    assert_eq!(m.pid, 4242);
    assert_eq!(m.binary_path, "ollama");
}

#[test]
fn pid_is_alive_recognises_self() {
    let me = std::process::id();
    assert!(
        pid_is_alive(me),
        "current process PID {me} should be reported alive"
    );
}
