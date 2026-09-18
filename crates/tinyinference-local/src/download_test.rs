use super::*;

#[test]
fn part_path_appends_part_suffix() {
    let p = part_path(Path::new("/tmp/foo.bin"));
    assert_eq!(
        p.file_name().unwrap().to_string_lossy(),
        "foo.bin.part",
        "should append .part"
    );
}

#[test]
fn part_path_handles_no_extension() {
    let p = part_path(Path::new("/tmp/binaryname"));
    assert_eq!(p.file_name().unwrap().to_string_lossy(), "binaryname.part");
}

#[test]
fn voice_install_state_as_str_is_stable() {
    // The UI relies on the lowercase string form — guard against an
    // accidental rename breaking the wire contract.
    assert_eq!(VoiceInstallState::Missing.as_str(), "missing");
    assert_eq!(VoiceInstallState::Installing.as_str(), "installing");
    assert_eq!(VoiceInstallState::Installed.as_str(), "installed");
    assert_eq!(VoiceInstallState::Broken.as_str(), "broken");
    assert_eq!(VoiceInstallState::Error.as_str(), "error");
}

#[test]
fn read_status_defaults_to_missing_for_unseen_engine() {
    let unique = format!("test-engine-{}", uuid::Uuid::new_v4());
    let snapshot = read_status(&unique);
    assert_eq!(snapshot.state, VoiceInstallState::Missing);
    assert_eq!(snapshot.engine, unique);
    assert!(snapshot.progress.is_none());
}

#[test]
fn write_and_read_status_roundtrip() {
    let engine = format!("rt-{}", uuid::Uuid::new_v4());
    let status = VoiceInstallStatus {
        engine: engine.clone(),
        state: VoiceInstallState::Installing,
        progress: Some(42),
        downloaded_bytes: Some(1024),
        total_bytes: Some(2048),
        stage: Some("downloading model".to_string()),
        error_detail: None,
    };
    write_status(status);
    let got = read_status(&engine);
    assert_eq!(got.state, VoiceInstallState::Installing);
    assert_eq!(got.progress, Some(42));
    assert_eq!(got.stage.as_deref(), Some("downloading model"));
    // Clean up so the suite stays deterministic for parallel runs.
    reset_status(&engine);
}

#[test]
fn reset_status_returns_engine_to_missing() {
    let engine = format!("rs-{}", uuid::Uuid::new_v4());
    write_status(VoiceInstallStatus {
        engine: engine.clone(),
        state: VoiceInstallState::Installed,
        progress: None,
        downloaded_bytes: None,
        total_bytes: None,
        stage: None,
        error_detail: None,
    });
    reset_status(&engine);
    assert_eq!(read_status(&engine).state, VoiceInstallState::Missing);
}

// Engine ids used by the slot tests below. The slot map is keyed by
// `&'static str`, so we can't use uuid-suffixed names like the
// status-table tests; we use these dedicated keys instead. Production
// engine ids (ENGINE_PIPER) are deliberately avoided so tests can't
// deadlock against a real install in another test.
const TEST_SLOT_ENGINE_A: &str = "__test_slot_engine_a__";
const TEST_SLOT_ENGINE_B: &str = "__test_slot_engine_b__";

fn slot_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Best-effort drain of a test slot so the global set is clean across
/// runs. Tests that leave a slot held (e.g. by forgetting it) would
/// pollute subsequent runs in the same `cargo test` invocation.
fn drain_test_slot(engine: &'static str) {
    if let Ok(mut g) = in_flight().lock() {
        g.remove(engine);
    }
}

#[test]
fn try_acquire_install_slot_grants_then_blocks_then_releases() {
    let _test_guard = slot_test_lock();
    drain_test_slot(TEST_SLOT_ENGINE_A);

    // First caller gets the slot.
    let slot = try_acquire_install_slot(TEST_SLOT_ENGINE_A);
    assert!(slot.is_some(), "first acquire should succeed");

    // Concurrent caller is rejected while the first slot lives.
    let second = try_acquire_install_slot(TEST_SLOT_ENGINE_A);
    assert!(
        second.is_none(),
        "second acquire must be rejected while slot is held"
    );

    // Releasing the first slot reopens the door for a fresh caller.
    drop(slot);
    let third = try_acquire_install_slot(TEST_SLOT_ENGINE_A);
    assert!(
        third.is_some(),
        "acquire after drop should succeed (Drop must release)"
    );

    drop(third);
    drain_test_slot(TEST_SLOT_ENGINE_A);
}

#[test]
fn install_slot_keys_are_independent_per_engine() {
    let _test_guard = slot_test_lock();
    drain_test_slot(TEST_SLOT_ENGINE_A);
    drain_test_slot(TEST_SLOT_ENGINE_B);

    let slot_a = try_acquire_install_slot(TEST_SLOT_ENGINE_A).expect("A acquire");
    // Holding the A slot must not block the B slot — installs for
    // different engines run independently.
    let slot_b =
        try_acquire_install_slot(TEST_SLOT_ENGINE_B).expect("B acquire must succeed independently");
    // Acquiring A again must still fail though.
    assert!(
        try_acquire_install_slot(TEST_SLOT_ENGINE_A).is_none(),
        "A slot is still held"
    );

    drop(slot_a);
    drop(slot_b);
    drain_test_slot(TEST_SLOT_ENGINE_A);
    drain_test_slot(TEST_SLOT_ENGINE_B);
}

/// Race-path test — the whole reason the slot exists. Spawn many
/// concurrent tasks that all try to acquire the slot for the same
/// engine; exactly one must succeed, all others must be rejected.
/// This is the unit-level analogue of "two RPC handlers fire at the
/// same time and both spawn install tasks" — the bug CodeRabbit
/// flagged on PR #1755.
#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn concurrent_install_slot_acquire_grants_exactly_one() {
    let _test_guard = slot_test_lock();
    drain_test_slot(TEST_SLOT_ENGINE_A);

    // 32 concurrent acquirers — high enough to make a non-atomic
    // implementation almost certainly fail, low enough to stay
    // hermetic and fast.
    const N: usize = 32;
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        handles.push(tokio::spawn(async move {
            try_acquire_install_slot(TEST_SLOT_ENGINE_A)
        }));
    }
    let mut winners = 0usize;
    let mut losers = 0usize;
    // Collect outcomes *before* any slot is dropped — winners must
    // hold their slot alive past every other acquirer's attempt.
    let mut held = Vec::new();
    for h in handles {
        match h.await.expect("task panicked") {
            Some(slot) => {
                winners += 1;
                held.push(slot);
            }
            None => losers += 1,
        }
    }
    assert_eq!(
        winners, 1,
        "exactly one concurrent acquirer must win (got {winners})"
    );
    assert_eq!(losers, N - 1, "all other acquirers must lose");

    // Now drop the winner — the slot becomes available again.
    held.clear();
    let after = try_acquire_install_slot(TEST_SLOT_ENGINE_A);
    assert!(
        after.is_some(),
        "slot must be reacquirable once the winner drops"
    );
    drop(after);
    drain_test_slot(TEST_SLOT_ENGINE_A);
}

#[tokio::test]
async fn synthetic_stream_rejects_undersized_payload() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("never.bin");
    let part = part_path(&dest);
    let stream =
        futures::stream::iter([Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"tiny"))]);
    let result = write_download_stream(
        stream,
        DownloadTarget {
            total: Some(4),
            part_path: &part,
            dest: &dest,
            expected_sha256: None,
            min_bytes: 10,
            log_prefix: "[voice-install:test]",
        },
        |_, _| {},
    )
    .await;
    assert!(
        matches!(result, Err(crate::Error::DownloadIntegrity(_))),
        "expected typed integrity error"
    );
    assert!(
        !part.exists(),
        "no part file should remain after validation failure"
    );
}

#[tokio::test]
async fn synthetic_stream_commits_validated_replacement() {
    let body = b"hello voice-install body";
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("hello.bin");
    tokio::fs::write(&dest, b"previous valid artifact")
        .await
        .unwrap();
    let part = part_path(&dest);
    let stream = futures::stream::iter([
        Ok::<_, std::io::Error>(bytes::Bytes::from_static(&body[..8])),
        Ok(bytes::Bytes::from_static(&body[8..])),
    ]);
    let mut last_progress = (0u64, None);
    let result = write_download_stream(
        stream,
        DownloadTarget {
            total: Some(body.len() as u64),
            part_path: &part,
            dest: &dest,
            expected_sha256: None,
            min_bytes: 5,
            log_prefix: "[voice-install:test]",
        },
        |downloaded, total| {
            last_progress = (downloaded, total);
        },
    )
    .await;
    assert!(result.is_ok(), "download failed: {result:?}");
    let on_disk = tokio::fs::read(&dest).await.unwrap();
    assert_eq!(on_disk.as_slice(), body, "wrong bytes landed on disk");
    assert!(last_progress.0 > 0, "progress callback should fire");
    assert!(!part.exists(), "part file should be renamed away");
}
