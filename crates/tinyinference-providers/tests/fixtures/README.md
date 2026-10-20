# OAuth pending-secret compatibility fixture

`oauth-pending-v041.json` is the exact UTF-8 byte sequence emitted by the
TinyInference `v0.4.1` `PendingOAuth` serializer. The annotated generator test
was injected into an archive of that release tag and ran against the released
crate; it initialized the same four fields as the fixture, with
`created_at = 1_800_000_000`. The tag resolves to
`fc68596741940607d3453b4081021d8f5bac40f7`.

To regenerate the bytes, archive `v0.4.1`, add the following test at the end
of `crates/tinyinference-providers/src/oauth_tests.rs`, then run
`cargo test -p tinyinference-providers emit_v041_pending_oauth_bytes_for_compat_evidence -- --nocapture`:

```rust
#[test]
fn emit_v041_pending_oauth_bytes_for_compat_evidence() {
    let pending = PendingOAuth {
        state: "expected".into(),
        verifier: "released-verifier".into(),
        redirect_uri: "http://127.0.0.1:1455/cb".into(),
        created_at: 1_800_000_000,
    };
    println!("{}", String::from_utf8(serde_json::to_vec(&pending).unwrap()).unwrap());
}
```

The compatibility test substitutes only the fixture's timestamp with the
current time before storing the bytes, so the released serialized shape is
unchanged while the normal OAuth expiry check remains active.
