use super::*;

#[test]
fn parses_openai_and_codex_catalogs() {
    let openai = serde_json::json!({
        "object": "list",
        "data": [
            { "id": "m1", "owned_by": "openai", "context_length": 8192 },
            { "id": "m2", "name": "Model 2", "pricing": {
                "inputPer1M": 1.25, "outputPer1M": 3.5
            }},
            { "id": "" },
        ],
    });
    let models = parse_models_response(&openai).expect("OpenAI catalog");
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].owned_by.as_deref(), Some("openai"));
    assert_eq!(models[0].context_window, Some(8192));
    assert_eq!(models[1].display_name.as_deref(), Some("Model 2"));
    assert_eq!(models[1].input_per_1m, Some(1.25));
    assert_eq!(models[1].output_per_1m, Some(3.5));

    let codex = serde_json::json!({
        "models": [
            { "slug": "gpt-5.5", "owned_by_organization": "openai", "max_context_window": 272000 },
            "gpt-5.4",
        ],
    });
    let models = parse_models_response(&codex).expect("Codex catalog");
    assert_eq!(models[0].id, "gpt-5.5");
    assert_eq!(models[0].context_window, Some(272000));
    assert_eq!(models[1].id, "gpt-5.4");
}

#[test]
fn distinguishes_missing_wrong_type_and_non_object_envelopes() {
    let err = parse_models_response(&serde_json::json!({ "items": [] }))
        .expect_err("missing catalog field");
    assert!(err.contains("missing `data` or `models` field"));

    for (kind, value) in [
        ("object", serde_json::json!({"message": "boom"})),
        ("string", serde_json::json!("models")),
        ("bool", serde_json::json!(true)),
        ("number", serde_json::json!(42)),
    ] {
        let err = parse_models_response(&serde_json::json!({ "data": value }))
            .expect_err("wrong catalog field type");
        assert!(err.contains(kind), "{err}");
    }

    for value in [
        serde_json::json!([]),
        serde_json::json!("body"),
        serde_json::Value::Null,
    ] {
        let err = parse_models_response(&value).expect_err("non-object response");
        assert!(err.contains("not a JSON object"));
    }
}

#[test]
fn null_is_empty_only_for_success_envelopes() {
    for body in [
        serde_json::json!({ "object": "list", "data": null }),
        serde_json::json!({ "models": null }),
    ] {
        assert!(
            parse_models_response(&body)
                .expect("empty catalog")
                .is_empty()
        );
    }

    let err = parse_models_response(&serde_json::json!({ "object": "error", "data": null }))
        .expect_err("error envelope");
    assert!(err.contains(r#""object" = "error""#));
}

#[test]
fn codex_hints_are_merged_without_duplicates() {
    let mut models = vec![ModelInfo {
        id: "gpt-5.4".into(),
        owned_by: None,
        context_window: None,
        display_name: None,
        input_per_1m: None,
        output_per_1m: None,
    }];
    merge_openai_codex_model_hints(&mut models);
    let ids = models
        .iter()
        .map(|model| model.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        ["gpt-5.4", "gpt-5.5", "gpt-5.3-codex-spark", "gpt-5.3-codex"]
    );
}

#[test]
fn raw_items_support_only_array_envelopes() {
    assert_eq!(
        model_items_from_body(&serde_json::json!({ "data": ["m1"] })),
        Some(vec![serde_json::json!("m1")])
    );
    assert_eq!(
        model_items_from_body(&serde_json::json!({ "models": ["m2"] })),
        Some(vec![serde_json::json!("m2")])
    );
    assert_eq!(
        model_items_from_body(&serde_json::json!({ "data": null })),
        None
    );
}
