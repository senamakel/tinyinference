//! Public tool-schema validation contracts.

use serde_json::json;
use tinyinference_llm::tool::{ToolCall, ToolSchema};

#[test]
fn invalid_provider_arguments_fail_even_with_permissive_schema() {
    let schema = ToolSchema::new("lookup", "lookup", json!({}));
    let call = ToolCall::invalid("call-1", "lookup", "{broken", "expected value");
    let error = schema.validate_call(&call).unwrap_err();
    assert!(error.to_string().contains("malformed arguments"));
}

#[test]
fn json_values_are_validated_against_the_structural_subset() {
    use tinyinference_llm::tool::validate_json_value;

    let schema = json!({
        "type": "object",
        "properties": {
            "score": { "type": "integer" },
            "tags": { "type": "array", "items": { "type": "string" } },
            "mode": { "enum": ["fast", "slow"] }
        },
        "required": ["score"],
        "additionalProperties": false
    });

    assert!(validate_json_value(&schema, &json!({"score": 3, "tags": ["a"]}), "v").is_ok());
    let cases = [
        (json!({"tags": []}), "v.score is required"),
        (json!({"score": "3"}), "v.score must be integer, got string"),
        (json!({"score": 1, "extra": 1}), "v.extra is not allowed"),
        (json!({"score": 1, "tags": [1]}), "v.tags[0] must be string, got integer"),
        (json!({"score": 1, "mode": "medium"}), "v.mode must be one of the declared enum values"),
    ];
    for (value, message) in cases {
        let error = validate_json_value(&schema, &value, "v").unwrap_err();
        assert!(error.to_string().contains(message), "{error} vs {message}");
    }

    let untyped = json!({"required": ["id"]});
    let error = validate_json_value(&untyped, &json!(5), "v").unwrap_err();
    assert!(error.to_string().contains("v must be an object with the declared fields, got integer"));

    let union = json!({"type": ["string", "null"]});
    assert!(validate_json_value(&union, &json!(null), "v").is_ok());
    let error = validate_json_value(&union, &json!(1), "v").unwrap_err();
    assert!(error.to_string().contains("v must be one of string, null, got integer"));

    assert!(validate_json_value(&json!({}), &json!(1), "v").is_ok());
    assert!(validate_json_value(&json!({"type": "uuid"}), &json!(1), "v").is_ok());
}

#[test]
fn canonical_values_sort_keys_at_every_depth() {
    use tinyinference_llm::cache::canonical_value;

    let value = canonical_value(json!({"b": [{"z": 1, "a": 2}], "a": {"y": 1, "x": 2}}));

    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        r#"{"a":{"x":2,"y":1},"b":[{"a":2,"z":1}]}"#
    );
}

#[test]
fn replacing_text_blocks_keeps_other_blocks_in_place() {
    use tinyinference_llm::ContentBlock;
    use tinyinference_llm::prompt_tools::replace_text_blocks;

    let content = vec![
        ContentBlock::Text("raw <tool>".into()),
        ContentBlock::Text("more".into()),
    ];
    assert_eq!(
        replace_text_blocks(content, "clean".into()),
        vec![ContentBlock::Text("clean".into())]
    );
    assert_eq!(replace_text_blocks(vec![], "only".into()), vec![ContentBlock::Text("only".into())]);
    assert_eq!(replace_text_blocks(vec![ContentBlock::Text("x".into())], String::new()), vec![]);
}
