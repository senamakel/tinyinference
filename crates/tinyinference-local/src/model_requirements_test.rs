use super::*;

const REQUIRED: u64 = 8_192;

#[test]
fn at_or_above_minimum_is_accepted() {
    let exact = evaluate_context(Some(8_192), REQUIRED);
    assert!(exact.is_accepted());
    assert_eq!(
        exact,
        ContextEligibility::Ok {
            context_length: 8_192
        }
    );

    let above = evaluate_context(Some(32_768), REQUIRED);
    assert!(above.is_accepted());
    assert!(!above.is_rejected());
}

#[test]
fn below_minimum_is_rejected_with_required_floor() {
    let verdict = evaluate_context(Some(2_048), REQUIRED);
    assert!(verdict.is_rejected());
    assert!(!verdict.is_accepted());
    assert_eq!(
        verdict,
        ContextEligibility::BelowMinimum {
            context_length: 2_048,
            required: 8_192,
        }
    );
}

#[test]
fn unknown_context_is_neither_accepted_nor_rejected() {
    let verdict = evaluate_context(None, REQUIRED);
    assert!(!verdict.is_accepted());
    assert!(!verdict.is_rejected());
    assert_eq!(verdict, ContextEligibility::Unknown { required: 8_192 });
}

#[test]
fn eligibility_serializes_tagged() {
    let json = serde_json::to_value(evaluate_context(Some(4_096), REQUIRED)).unwrap();
    assert_eq!(json["status"], "below_minimum");
    assert_eq!(json["context_length"], 4_096);
    assert_eq!(json["required"], 8_192);

    let ok = serde_json::to_value(evaluate_context(Some(8_192), REQUIRED)).unwrap();
    assert_eq!(ok["status"], "ok");
    assert_eq!(ok["context_length"], 8_192);

    let unknown = serde_json::to_value(evaluate_context(None, REQUIRED)).unwrap();
    assert_eq!(unknown["status"], "unknown");
    assert_eq!(unknown["required"], 8_192);
}
