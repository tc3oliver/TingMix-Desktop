use crate::*;

fn partial(seg: u64, rev: u32, text: &str) -> Event {
    Event::Partial(PartialEvent {
        segment_id: seg,
        revision: rev,
        text: text.into(),
        start: 0.0,
        end: 1.0,
        stable_prefix_len: 0,
    })
}

fn fin(seg: u64, rev: u32, text: &str) -> Event {
    Event::Final(FinalEvent {
        segment_id: seg,
        revision: rev,
        text: text.into(),
        start: 0.0,
        end: 1.0,
    })
}

#[test]
fn json_matches_python_protocol() {
    let p = serde_json::json!({"type":"partial","segment_id":12,"revision":3,"text":"今天我們 review 這個 API","start":12.5,"end":15.1,"stable_prefix_len":4});
    let ev: Event = serde_json::from_value(p.clone()).unwrap();
    assert_eq!(serde_json::to_value(&ev).unwrap(), p);
    let f = serde_json::json!({"type":"final","segment_id":12,"revision":4,"text":"x","start":12.5,"end":15.1});
    let ev: Event = serde_json::from_value(f.clone()).unwrap();
    assert!(matches!(ev, Event::Final(_)));
    assert_eq!(serde_json::to_value(&ev).unwrap(), f);
    assert!(serde_json::from_str::<Event>(r#"{"type":"bogus","segment_id":1}"#).is_err());
}

#[test]
fn transcript_numbers_revisions_and_closes_segments() {
    let mut t = Transcript::new();
    assert_eq!(t.partial(0, "a".into(), 0.0, 1.0, 0).unwrap().revision, 0);
    assert_eq!(t.partial(0, "ab".into(), 0.0, 1.0, 1).unwrap().revision, 1);
    assert_eq!(t.final_(0, "abc".into(), 0.0, 1.0).unwrap().revision, 2);
    assert_eq!(
        t.partial(0, "x".into(), 0.0, 1.0, 0),
        Err(TranscriptError::SegmentClosed(0))
    );
    assert_eq!(
        t.final_(0, "x".into(), 0.0, 1.0),
        Err(TranscriptError::SegmentClosed(0))
    );
    assert_eq!(t.final_(1, "".into(), 1.0, 2.0).unwrap().revision, 0);
    assert_eq!(t.finals().len(), 2);
}

#[test]
fn transcript_text_joins_cjk_without_spaces() {
    let mut t = Transcript::new();
    t.final_(0, "今天".into(), 0.0, 1.0).unwrap();
    t.final_(1, "很好".into(), 1.0, 2.0).unwrap();
    assert_eq!(t.text(), "今天很好");
    t.final_(2, "good morning".into(), 2.0, 3.0).unwrap();
    assert_eq!(t.text(), "今天 很好 good morning");
}

#[test]
fn reducer_rejects_stale_duplicate_and_late_partials() {
    let mut r = Reducer::new();
    r.apply(partial(0, 0, "a")).unwrap();
    r.apply(partial(0, 2, "abc")).unwrap();
    assert_eq!(r.apply(partial(0, 1, "ab")), Err(Rejected::StaleRevision));
    assert_eq!(r.apply(partial(0, 2, "abc")), Err(Rejected::StaleRevision));
    assert_eq!(r.apply(fin(0, 2, "abc")), Err(Rejected::StaleRevision));
    assert_eq!(r.open().next().unwrap().text, "abc");
    r.apply(fin(0, 3, "abcd")).unwrap();
    assert_eq!(r.apply(partial(0, 9, "zzz")), Err(Rejected::SegmentClosed));
    assert_eq!(r.apply(fin(0, 4, "zzz")), Err(Rejected::SegmentClosed));
    assert_eq!(r.open().count(), 0);
    assert_eq!(r.finals().next().unwrap().text, "abcd");
}

#[test]
fn reducer_accepts_final_without_partials_and_orders_by_segment() {
    let mut r = Reducer::new();
    r.apply(fin(1, 0, "second")).unwrap();
    r.apply(fin(0, 0, "first")).unwrap();
    let texts: Vec<_> = r.finals().map(|f| f.text.as_str()).collect();
    assert_eq!(texts, ["first", "second"]);
}

#[test]
fn snapshot_restores_state() {
    let mut r = Reducer::new();
    r.apply(fin(0, 1, "done")).unwrap();
    r.apply(partial(1, 2, "open")).unwrap();
    let json = serde_json::to_string(&r.snapshot()).unwrap();
    let mut r2 = Reducer::from_snapshot(serde_json::from_str(&json).unwrap());
    assert_eq!(r2.snapshot(), r.snapshot());
    assert_eq!(
        r2.apply(partial(0, 5, "late")),
        Err(Rejected::SegmentClosed)
    );
    assert_eq!(r2.apply(partial(1, 2, "dup")), Err(Rejected::StaleRevision));
    r2.apply(partial(1, 3, "open!")).unwrap();
}
