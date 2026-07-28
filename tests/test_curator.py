from curator import aggregate_somatic_signals, duplicate_similarity, memory_fingerprint, normalize_curate_payload


def test_revision_is_always_a_candidate_and_evidence_is_scoped():
    payload = normalize_curate_payload({
        "session_id": "session-1",
        "source_message_ids": ["u1"],
        "memories": [{
            "title": "可能变化的偏好",
            "content": "这是一条等待 Claire 确认的新版本。",
            "operation": "revision",
            "status": "confirmed",
            "confidence": 0.96,
            "evidence_message_ids": ["u1", "invented"],
        }],
    })
    memory = payload["memories"][0]
    assert memory["status"] == "candidate"
    assert memory["evidence_message_ids"] == ["u1"]


def test_shared_home_provenance_is_normalized_without_splitting_access():
    payload = normalize_curate_payload({
        "session_id": "shared-room",
        "source_message_ids": ["u1", "a1"],
        "memories": [{
            "title": "共同记住的约定",
            "content": "Claire 和 Calder 约好周末一起整理照片。",
            "status": "confirmed",
            "confidence": 0.98,
            "evidence_message_ids": ["u1", "a1"],
            "evidence_quotes": [
                {"message_id": "u1", "quote": "周末一起整理照片。"},
                {"message_id": "invented", "quote": "不该被收下。"},
            ],
            "signed_by": ["Calder", "Calder"],
            "evidence_speakers": ["Claire", "Calder"],
            "participants": ["Claire", "Calder"],
            "curated_by": "sonnet_secretary",
            "source_surface": "kelo_home",
        }],
    })
    memory = payload["memories"][0]
    assert memory["memory_scope"] == "home_shared"
    assert memory["signed_by"] == ["Calder"]
    assert memory["evidence_speakers"] == ["Claire", "Calder"]
    assert memory["participants"] == ["Claire", "Calder"]
    assert memory["evidence_quotes"] == [{"message_id": "u1", "quote": "周末一起整理照片。"}]


def test_fingerprint_is_stable_for_retries():
    item = {
        "operation": "add",
        "evidence_message_ids": ["a2", "u1"],
        "title": "一条记忆",
        "content": "相同批次重跑时不能重复入库。",
        "supersedes": "",
    }
    assert memory_fingerprint("session-1", item) == memory_fingerprint("session-1", item)


def test_duplicate_similarity_compares_memory_content_not_recall_weight():
    item = {
        "title": "她喜欢茉莉花背景",
        "content": "Claire 明确说小家首页要保留透明的茉莉花背景。",
        "tags": ["小家", "茉莉花"],
    }
    same = {
        "content": "Claire 明确说小家首页要保留透明的茉莉花背景。",
        "metadata": {"name": "茉莉花背景要保留", "tags": ["小家", "茉莉花"]},
    }
    different = {
        "content": "Claire 今天想把工作室的抽签维度展开。",
        "metadata": {"name": "工作室抽签", "tags": ["工作室"]},
    }
    assert duplicate_similarity(item, same) >= 84
    assert duplicate_similarity(item, different) < 84


def test_somatic_batch_is_merged_and_capped_once():
    result = aggregate_somatic_signals([
        {"type": "affection", "weight": 0.8},
        {"type": "affection", "weight": 0.6},
        {"type": "mood:missing", "weight": 0.9},
        {"type": "not-a-real-signal", "weight": 1},
    ])
    assert [signal["type"] for signal in result["signals"]].count("affection") == 1
    numeric_pulses = [value for value in result["pulses"].values() if isinstance(value, (int, float))]
    assert numeric_pulses
    assert all(abs(value) <= 0.12 for value in numeric_pulses)
