import json

import pytest

import server


class FakeBucketManager:
    def __init__(self):
        self.bucket = {
            "id": "shared-memory",
            "content": "同一份共同记忆。",
            "metadata": {
                "agent_stances": [
                    {"actor": "Calder", "stance": "hold", "note": "还想想", "updated_at": "old"},
                ],
            },
        }

    async def get(self, bucket_id):
        return self.bucket if bucket_id == self.bucket["id"] else None

    async def update(self, bucket_id, **updates):
        if bucket_id != self.bucket["id"]:
            return False
        self.bucket["metadata"].update(updates)
        return True


@pytest.mark.asyncio
async def test_two_agents_keep_independent_stances_on_one_bucket(monkeypatch):
    manager = FakeBucketManager()
    monkeypatch.setattr(server, "bucket_mgr", manager)

    result = json.loads(await server.memory_stance("shared-memory", "珂洛", "claim", "这条我认。"))

    assert result["ok"] is True
    assert {item["actor"] for item in result["agent_stances"]} == {"珂洛", "Calder"}
    assert manager.bucket["content"] == "同一份共同记忆。"


@pytest.mark.asyncio
async def test_agent_can_clear_only_their_own_stance(monkeypatch):
    manager = FakeBucketManager()
    manager.bucket["metadata"]["agent_stances"].append(
        {"actor": "珂洛", "stance": "reject", "note": "", "updated_at": "old"}
    )
    monkeypatch.setattr(server, "bucket_mgr", manager)

    result = json.loads(await server.memory_stance("shared-memory", "Calder", "clear"))

    assert result["ok"] is True
    assert [item["actor"] for item in result["agent_stances"]] == ["珂洛"]


def test_recall_exposes_stances_without_rewriting_memory_text():
    line = server._agent_stance_recall_line({
        "agent_stances": [
            {"actor": "珂洛", "stance": "claim", "note": "这条我认。"},
            {"actor": "Calder", "stance": "hold", "note": ""},
        ],
    })

    assert line == "\n[页边表态: 珂洛=认同（这条我认。）；Calder=保留]"
