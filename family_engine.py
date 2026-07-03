# -*- coding: utf-8 -*-
"""
记忆家族（graph 雏形）：向量相似的记忆自动聚成"家族"，家族够大时生成/增量更新摘要。

- 家族表存在 embeddings.db 里（和向量同库）。
- 聚族纯数学（余弦 vs 家族质心），不花钱；摘要走现成的 DeepSeek 客户端（借 dehydrator 的）。
- 挂在 EmbeddingEngine 的 on_stored/on_deleted 回调上：所有入库/删除路径一处接住。
- 阈值可调：OMBRE_FAMILY_THRESHOLD（默认 0.70）；触发摘要的成员数 OMBRE_FAMILY_SUMMARY_MIN（默认 5）。
"""

import os
import json
import uuid
import sqlite3
import logging
import datetime

logger = logging.getLogger("ombre.family")

THRESHOLD = float(os.environ.get("OMBRE_FAMILY_THRESHOLD", "0.82") or 0.82)  # 实测 Gemini 向量上 0.70 会聚成大杂烩，0.82 甜点位
SUMMARY_MIN = int(os.environ.get("OMBRE_FAMILY_SUMMARY_MIN", "5") or 5)
REBUILD_SUMMARY_CAP = int(os.environ.get("OMBRE_FAMILY_REBUILD_SUMMARY_CAP", "15") or 15)

_db_path = None
_bucket_loader = None   # async fn(bucket_id) -> {"content":..., "metadata":...} | None
_dehydrator = None      # 借用它的 AsyncOpenAI client + model


def init(config, bucket_loader, dehydrator):
    global _db_path, _bucket_loader, _dehydrator
    _db_path = os.path.join(config["buckets_dir"], "embeddings.db")
    _bucket_loader = bucket_loader
    _dehydrator = dehydrator
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS families (
                id TEXT PRIMARY KEY,
                name TEXT,
                summary TEXT,
                member_ids TEXT NOT NULL,
                centroid TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                dirty INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)
    logger.info(f"Family engine ready | threshold={THRESHOLD} summary_min={SUMMARY_MIN}")


def _conn():
    return sqlite3.connect(_db_path)


def _now():
    return datetime.datetime.utcnow().isoformat()


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _rows():
    with _conn() as c:
        cur = c.execute("SELECT id, name, summary, member_ids, centroid, member_count, dirty, updated_at FROM families")
        out = []
        for r in cur.fetchall():
            out.append({
                "id": r[0], "name": r[1], "summary": r[2],
                "member_ids": json.loads(r[3]), "centroid": json.loads(r[4]),
                "member_count": r[5], "dirty": bool(r[6]), "updated_at": r[7],
            })
        return out


def _save(fam):
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO families (id, name, summary, member_ids, centroid, member_count, dirty, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (fam["id"], fam.get("name"), fam.get("summary"),
             json.dumps(fam["member_ids"], ensure_ascii=False), json.dumps(fam["centroid"]),
             fam["member_count"], 1 if fam.get("dirty") else 0, _now()),
        )


def _delete_family(fid):
    with _conn() as c:
        c.execute("DELETE FROM families WHERE id = ?", (fid,))


def _find_membership(bucket_id):
    for fam in _rows():
        if bucket_id in fam["member_ids"]:
            return fam
    return None


def _assign_to_families(bucket_id, embedding, families):
    """纯逻辑：把一条向量分给最像的家族（≥阈值）或自立门户。families 就地修改，返回 (family, joined)。"""
    best, best_sim = None, -1.0
    for fam in families:
        sim = _cos(embedding, fam["centroid"])
        if sim > best_sim:
            best, best_sim = fam, sim
    if best is not None and best_sim >= THRESHOLD:
        n = best["member_count"]
        best["centroid"] = [(c * n + e) / (n + 1) for c, e in zip(best["centroid"], embedding)]
        best["member_ids"].append(bucket_id)
        best["member_count"] = n + 1
        if best["member_count"] >= SUMMARY_MIN:
            best["dirty"] = True
        return best, True
    fam = {
        "id": uuid.uuid4().hex[:10], "name": None, "summary": None,
        "member_ids": [bucket_id], "centroid": list(embedding),
        "member_count": 1, "dirty": False,
    }
    families.append(fam)
    return fam, False


async def on_stored(bucket_id, embedding, content):
    """EmbeddingEngine 存完一条向量后回调。"""
    try:
        existing = _find_membership(bucket_id)
        if existing:
            # 内容更新：标脏，摘要下次刷新；不重新分家（保持家族稳定）
            if existing["member_count"] >= SUMMARY_MIN:
                existing["dirty"] = True
                _save(existing)
            return
        families = _rows()
        fam, _joined = _assign_to_families(bucket_id, embedding, families)
        _save(fam)
        if fam.get("dirty"):
            await _refresh_summary(fam)
    except Exception as e:
        logger.warning(f"Family on_stored failed for {bucket_id}: {e}")


def on_deleted(bucket_id):
    """EmbeddingEngine 删除向量时回调（同步、无 LLM）。"""
    try:
        fam = _find_membership(bucket_id)
        if not fam:
            return
        fam["member_ids"] = [m for m in fam["member_ids"] if m != bucket_id]
        fam["member_count"] = len(fam["member_ids"])
        if fam["member_count"] == 0:
            _delete_family(fam["id"])
            return
        # 质心从剩余成员重算
        vecs = _load_vectors(fam["member_ids"])
        if vecs:
            dims = len(next(iter(vecs.values())))
            centroid = [0.0] * dims
            for v in vecs.values():
                centroid = [c + x for c, x in zip(centroid, v)]
            fam["centroid"] = [c / len(vecs) for c in centroid]
        fam["dirty"] = fam["member_count"] >= SUMMARY_MIN
        _save(fam)
    except Exception as e:
        logger.warning(f"Family on_deleted failed for {bucket_id}: {e}")


def _load_vectors(bucket_ids=None):
    with _conn() as c:
        if bucket_ids is None:
            cur = c.execute("SELECT bucket_id, embedding FROM embeddings")
        else:
            marks = ",".join("?" for _ in bucket_ids)
            cur = c.execute(f"SELECT bucket_id, embedding FROM embeddings WHERE bucket_id IN ({marks})", list(bucket_ids))
        return {r[0]: json.loads(r[1]) for r in cur.fetchall()}


async def _refresh_summary(fam):
    """生成/增量改写家族摘要。第一行=主题名，其后=摘要正文。"""
    if _dehydrator is None or not getattr(_dehydrator, "api_available", False):
        return
    try:
        members = []
        for mid in fam["member_ids"][-12:]:
            b = await _bucket_loader(mid)
            if b:
                name = b["metadata"].get("name", "")
                content = (b.get("content") or "")[:400]
                members.append(f"《{name}》{content}")
        if not members:
            return
        if fam.get("summary"):
            prompt = (
                f"这是一个记忆家族的旧摘要：\n{fam['summary']}\n\n"
                f"家族新并入/更新了这些记忆：\n" + "\n---\n".join(members[-4:]) +
                "\n\n请增量改写摘要：保留旧摘要中仍然成立的部分，融入新信息。"
                "第一行输出家族主题名（不超过10个字），空一行后输出摘要正文（3-6句，第三人称，把这些记忆讲成一段连贯的认知）。"
            )
        else:
            prompt = (
                "以下是语义上相近的一组记忆条目：\n" + "\n---\n".join(members) +
                "\n\n请把它们概括成一份家族摘要。第一行输出家族主题名（不超过10个字），"
                "空一行后输出摘要正文（3-6句，第三人称，把这些记忆讲成一段连贯的认知，突出共同主题和演变）。"
            )
        resp = await _dehydrator.client.chat.completions.create(
            model=_dehydrator.model,
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return
        lines = text.split("\n", 1)
        fam["name"] = lines[0].strip().strip("《》#* ")[:24] or fam.get("name")
        fam["summary"] = (lines[1].strip() if len(lines) > 1 else text)
        fam["dirty"] = False
        _save(fam)
        logger.info(f"Family summary refreshed: {fam['name']} ({fam['member_count']} members)")
    except Exception as e:
        logger.warning(f"Family summary refresh failed: {e}")


def families_for(bucket_ids):
    """bucket_id 集合 → {family_id: {family..., 'hits': [命中的成员id]}}，只回有摘要或成员≥SUMMARY_MIN 的家族。"""
    out = {}
    ids = set(bucket_ids)
    for fam in _rows():
        hits = [m for m in fam["member_ids"] if m in ids]
        if hits and (fam.get("summary") or fam["member_count"] >= SUMMARY_MIN):
            fam["hits"] = hits
            out[fam["id"]] = fam
    return out


async def rebuild(dry_run=True, bucket_meta_loader=None, threshold=None):
    """全量重聚：读全部向量，按创建时间顺序增量聚族。dry_run 只出预览。threshold 可临时覆盖。"""
    global THRESHOLD
    if threshold:
        THRESHOLD = float(threshold)
    vectors = _load_vectors()
    if not vectors:
        return {"ok": False, "note": "没有任何向量，无法聚族。"}
    order = list(vectors.keys())
    names = {}
    if bucket_meta_loader:
        metas = await bucket_meta_loader()
        created = {m["id"]: m["metadata"].get("created", "") for m in metas}
        names = {m["id"]: m["metadata"].get("name", m["id"]) for m in metas}
        order.sort(key=lambda i: created.get(i, ""))
    families = []
    for bid in order:
        _assign_to_families(bid, vectors[bid], families)
    families.sort(key=lambda f: -f["member_count"])
    preview = [{
        "size": f["member_count"],
        "members": [names.get(m, m) for m in f["member_ids"][:8]] + (["…"] if f["member_count"] > 8 else []),
    } for f in families if f["member_count"] >= 2]
    singles = sum(1 for f in families if f["member_count"] == 1)
    if dry_run:
        return {"ok": True, "dryRun": True, "threshold": THRESHOLD,
                "familiesOf2Plus": len(preview), "singles": singles, "preview": preview[:40]}
    with _conn() as c:
        c.execute("DELETE FROM families")
    summarized = 0
    for fam in families:
        fam["dirty"] = fam["member_count"] >= SUMMARY_MIN
        _save(fam)
    for fam in families:
        if fam["dirty"] and summarized < REBUILD_SUMMARY_CAP:
            await _refresh_summary(fam)
            summarized += 1
    return {"ok": True, "dryRun": False, "threshold": THRESHOLD,
            "familiesOf2Plus": len(preview), "singles": singles,
            "summarized": summarized, "note": f"摘要生成了 {summarized} 个（上限 {REBUILD_SUMMARY_CAP}），其余标脏后续增量补。"}


def status():
    fams = _rows()
    return {
        "threshold": THRESHOLD,
        "summaryMin": SUMMARY_MIN,
        "families": len(fams),
        "withSummary": sum(1 for f in fams if f.get("summary")),
        "dirty": sum(1 for f in fams if f.get("dirty")),
        "largest": max((f["member_count"] for f in fams), default=0),
    }
