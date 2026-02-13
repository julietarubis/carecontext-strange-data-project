# src/rag.py
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedItem:
    row_index: int
    score: float
    text: str
    meta: Dict[str, Any]


def time_bucket(t: str) -> str:
    """Convert 'HH:MM' -> morning/afternoon/evening/night."""
    hh = int(str(t).split(":")[0])
    if 5 <= hh < 12:
        return "morning"
    if 12 <= hh < 17:
        return "afternoon"
    if 17 <= hh < 22:
        return "evening"
    return "night"


def _get(row: pd.Series, *keys: str, default: str = "") -> str:
    """Return the first existing column value among keys."""
    for k in keys:
        if k in row and pd.notna(row[k]):
            return str(row[k]).strip()
    return default


def build_snapshot(row: pd.Series) -> Tuple[str, Dict[str, Any]]:
    """
    Build a consistent text snapshot + metadata from one CSV row.
    Works with either column naming style:
    - (resident_id, name, date, time, ...) or
    - (resident_id, resident_name, event_date, event_time, ...)
    """
    resident_id = _get(row, "resident_id", default="UNKNOWN")
    resident_name = _get(row, "name", "resident_name", default="")
    date = _get(row, "date", "event_date", default="")
    time = _get(row, "time", "event_time", default="")
    location = _get(row, "location", default="")
    activity = _get(row, "activity", default="")
    mobility_aid = _get(row, "mobility_aid", "walking_aid", default="")
    supervision = _get(row, "supervision", "assist_level", default="")
    staff_present = _get(row, "staff_present", default="")
    staff_action = _get(row, "staff_action", default="")
    notes = _get(row, "notes", default="")
    fell = _get(row, "fell", default="").lower()

    tb = time_bucket(time) if time else "unknown"
    fell_label = "FALL" if fell in ("yes", "y", "true", "1") else "NO_FALL"

    meta = {
        "resident_id": resident_id,
        "resident_name": resident_name,
        "date": date,
        "time": time,
        "time_bucket": tb,
        "location": location,
        "activity": activity,
        "mobility_aid": mobility_aid,
        "supervision": supervision,
        "staff_present": staff_present,
        "staff_action": staff_action,
        "fell": fell_label,
    }

    # This text is what we embed/search over
    text = (
        f"{fell_label}. "
        f"Resident {resident_id} {('('+resident_name+')') if resident_name else ''} "
        f"at {time} ({tb}) on {date}. "
        f"Location: {location}. Activity: {activity}. "
        f"Mobility aid: {mobility_aid}. Supervision/assist: {supervision}. "
        f"Staff present: {staff_present}. Staff action: {staff_action}. "
        f"Notes: {notes}."
    )
    return text, meta


class CareContextRAG:
    """
    Lightweight offline RAG:
    - TF-IDF embeddings (no API required)
    - cosine similarity retrieval
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []

        for _, row in self.df.iterrows():
            text, meta = build_snapshot(row)
            self.texts.append(text)
            self.metas.append(meta)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, k: int = 3) -> List[RetrievedItem]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]

        results: List[RetrievedItem] = []
        for i in top_idx:
            results.append(
                RetrievedItem(
                    row_index=int(i),
                    score=float(sims[i]),
                    text=self.texts[i],
                    meta=self.metas[i],
                )
            )
        return results


def baseline_llm_like_answer(situation: str) -> str:
    """
    Offline baseline response: generic advice (simulates an LLM without your data).
    """
    return (
        "BASELINE (no care-context data):\n"
        "- Ensure supervision during mobilisation.\n"
        "- Ensure walking aid is appropriate and within reach.\n"
        "- Keep the environment safe (lighting, clutter-free, dry floors).\n"
        "- Encourage slow movement and use of call bell.\n"
    )


def enhanced_llm_like_answer(situation: str, retrieved: List[RetrievedItem]) -> str:
    """
    Offline enhanced response: uses retrieved evidence to produce specific actions.
    """
    evidence = []
    risk_reasons = set()

    for item in retrieved:
        m = item.meta
        evidence.append(
            f"- {m['date']} {m['time']} ({m['time_bucket']}), {m['location']}, "
            f"{m['activity']}, supervision={m['supervision']}, staff_action={m['staff_action']}, "
            f"result={m['fell']}"
        )

        if m["fell"] == "FALL":
            sup = (m.get("supervision") or "").lower()
            act = (m.get("activity") or "").lower()
            staff_action = (m.get("staff_action") or "").lower()

            if "alone" in sup or "unsuper" in sup:
                risk_reasons.add("unsupervised / alone mobilisation")
            if "left" in staff_action or "left" in sup:
                risk_reasons.add("resident left standing during a transfer/transition")
            if "toilet" in act:
                risk_reasons.add("toileting-related mobility risk")
            if m.get("time_bucket") in ("evening", "night"):
                risk_reasons.add("higher risk in evening/night (fatigue/low visibility)")

    if not risk_reasons:
        risk_reasons = {"similar high-risk context observed in historical incidents"}

    return (
        "ENHANCED (with CareContext retrieval):\n"
        f"Current situation:\n{situation}\n\n"
        "Retrieved similar incidents:\n"
        + "\n".join(evidence)
        + "\n\n"
        "Risk assessment:\n"
        f"- Higher fall risk due to: {', '.join(sorted(risk_reasons))}.\n\n"
        "Recommended actions (specific):\n"
        "- Do not leave the resident standing while staff fetch equipment.\n"
        "- If a wheelchair is needed, bring it BEFORE standing/transfer.\n"
        "- Encourage seated waiting; keep the walking frame positioned safely.\n"
        "- Increase supervision during morning toileting and evening transitions.\n"
    )
