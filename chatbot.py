# chatbot.py
import json
import sqlite3
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import requests
import feedparser

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ChatbotBase import ChatbotBase

DEFAULT_BASE_MODEL = "google/flan-t5-small"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean_text(s: str) -> str:
    s = re.sub(r"<[^<]+?>", "", s or "")
    s = re.sub(r"&nbsp;+", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _truncate(s: Optional[str], max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _strip_bullets(text: str) -> str:
    """Remove bullet lines + obvious headline-ish clutter from saved context."""
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("-", "•", "*")):
            continue
        if "http://" in s or "https://" in s:
            continue
        # drop “Sources:” sections / “Why it matters:” labels from news blocks
        if s.lower().startswith(("sources", "why it matters")):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


class TechBot(ChatbotBase):
    """
    TechBot Pro (CPU-only) with:
      - LLM-based multi-intent + topic extraction (JSON)
      - Context memory (follow-ups)
      - Global tech news (Static RSS + Google News RSS search)
      - Personalization by knowledge level
      - Learning progress tracking
      - Straight answers by default (links only if asked)
    """

    def __init__(
        self,
        name: str = "TechBot Pro",
        db_path: str = "techbot.db",
        base_model: str = DEFAULT_BASE_MODEL,
        hf_cache_dir: Optional[str] = None,
        max_history_turns: int = 6,  # keep smaller to avoid loops
    ):
        super().__init__(name)
        self.db_path = db_path
        self.max_history_turns = max_history_turns

        self.keyword_expansion = self._load_keyword_expansion()
        self.learning_resources = self._load_learning_resources()

        self._init_database()
        self._init_llm(base_model=base_model, hf_cache_dir=hf_cache_dir)

        # cache: query -> (timestamp, items)
        self._news_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

        # follow-up helpers
        self._generic_topics = {
            "", "tech", "other", "update", "updates", "news", "latest", "recent",
            "how", "why", "what", "tell", "me", "about", "it", "this", "that", "more", "again"
        }

        print(f"✅ {self.name} initialized successfully!")
        print("   - Global tech news enabled (RSS + Google News RSS)")
        print("   - LLM multi-intent + topic extraction enabled (CPU)")
        print("   - Context memory enabled (follow-ups supported)")
        print("   - User profiles + progress tracking enabled")
        print("   - Straight answers (links only if asked)")
        print("   - Database initialized\n")

    # =========================
    # DB
    # =========================
    def _init_database(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                knowledge_level TEXT DEFAULT 'beginner',
                interests TEXT DEFAULT '[]',
                goals TEXT DEFAULT '[]',
                created_at TEXT
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_text TEXT,
                bot_text TEXT,
                ts TEXT
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                user_id TEXT PRIMARY KEY,
                summary TEXT DEFAULT '',
                last_updated TEXT
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic TEXT,
                resource TEXT,
                status TEXT DEFAULT 'suggested',
                ts TEXT
            )
            """
        )

        self.conn.commit()

    # =========================
    # PROFILE
    # =========================
    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.get_user_profile(user_id)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        self.cursor.execute(
            "SELECT user_id, knowledge_level, interests, goals, created_at FROM users WHERE user_id=?",
            (user_id,),
        )
        row = self.cursor.fetchone()
        if row:
            return {
                "user_id": row[0],
                "knowledge_level": row[1],
                "interests": json.loads(row[2] or "[]"),
                "goals": json.loads(row[3] or "[]"),
                "created_at": row[4],
            }
        return {
            "user_id": user_id,
            "knowledge_level": "beginner",
            "interests": [],
            "goals": [],
            "created_at": _now_iso(),
        }

    def set_user_profile(
        self,
        user_id: str,
        level: Optional[str] = None,
        interests: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
    ) -> str:
        current = self.get_user_profile(user_id)
        if level:
            current["knowledge_level"] = level
        if interests is not None:
            current["interests"] = interests
        if goals is not None:
            current["goals"] = goals

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO users (user_id, knowledge_level, interests, goals, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                current["knowledge_level"],
                json.dumps(current["interests"]),
                json.dumps(current["goals"]),
                current["created_at"],
            ),
        )
        self.conn.commit()
        self.context["current_user"] = user_id
        return f"✅ Profile updated for {user_id}"

    # =========================
    # MEMORY
    # =========================
    def _get_memory_summary(self, user_id: str) -> str:
        self.cursor.execute("SELECT summary FROM memories WHERE user_id=?", (user_id,))
        row = self.cursor.fetchone()
        return _truncate((row[0] if row and row[0] else ""), 350)

    def _set_memory_summary(self, user_id: str, summary: str):
        summary = _truncate(summary, 350)
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO memories (user_id, summary, last_updated)
            VALUES (?, ?, ?)
            """,
            (user_id, summary.strip(), _now_iso()),
        )
        self.conn.commit()

    def _get_recent_dialogue(self, user_id: str, limit: int = 6) -> List[Dict[str, str]]:
        self.cursor.execute(
            "SELECT user_text, bot_text FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = list(reversed(self.cursor.fetchall()))
        out = [{"user": r[0], "bot": r[1]} for r in rows]
        for t in out:
            t["user"] = _truncate(t["user"], 220)
            # IMPORTANT: strip bullets/headlines from context before feeding LLM
            t["bot"] = _truncate(_strip_bullets(t["bot"]), 260)
        return out

    def _update_memory_summary(self, user_id: str):
        if not getattr(self, "llm_ready", False):
            return
        turns = self._get_recent_dialogue(user_id, limit=self.max_history_turns)
        if not turns:
            return

        transcript = "\n".join([f"User: {t['user']}\nBot: {t['bot']}" for t in turns])
        transcript = _truncate(transcript, 900)

        prompt = f"""
Return 3-5 short bullets about the user's ongoing tech interests.
No repeating phrases. No generic "free app" text. No policy text.

Conversation:
{transcript}

Bullets:
""".strip()

        mem = self._llm_generate(prompt, max_new_tokens=120)
        mem = self._clean_llm_output(mem)
        if mem and not self._is_junk_output(mem):
            self._set_memory_summary(user_id, mem)

    # =========================
    # PROGRESS
    # =========================
    def track_resource(self, user_id: str, topic: str, resource: str, status: str = "suggested"):
        self.cursor.execute(
            "INSERT INTO progress (user_id, topic, resource, status, ts) VALUES (?, ?, ?, ?, ?)",
            (user_id, topic, resource, status, _now_iso()),
        )
        self.conn.commit()

    def mark_resource_done(self, user_id: str, resource_contains: str) -> str:
        k = (resource_contains or "").strip().lower()
        if not k:
            return "❌ Usage: done <part of resource name>"
        self.cursor.execute(
            "UPDATE progress SET status='done' WHERE user_id=? AND LOWER(resource) LIKE ?",
            (user_id, f"%{k}%"),
        )
        self.conn.commit()
        if self.cursor.rowcount == 0:
            return "⚠️ I couldn't find a suggested resource matching that."
        return f"✅ Marked {self.cursor.rowcount} resource(s) as done."

    def get_progress_report(self, user_id: str) -> str:
        self.cursor.execute(
            """
            SELECT topic, resource, status, ts
            FROM progress
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 20
            """,
            (user_id,),
        )
        rows = self.cursor.fetchall()
        if not rows:
            return "No progress tracked yet. Ask for resources and then mark them with: done <keyword>"

        lines = ["📈 Progress (latest 20):"]
        for topic, resource, status, ts in rows:
            lines.append(f"- [{status}] ({topic}) {resource} — {ts}")
        return "\n".join(lines)

    # =========================
    # KEYWORDS
    # =========================
    def _load_keyword_expansion(self) -> Dict[str, List[str]]:
        return {
            "ai": ["artificial intelligence", "machine learning", "deep learning", "neural network", "llm", "genai"],
            "llm": ["large language model", "transformer", "rag", "fine-tuning", "lora"],
            "openai": ["chatgpt", "gpt", "sora", "dall-e", "codex"],
            "chatgpt": ["openai", "gpt"],
            "claude": ["anthropic", "claude ai", "constitutional ai"],
            "nvidia": ["gpu", "cuda", "h100", "h200", "blackwell"],
            "microsoft": ["azure", "copilot", "windows"],
            "google": ["gemini", "deepmind", "cloud", "android"],
            "apple": ["ios", "mac", "m-series", "vision pro"],
            "uk": ["united kingdom", "britain", "london", "uk tech"],
            "cybersecurity": ["infosec", "vulnerability", "breach", "malware", "ransomware", "zero-day"],
        }

    def _expand_keywords(self, text: str) -> List[str]:
        t = (text or "").lower()
        keys = set(re.findall(r"[a-z0-9\-\+\.]+", t))

        for phrase, ex in self.keyword_expansion.items():
            if phrase in t:
                keys.add(phrase)
                keys.update(ex)

        for w in list(keys):
            if w in self.keyword_expansion:
                keys.update(self.keyword_expansion[w])

        cleaned = []
        for k in keys:
            k = k.strip().lower()
            if len(k) >= 2:
                cleaned.append(k)
        return sorted(set(cleaned))

    # =========================
    # LLM
    # =========================
    def _init_llm(self, base_model: str, hf_cache_dir: Optional[str]):
        self.llm_name = base_model
        self.llm_cache_dir = hf_cache_dir
        try:
            print("🔄 Loading local CPU model:", base_model)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=hf_cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model, cache_dir=hf_cache_dir)
            self.model.eval()
            self.llm_ready = True
            print("✅ Local CPU model loaded.")
        except Exception as e:
            self.llm_ready = False
            self.tokenizer = None
            self.model = None
            print("⚠️ Could not load local model.")
            print("   Reason:", e)

    def _llm_generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if not getattr(self, "llm_ready", False):
            return ""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=2,
                repetition_penalty=1.25,
                no_repeat_ngram_size=3,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return (text or "").strip()

    def _safe_json(self, text: str) -> Optional[Any]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        m2 = re.search(r"\[.*\]", text, flags=re.S)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass

        return None

    def _clean_llm_output(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        # Drop echoed prompt headers/transcript
        lines = []
        for line in text.splitlines():
            low = line.strip().lower()
            if low.startswith(("rules:", "user level:", "memory:", "recent:", "question:", "answer:", "you are techbot")):
                continue
            if low.startswith(("user:", "bot:")):
                continue
            # Drop “policy-ish” or instruction-like lines if model repeats them
            if "do not talk about" in low or "only output json" in low:
                continue
            lines.append(line)

        text = "\n".join(lines).strip()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _is_junk_output(self, out: str) -> bool:
        o = (out or "").strip()
        if not o:
            return True
        low = o.lower()

        junk_phrases = [
            "chatgpt is a free app",
            "customize your settings",
            "do not talk about",
            "only output json",
            "you are a strict json generator",
        ]
        if any(p in low for p in junk_phrases):
            return True

        # If answer is basically just a headline fragment, treat as junk
        if len(low) < 60 and any(x in low for x in ["invests", "announces", "report:", "says", "push", "orders"]):
            # not always junk, but in your case it's the common failure mode
            return True

        # repetition detector
        sentences = re.split(r"[.!?\n]+", low)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        if len(sentences) >= 6:
            top = max(sentences.count(s) for s in set(sentences))
            if top >= 4:
                return True

        return False

    # =========================
    # PREPROCESS / TOPIC / FOLLOWUP
    # =========================
    def preprocess(self, user_input: str) -> str:
        cleaned = re.sub(r"\s+", " ", (user_input or "").strip())
        processed = cleaned.lower()
        self.context["preprocess_meta"] = {
            "cleaned": cleaned,
            "keywords": self._expand_keywords(cleaned),
            "ts": _now_iso(),
        }
        return processed

    def _normalize_topic(self, text: str) -> str:
        t = (text or "").lower().strip()

        # typo fixes
        t = t.replace("open ai", "openai").replace("chat gpt", "chatgpt")
        t = t.replace("cloude", "claude").replace("claude ai", "claude")

        # special: UK tech
        if "uk" in t and "tech" in t:
            return "uk tech"
        if "united kingdom" in t and "tech" in t:
            return "uk tech"

        for k in [
            "uk tech", "openai", "chatgpt", "claude", "nvidia",
            "microsoft", "google", "apple", "cybersecurity", "ai"
        ]:
            if k in t:
                return k

        return t if len(t) <= 40 else t[:40]

    def _is_followup_like(self, processed_input: str) -> bool:
        p = (processed_input or "").strip().lower()
        if not p:
            return False
        # very short follow-ups
        if len(p.split()) <= 2 and p in {"how", "why", "what", "more", "again", "and", "then", "ok"}:
            return True
        # contains classic follow-up tokens
        toks = set(p.split())
        if toks.intersection({"it", "that", "this", "more", "again", "them", "those"}):
            return True
        return False

    # =========================
    # INTENT
    # =========================
    def classify_intent(self, processed_input: str) -> Dict[str, Any]:
        raw = self.context.get("preprocess_meta", {}).get("cleaned", processed_input)
        keywords = self.context.get("preprocess_meta", {}).get("keywords", [])

        if any(x in processed_input for x in ["hello", "hi", "hey"]):
            return {"greeting": True}
        if any(x in processed_input for x in ["bye", "quit", "exit"]):
            return {"farewell": True}

        if processed_input.strip() == "progress":
            return {"intents": ["progress"], "topic": "progress", "keywords": keywords}
        if processed_input.strip().startswith("done "):
            return {"intents": ["done"], "topic": "progress", "keywords": keywords}

        wants_sources = any(x in processed_input for x in ["links", "link", "source", "sources", "url"])

        # explicit news trigger
        news_triggers = [
            "what's new", "whats new", "what new", "latest", "recent", "news", "updates", "new feature", "release"
        ]
        if any(t in processed_input for t in news_triggers):
            topic = self._normalize_topic(raw)
            # if user said “update tech uk” ensure uk tech
            return {
                "intents": ["news"],
                "topic": topic,
                "entities": [],
                "wants_sources": wants_sources,
                "followup": False,
                "keywords": keywords,
            }

        followup = self._is_followup_like(processed_input) and len(self.conversation_history) > 0

        # If LLM exists, try JSON, but guard it hard.
        if getattr(self, "llm_ready", False):
            prompt = f"""
Return JSON only:
- intents: array from ["news","explain","resources","compare","howto","other"]
- topic: short topic string
- entities: array
- wants_sources: true/false
- followup: true/false

User: {raw}
JSON:
""".strip()
            out = self._llm_generate(prompt, max_new_tokens=120)
            parsed = self._safe_json(out)

            if isinstance(parsed, dict):
                parsed["keywords"] = keywords
                parsed["wants_sources"] = bool(parsed.get("wants_sources", wants_sources))
                parsed["followup"] = bool(parsed.get("followup", followup))
                parsed["topic"] = self._normalize_topic(parsed.get("topic", raw))
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed0 = parsed[0]
                parsed0["keywords"] = keywords
                parsed0["wants_sources"] = bool(parsed0.get("wants_sources", wants_sources))
                parsed0["followup"] = bool(parsed0.get("followup", followup))
                parsed0["topic"] = self._normalize_topic(parsed0.get("topic", raw))
                return parsed0

        # fallback classifier
        explain = any(x in processed_input for x in ["what is", "explain", "why", "how does", "meaning", "tell me about", "how "])
        resources = any(x in processed_input for x in ["learn", "course", "resources", "tutorial", "roadmap"])

        intents = []
        if explain:
            intents.append("explain")
        if resources:
            intents.append("resources")
        if not intents:
            intents.append("other")

        topic = self._normalize_topic(raw)

        # follow-up topic carryover if topic is generic/vague
        if followup and (topic in self._generic_topics or len(topic.split()) <= 1):
            topic = self.context.get("last_topic", topic)

        return {
            "intents": intents,
            "topic": topic,
            "entities": [],
            "wants_sources": wants_sources,
            "followup": followup,
            "keywords": keywords,
        }

    # =========================
    # NEWS
    # =========================
    def _static_rss_sources(self) -> Dict[str, str]:
        return {
            "BBC Tech": "https://feeds.bbci.co.uk/news/technology/rss.xml",
            "TechCrunch": "https://techcrunch.com/feed/",
            "The Verge": "https://www.theverge.com/rss/index.xml",
            "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
            "Wired": "https://www.wired.com/feed/rss",
        }

    def _google_news_rss(self, query: str, days: int = 7, hl: str = "en-GB", gl: str = "GB", ceid: str = "GB:en") -> str:
        q = requests.utils.quote(f"{query} when:{days}d")
        return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

    def _fetch_rss(self, url: str, source: str, timeout: int = 12) -> List[Dict[str, Any]]:
        headers = {"User-Agent": "TechBot/1.0 (RSS Reader)"}
        out: List[Dict[str, Any]] = []
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            feed = feedparser.parse(r.text)
            for e in feed.entries[:20]:
                title = _clean_text(getattr(e, "title", ""))
                link = _clean_text(getattr(e, "link", ""))
                published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
                summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
                summary = _clean_text(summary)
                if title:
                    out.append({"source": source, "title": title, "link": link, "published": published, "summary": summary})
        except Exception:
            return []
        return out

    def _fetch_news_items(self, query_hint: str) -> List[Dict[str, Any]]:
        key = (query_hint or "general").lower().strip()
        now = datetime.now().timestamp()
        if key in self._news_cache:
            ts, items = self._news_cache[key]
            if (now - ts) < 90:
                return items

        items: List[Dict[str, Any]] = []
        for src, url in self._static_rss_sources().items():
            items.extend(self._fetch_rss(url, src))

        if query_hint and len(query_hint.strip()) >= 2:
            items.extend(self._fetch_rss(self._google_news_rss(query_hint), "Google News"))

        seen = set()
        deduped = []
        for it in items:
            k = it.get("title", "").lower()
            if k and k not in seen:
                seen.add(k)
                deduped.append(it)

        self._news_cache[key] = (now, deduped)
        return deduped

    def _score_news(self, item: Dict[str, Any], keywords: List[str]) -> int:
        text = f"{item.get('title','')} {item.get('summary','')}".lower()
        score = 0
        for k in keywords:
            kk = (k or "").lower().strip()
            if len(kk) >= 3 and kk in text:
                score += 1
        return score

    def _select_top_news(self, items: List[Dict[str, Any]], keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        if not items:
            return []
        scored = [(self._score_news(it, keywords), it) for it in items]
        scored.sort(key=lambda x: x[0], reverse=True)
        relevant = [it for s, it in scored if s > 0]
        return (relevant[:limit] if relevant else [it for _, it in scored][:limit])

    # =========================
    # ANSWERING
    # =========================
    def _build_answer_prompt(
        self,
        profile: Dict[str, Any],
        memory_summary: str,
        recent_turns: List[Dict[str, str]],
        question: str,
        extra_context: Optional[str] = None,
    ) -> str:
        level = profile.get("knowledge_level", "beginner")

        # Keep *very* small context; avoid news bullets/headlines causing “headline answers”
        conv = []
        for t in recent_turns[-min(3, self.max_history_turns) :]:
            conv.append(f"User: {t['user']}\nBot: {t['bot']}")
        conv_text = _truncate("\n\n".join(conv).strip(), 650)

        memory_summary = _truncate(memory_summary, 250)
        extra = f"\nExtra context:\n{_truncate(extra_context, 160)}\n" if extra_context else ""

        return f"""
You are TechBot Pro. Answer the user's tech question.

Rules:
- Give a real tech answer (not headlines, not policy).
- Start with a direct answer (1-3 sentences).
- Then "Why it matters:" + 2-4 bullets.
- If beginner: briefly define key terms.
- Do NOT talk about "ChatGPT app" or "customize settings" unless user asked about ChatGPT settings.
- Avoid repetition. Do not repeat the question.

User level: {level}

Memory (short):
{memory_summary if memory_summary else "(none)"}

Recent (short):
{conv_text if conv_text else "(none)"}
{extra}

Question: {question}

Answer:
""".strip()

    def _fallback_answer(self, question: str) -> str:
        q = (question or "").strip()
        topic = self._normalize_topic(q) or self.context.get("last_topic", "")

        if topic == "uk tech":
            return (
                "Do you mean **UK tech news** (startups, policy, funding) or **UK cloud/AI** (Azure/AWS/GCP in UK regions)?\n\n"
                "Reply with one of:\n"
                "- `uk tech news`\n"
                "- `uk startup funding`\n"
                "- `uk ai policy`\n"
                "- `uk cloud (aws/azure/gcp)`"
            )

        if "cloud" in q.lower() or "cloud" in (topic or ""):
            return (
                "If you mean **cloud (AWS/Azure/GCP)** updates: tell me which one.\n\n"
                "Why it matters:\n"
                "- Cloud changes affect cost, security, and performance.\n"
                "- New AI services and GPUs change what you can deploy.\n\n"
                "Example: `what's new in Azure AI this week`"
            )

        if topic == "claude":
            return (
                "Claude (by Anthropic) is an LLM assistant used for writing, coding, and analysis.\n\n"
                "Why it matters:\n"
                "- Teams compare it with GPT models on reasoning and long-context work.\n"
                "- “New features” usually means new Claude versions, longer context, or new tooling.\n\n"
                "If you meant *latest Claude updates*, type: `latest claude with links`."
            )

        # generic “how/more” follow-up handler
        if self._is_followup_like(q.lower()):
            last = self.context.get("last_topic", "")
            if last:
                return f"Do you mean: **how students can take advantage of {last}** (learning, projects, jobs), or something else?"
            return "Can you add 2–3 words? Example: `how students take advantage of NVIDIA GPUs`."

        return (
            "I can answer that, but I need 1 detail: which product/company are you referring to?\n"
            "Example: `tell me about AWS new features` or `tell me about Gemini update`."
        )

    def _answer_with_llm(self, profile: Dict[str, Any], question: str, extra_context: Optional[str] = None) -> str:
        if not getattr(self, "llm_ready", False):
            return self._fallback_answer(question)

        # If question is too vague, ask for clarification (prevents “random headline” answers)
        q = (question or "").strip()
        if len(q.split()) <= 2 and q.lower() in {"how", "why", "more", "again"}:
            return self._fallback_answer(q)

        user_id = self.context.get("current_user", "default_user")
        mem = self._get_memory_summary(user_id)
        recent_turns = self._get_recent_dialogue(user_id, limit=self.max_history_turns)

        prompt = self._build_answer_prompt(profile, mem, recent_turns, question, extra_context=extra_context)
        out = self._llm_generate(prompt, max_new_tokens=220)
        out = self._clean_llm_output(out)

        if self._is_junk_output(out):
            return self._fallback_answer(question)

        return out

    # =========================
    # RESOURCES
    # =========================
    def _load_learning_resources(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "beginner": {
                "ai": ["Coursera: AI For Everyone (Andrew Ng)", "YouTube: 3Blue1Brown – Neural Networks"],
                "programming": ["Python Crash Course (Eric Matthes)", "freeCodeCamp – Python track"],
            },
            "intermediate": {
                "ai": ["fast.ai – Practical Deep Learning", "Build: RAG chatbot with embeddings + vector DB"],
            },
            "advanced": {
                "ai": ["Fine-tune with LoRA + eval harness", "Build: agent + tools + retrieval + monitoring"],
            },
        }

    def _get_personalized_resources(self, profile: Dict[str, Any], keywords: List[str]) -> List[str]:
        level = profile.get("knowledge_level", "beginner")
        bucket = self.learning_resources.get(level, self.learning_resources["beginner"])

        interests = [x.lower() for x in (profile.get("interests") or [])]
        kw = [k.lower() for k in keywords]

        candidates: List[str] = []
        for topic, arr in bucket.items():
            if topic in kw or topic in interests:
                candidates.extend(arr)

        if not candidates:
            candidates = sum(bucket.values(), [])

        out, seen = [], set()
        for r in candidates:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    # =========================
    # SAVE TURN + MEMORY UPDATE
    # =========================
    def save_turn(self, user_id: str, user_text: str, bot_text: str):
        self.cursor.execute(
            "INSERT INTO conversations (user_id, user_text, bot_text, ts) VALUES (?, ?, ?, ?)",
            (user_id, user_text, bot_text, _now_iso()),
        )
        self.conn.commit()
        self._update_memory_summary(user_id)

    # =========================
    # MAIN RETRIEVE
    # =========================
    def retrieve_info(self, intent: Dict[str, Any], processed_input: str) -> Dict[str, Any]:
        user_id = self.context.get("current_user", "default_user")
        profile = self.get_user_profile(user_id)

        cleaned = self.context.get("preprocess_meta", {}).get("cleaned", processed_input)
        keywords = intent.get("keywords") or self.context.get("preprocess_meta", {}).get("keywords", [])

        if intent.get("greeting"):
            return {"content": [f"👋 Hi! I’m {self.name}. Ask me a tech question (e.g., 'latest chatgpt updates')."], "resources": []}

        if intent.get("farewell"):
            return {"content": ["👋 Goodbye!"], "resources": []}

        intents = intent.get("intents") or ["other"]
        topic = (intent.get("topic") or "tech").strip()
        wants_sources = bool(intent.get("wants_sources", False))
        followup = bool(intent.get("followup", False))

        # follow-up carryover
        if followup and (topic in self._generic_topics or len(topic.split()) <= 1):
            topic = self.context.get("last_topic", topic)

        if "progress" in intents:
            return {"content": [self.get_progress_report(user_id)], "resources": []}

        if "done" in intents:
            k = cleaned.split("done", 1)[-1].strip()
            return {"content": [self.mark_resource_done(user_id, k)], "resources": []}

        sections: List[str] = []

        if "news" in intents:
            query_hint = topic if topic and topic not in ["tech", "other"] else cleaned
            items = self._fetch_news_items(query_hint=query_hint)
            top = self._select_top_news(items, keywords=keywords, limit=5)

            if not top:
                sections.append("⚠️ I couldn’t fetch live news right now (network/RSS blocked).")
            else:
                summaries = []
                for it in top[:4]:
                    # prefer title, then summary
                    s = it.get("title") or it.get("summary") or ""
                    s = _truncate(_clean_text(s), 220)
                    summaries.append(f"- {s}")

                sections.append(f"Here are recent updates related to **{topic or 'tech'}**:")
                sections.extend(summaries)
                sections.append("")
                sections.append("Why it matters:")
                sections.append("- Helps you track fast-moving product changes and releases.")
                sections.append("- Impacts what tools/features are available for building software.")

                if wants_sources:
                    sections.append("")
                    sections.append("Sources (requested):")
                    for it in top:
                        if it.get("link"):
                            sections.append(f"- {it['title']} ({it.get('source','')}) → {it['link']}")

        need_answer = any(x in intents for x in ["explain", "howto", "compare", "other", "resources"])
        if need_answer:
            extra_context = f"Follow-up on topic: {topic}" if followup and topic else None
            sections.append(self._answer_with_llm(profile, cleaned, extra_context=extra_context))

        if "resources" in intents:
            res = self._get_personalized_resources(profile, keywords)
            if res:
                sections.append("")
                sections.append("📚 Learning resources (picked for your level):")
                for r in res[:5]:
                    sections.append(f"- {r}")
                    self.track_resource(user_id, topic=topic, resource=r, status="suggested")
                sections.append("Tip: mark as done → done coursera  (or any keyword)")

        if not sections:
            sections = ["Ask me any tech question, like: 'uk tech news', 'latest nvidia updates', or 'explain cloud computing'."]

        # ✅ store “last topic” so “how / more” follows the right thing
        self.context["last_topic"] = topic
        self.context["last_intents"] = intents

        return {"content": [s for s in sections if s.strip() or s == ""], "resources": []}

    # =========================
    # IMPORTANT: Do not override respond(); override generate_response() to save turns.
    # =========================
    def generate_response(self, retrieved_info: dict) -> str:
        response = super().generate_response(retrieved_info)

        try:
            user_id = self.context.get("current_user", "default_user")
            user_text = ""
            if self.conversation_history:
                user_text = self.conversation_history[-1].get("user", "") or ""
            if user_text and response:
                self.save_turn(user_id, user_text, response)
        except Exception as e:
            print("⚠️ Save turn failed:", e)

        return response
