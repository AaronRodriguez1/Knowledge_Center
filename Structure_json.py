'''
Structure_json.py   

Create SQLite database from ChatGPT export
'''

import json, sqlite3, pathlib, datetime as dt
from sentence_transformers import SentenceTransformer
import hdbscan, numpy as np, pandas as pd
from tqdm import tqdm
import datetime as dt
EXPORT_FILE = "conversations.json"        
DB_FILE     = "knowledge.db"


def flatten_parts(parts):
    """Helper for legacy 'parts' 
    """
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            # try common text fields
            for k in ("text", "value", "caption", "alt"):
                if k in p and isinstance(p[k], str):
                    out.append(p[k])
                    break
    return " ".join(out).strip()

def content_to_text(content):
    """
    Pull text from ANY message['content'] variant.
    """
    if content is None:
        return ""

    if isinstance(content, dict) and "parts" in content:
        return flatten_parts(content["parts"])

    if isinstance(content, dict):
        for k in ("text", "value", "caption", "alt"):
            if k in content and isinstance(content[k], str):
                return content[k].strip()


    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return flatten_parts(content)

    return ""   # non-text (binary, images, etc.)


def linearise_conversation(conv: dict):
    title = conv.get("title", "Untitled-Chat")
    rows  = []

    for node in conv["mapping"].values():
        msg = node.get("message")
        if not msg:
            continue

        role = msg["author"]["role"]
        if role not in {"user", "assistant"}:
            continue

        text = content_to_text(msg.get("content"))
        if not text:
            continue

        ts_raw = msg.get("create_time")
        ts = (dt.datetime.fromtimestamp(ts_raw, tz=dt.timezone.utc)
              if ts_raw else None)

        rows.append((title, role, text, ts))

    rows.sort(key=lambda r: (r[3] or dt.datetime.min.replace(tzinfo=dt.timezone.utc)))
    return rows


print("Reading export …")
raw = json.loads(pathlib.Path(EXPORT_FILE).read_text())

all_rows = []
for conv in tqdm(raw, desc="Conversations"):
    all_rows.extend(linearise_conversation(conv))

df = pd.DataFrame(all_rows, columns=["conv","role","text","ts"])
df["ts"] = pd.to_datetime(df["ts"])          # NaT for missing


print("Embedding & clustering …")
model   = SentenceTransformer("all-MiniLM-L6-v2")
embeds  = model.encode(df["text"].tolist(), show_progress_bar=True)
cluster = hdbscan.HDBSCAN(min_cluster_size=20, metric="euclidean").fit(embeds)
df["topic"] = cluster.labels_           #  -1 == noise / misc.


print("Writing SQLite …")
con = sqlite3.connect(DB_FILE)
cur = con.cursor()
cur.executescript("""
CREATE TABLE IF NOT EXISTS topic (
  id INTEGER PRIMARY KEY,
  title   TEXT,
  summary TEXT
);
CREATE TABLE IF NOT EXISTS message (
  id INTEGER PRIMARY KEY,
  topic_id INTEGER,
  conv   TEXT,
  role   TEXT,
  ts     TEXT,
  text   TEXT
);
""")

for label in sorted(set(df["topic"])):
    subset = df[df["topic"] == label]
    
    head = subset.iloc[0]
    title = (head["text"][:60] + "…") if len(head["text"]) > 60 else head["text"]
    cur.execute("INSERT INTO topic(title) VALUES (?)", (title,))
    topic_id = cur.lastrowid

    for _, r in subset.iterrows():
        cur.execute(
            "INSERT INTO message(topic_id,conv,role,ts,text) VALUES (?,?,?,?,?)",
            (topic_id, r["conv"], r["role"],
             r["ts"].isoformat() if pd.notnull(r["ts"]) else None,
             r["text"])
        )

con.commit()
con.close()
print("✅ knowledge.db ready")