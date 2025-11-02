import os
import sys
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import re

import streamlit as st
from pdfminer.high_level import extract_text as pdf_extract_text
import chardet
import openpyxl
from PIL import Image
from PIL.ExifTags import TAGS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sehr große Bilder erlauben
Image.MAX_IMAGE_PIXELS = None

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False


# ------------------ KONFIG ------------------
BASE_DIR = Path(r"C:\Users\alexj\findit_neu")
DB_PATH = BASE_DIR / "embeddings.db"
LOG_PATH = BASE_DIR / "findit.log"

BASE_DIR.mkdir(exist_ok=True)

DEFAULT_DOC_DIRS = [r"C:\Users\alexj"]
DEFAULT_PHOTO_DIRS = [r"C:\Users\alexj"]

VALID_TEXT_EXT = {".txt", ".md"}
VALID_EXCEL_EXT = {".xlsx", ".xlsm"}
VALID_PDF_EXT = {".pdf"}
VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".bmp", ".gif"}

SKIP_DIRS = {
    ".git", ".venv", "node_modules", "AppData", "$RECYCLE.BIN", "__pycache__",
    "iCloudDrive", "venv", "findit_neu", "ProgramData", "Windows", ".cache"
}

CURRENT_MODEL = "clip-ViT-B-32"
EMBEDDING_DIM = 512

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ------------------ CUSTOM CSS ------------------
def load_custom_css():
    st.markdown("""
    <style>
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        color: white;
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    .score-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px 0;
    }
    .score-high { background: #10b981; color: white; }
    .score-med { background: #f59e0b; color: white; }
    .score-low { background: #ef4444; color: white; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .stat-box {
        background: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .path-display {
        background: #1e293b;
        color: #94a3b8;
        padding: 10px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        overflow-x: auto;
    }
    </style>
    """, unsafe_allow_html=True)


# ------------------ DATEI-OPERATIONEN ------------------
def open_file_windows(filepath: str) -> bool:
    """Versucht, eine Datei unter Windows zu öffnen – robust, auch bei Leerzeichen."""
    try:
        os.startfile(filepath)  # erster Versuch
        logging.info(f"✅ os.startfile erfolgreich: {filepath}")
        return True
    except Exception as e1:
        logging.error(f"⚠️ os.startfile fehlgeschlagen: {e1}")
        try:
            cmd = f'start "" "{filepath}"'
            subprocess.Popen(cmd, shell=True)
            logging.info(f"✅ cmd start erfolgreich: {filepath}")
            return True
        except Exception as e2:
            logging.error(f"❌ Open failed: {e1} | {e2}")
            return False


def show_in_explorer(filepath: str) -> bool:
    """Zeigt die Datei im Windows Explorer an."""
    try:
        subprocess.Popen(["explorer", f'/select,"{filepath}"'])
        logging.info(f"✅ explorer /select erfolgreich: {filepath}")
        return True
    except Exception as e1:
        logging.error(f"⚠️ explorer /select fehlgeschlagen: {e1}")
        try:
            folder = os.path.dirname(filepath)
            if not folder:
                folder = os.getcwd()
            subprocess.Popen(["explorer", folder])
            logging.info(f"✅ explorer folder erfolgreich: {folder}")
            return True
        except Exception as e2:
            logging.error(f"❌ Show failed: {e1} | {e2}")
            return False


# ------------------ DB ------------------
def get_conn():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn
    except sqlite3.Error as e:
        logging.error(f"DB-Fehler: {e}")
        return None


def init_db():
    try:
        conn = get_conn()
        if not conn:
            return
        cur = conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS text_docs(
                path TEXT PRIMARY KEY,
                content TEXT,
                mtime INTEGER
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(
                path UNINDEXED,
                content,
                tokenize='unicode61'
            );
            CREATE TABLE IF NOT EXISTS image_embeds(
                path TEXT PRIMARY KEY,
                vector BLOB,
                mtime INTEGER,
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                date_taken INTEGER
            );
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"DB-Init-Fehler: {e}")


# ------------------ TEXT ------------------
def read_text_file(path: Path) -> str:
    try:
        data = path.read_bytes()
        enc = chardet.detect(data).get("encoding") or "utf-8"
        return data.decode(enc, errors="replace")
    except:
        return ""


def read_excel_file(path: Path) -> str:
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        chunks = []
        for ws in wb.worksheets:
            try:
                for row in ws.iter_rows(values_only=True):
                    vals = [str(c) for c in row if c not in (None, "None")]
                    if vals:
                        chunks.append(" ".join(vals))
            except:
                continue
        wb.close()
        return "\n".join(chunks)
    except:
        return ""


def read_pdf_file(path: Path) -> str:
    try:
        return pdf_extract_text(str(path)) or ""
    except:
        return ""


def index_text_docs(doc_dirs, force_reindex: bool = False):
    init_db()
    conn = get_conn()
    if not conn:
        return

    cur = conn.cursor()
    indexed = 0
    skipped = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        all_files = []
        for base_str in doc_dirs:
            base = Path(base_str)
            if not base.exists():
                continue

            for root, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for fn in filenames:
                    p = Path(root) / fn
                    ext = p.suffix.lower()
                    if ext in VALID_TEXT_EXT or ext in VALID_EXCEL_EXT or ext in VALID_PDF_EXT:
                        all_files.append(p)

        total_files = len(all_files)
        if total_files == 0:
            st.warning("Keine Dokumente gefunden")
            conn.close()
            return

        st.info(f"🔍 {total_files} Dokumente gefunden")

        for idx, p in enumerate(all_files):
            try:
                ext = p.suffix.lower()
                if not p.exists():
                    continue

                content = ""
                if ext in VALID_TEXT_EXT:
                    content = read_text_file(p)
                elif ext in VALID_EXCEL_EXT:
                    content = read_excel_file(p)
                elif ext in VALID_PDF_EXT:
                    content = read_pdf_file(p)

                if not content.strip():
                    skipped += 1
                    continue

                try:
                    mtime = int(p.stat().st_mtime)
                except:
                    mtime = 0

                if not force_reindex:
                    existing = cur.execute("SELECT mtime FROM text_docs WHERE path = ?", (str(p),)).fetchone()
                    if existing and existing[0] == mtime:
                        skipped += 1
                        continue

                cur.execute(
                    "REPLACE INTO text_docs(path, content, mtime) VALUES (?,?,?)",
                    (str(p), content, mtime)
                )
                cur.execute("DELETE FROM text_fts WHERE path = ?", (str(p),))
                cur.execute("INSERT INTO text_fts(path, content) VALUES (?,?)", (str(p), content))
                indexed += 1

                if idx % 10 == 0:
                    progress_bar.progress(min((idx + 1) / total_files, 1.0))
                    status_text.text(f"📊 {idx+1}/{total_files} | ✅ {indexed}")
            except:
                skipped += 1

        conn.commit()
        progress_bar.progress(1.0)

        if indexed > 0:
            st.success(f"✅ {indexed} indexiert, {skipped} übersprungen")
        else:
            st.info(f"ℹ️ Alle aktuell ({skipped} übersprungen)")
    except Exception as e:
        st.error(f"Fehler: {e}")
    finally:
        conn.close()


def search_text(query: str, limit: int = 100):
    init_db()
    conn = get_conn()
    if not conn:
        return []

    cur = conn.cursor()
    try:
        rows = list(cur.execute("""
            SELECT path, snippet(text_fts, 1, '[', ']', ' … ', 10) AS snip, rank
            FROM text_fts 
            WHERE text_fts MATCH ? 
            LIMIT ?
        """, (query, limit)))
    except:
        rows = []

    results = []
    for path, snip, rank in rows:
        try:
            info = conn.execute("SELECT mtime FROM text_docs WHERE path = ?", (path,)).fetchone()
            mtime = info[0] if info else 0
            results.append({'path': path, 'mtime': mtime, 'snip': snip, 'rank': rank})
        except:
            continue

    conn.close()
    results.sort(key=lambda x: x['rank'])
    return [(r['path'], r['mtime'], r['snip']) for r in results]


# ------------------ BILDER ------------------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = SentenceTransformer(CURRENT_MODEL)
        logging.info(f"Modell geladen: {CURRENT_MODEL}")
        return model
    except Exception as e:
        st.error(f"Modell-Fehler: {e}")
        return None


def extract_exif_date(img: Image.Image, fallback: int) -> int:
    try:
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ("DateTime", "DateTimeOriginal", "DateTimeDigitized"):
                    try:
                        return int(datetime.strptime(value, "%Y:%m:%d %H:%M:%S").timestamp())
                    except:
                        continue
    except:
        pass
    return fallback


def index_images_batch(photo_dirs, batch_size: int = 32, force_reindex: bool = False):
    init_db()
    conn = get_conn()
    if not conn:
        return

    cur = conn.cursor()
    model = load_model()
    if not model:
        conn.close()
        return

    stats = {'indexed': 0, 'already_indexed': 0, 'skipped': 0}
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        all_files = []
        for base_str in photo_dirs:
            base = Path(base_str)
            if not base.exists():
                continue
            for root, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for fn in filenames:
                    p = Path(root) / fn
                    if p.suffix.lower() in VALID_IMG_EXT:
                        all_files.append(p)

        total_files = len(all_files)
        if total_files == 0:
            st.warning("Keine Bilder gefunden")
            conn.close()
            return

        st.info(f"🎯 {total_files} Bilder gefunden")

        batch_paths, batch_images, batch_metadata = [], [], []

        for idx, p in enumerate(all_files):
            try:
                if not p.exists():
                    continue

                stat_info = p.stat()
                mtime, file_size = int(stat_info.st_mtime), stat_info.st_size

                if file_size < 1000:
                    stats['skipped'] += 1
                    continue

                if not force_reindex:
                    existing = cur.execute("SELECT mtime FROM image_embeds WHERE path = ?", (str(p),)).fetchone()
                    if existing and existing[0] == mtime:
                        stats['already_indexed'] += 1
                        continue

                img = Image.open(p).convert("RGB")
                width, height = img.size
                date_taken = extract_exif_date(img, mtime)

                img_thumb = img.copy()
                img_thumb.thumbnail((384, 384), Image.Resampling.LANCZOS)

                batch_paths.append(p)
                batch_images.append(img_thumb)
                batch_metadata.append({
                    'mtime': mtime,
                    'width': width,
                    'height': height,
                    'file_size': file_size,
                    'date_taken': date_taken
                })

                if len(batch_images) >= batch_size or idx == total_files - 1:
                    if batch_images:
                        embeddings = model.encode(
                            batch_images,
                            batch_size=len(batch_images),
                            convert_to_tensor=False,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )

                        for path, emb, meta in zip(batch_paths, embeddings, batch_metadata):
                            embedding = np.array(emb, dtype=np.float32).flatten()
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm

                            blob = embedding.tobytes()
                            cur.execute(
                                """REPLACE INTO image_embeds(path, vector, mtime, width, height, file_size, date_taken) 
                                   VALUES (?,?,?,?,?,?,?)""",
                                (
                                    str(path), blob, meta['mtime'],
                                    meta['width'], meta['height'],
                                    meta['file_size'], meta['date_taken']
                                )
                            )
                            stats['indexed'] += 1

                        conn.commit()
                        batch_paths, batch_images, batch_metadata = [], [], []

                if idx % 5 == 0:
                    progress_bar.progress(min((idx + 1) / total_files, 1.0))
                    status_text.text(f"📊 {idx+1}/{total_files} | ✅ {stats['indexed']}")
            except:
                stats['skipped'] += 1

        progress_bar.progress(1.0)

        if stats['indexed'] > 0:
            st.balloons()
            st.success(f"🎉 {stats['indexed']} Bilder indexiert!")
        if stats['already_indexed'] > 0:
            st.info(f"ℹ️ {stats['already_indexed']} bereits aktuell")
    except Exception as e:
        st.error(f"Fehler: {e}")
    finally:
        conn.close()


def search_images(query: str, top_k: int = 50, min_score: float = 0.20, base_filter: str = ""):
    """
    Bildsuche:
    - CLIP-Score (visuell)
    - + Dateiname/Ordner-Score
    - optional: nur Treffer aus Pfaden, die base_filter enthalten
    """
    init_db()
    conn = get_conn()
    if not conn:
        return []

    try:
        rows = list(conn.execute("SELECT path, vector, date_taken FROM image_embeds"))
    except:
        conn.close()
        return []

    if not rows:
        conn.close()
        return []

    model = load_model()
    if not model:
        conn.close()
        return []

    # 1) CLIP-Embedding der Query (Englisch bleibt so)
    try:
        q_emb = model.encode(query, convert_to_tensor=False, normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32).flatten()
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
    except:
        conn.close()
        return []

    # 2) Alle Bild-Embeddings laden
    paths, embeddings, dates = [], [], []
    base_filter = base_filter.strip().lower()
    for path, blob, date_taken in rows:
        # optionaler Ordnerfilter
        if base_filter and base_filter not in path.lower():
            continue
        try:
            vec = np.frombuffer(blob, dtype="float32")
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
            paths.append(path)
            dates.append(date_taken or 0)
        except:
            continue

    if not embeddings:
        conn.close()
        return []

    # 3) Cosine-Similarity (CLIP)
    embedding_matrix = np.vstack(embeddings)
    clip_scores = cosine_similarity(embedding_matrix, q_emb.reshape(1, -1)).flatten()

    # 4) zusätzlich: Filename-/Folder-Score
    from difflib import SequenceMatcher

    def filename_score(q: str, p: str) -> float:
        q = q.lower()
        pth = Path(p)
        name = pth.name.lower()
        parent = pth.parent.name.lower()
        s1 = SequenceMatcher(None, q, name).ratio()
        s2 = SequenceMatcher(None, q, parent).ratio()
        return max(s1, s2)

    combined = []
    for i, p in enumerate(paths):
        cscore = float(clip_scores[i])
        # CLIP-Score Mindestschwelle – aber nicht komplett wegwerfen,
        # denn Dateiname kann retten
        if cscore < 0.05:
            # sehr schwach → trotzdem zulassen, aber der final score wird niedrig
            pass
        fscore = filename_score(query, p)
        final_score = 0.7 * cscore + 0.3 * fscore
        if final_score >= min_score * 0.6:  # etwas großzügiger als reiner CLIP
            combined.append((p, final_score))

    # 5) sortieren + top_k
    combined.sort(key=lambda x: x[1], reverse=True)
    results = combined[:top_k]

    conn.close()
    return results


# ------------------ ACTION STATE ------------------
def execute_action(filepath, action_type):
    """Führt Aktion aus und speichert Ergebnis in session_state."""
    logging.info(f"🎯 EXECUTE_ACTION: {action_type} für {filepath}")

    if not os.path.exists(filepath):
        result = {"success": False, "message": "Datei existiert nicht"}
    else:
        if action_type == "open":
            success = open_file_windows(filepath)
            result = {
                "success": success,
                "message": "Datei wird geöffnet" if success else "Fehler beim Öffnen"
            }
        else:
            success = show_in_explorer(filepath)
            result = {
                "success": success,
                "message": "Explorer öffnet" if success else "Fehler"
            }

    st.session_state.action_result = result
    logging.info(f"✅ ACTION_RESULT: {result}")
    return result


# ------------------ UI ------------------
def main():
    st.set_page_config(
        page_title="FindIt AI",
        layout="wide",
        page_icon="🔎",
        initial_sidebar_state="collapsed"
    )

    # Session-Defaults
    if "text_results" not in st.session_state:
        st.session_state["text_results"] = []
    if "image_results" not in st.session_state:
        st.session_state["image_results"] = []
    if "action_result" not in st.session_state:
        st.session_state.action_result = None

    load_custom_css()

    st.markdown("""
    <div class="main-header">
        <h1>🔎 FindIt AI v12.0</h1>
        <p style="font-size: 18px; margin: 0;">Moderne AI-Dateisuche • Schnell • Präzise • Schön</p>
    </div>
    """, unsafe_allow_html=True)

    # Action-Rückmeldung
    if st.session_state.action_result:
        result = st.session_state.action_result
        if result["success"]:
            st.success(f"✅ {result['message']}")
        else:
            st.error(f"❌ {result['message']}")
        st.session_state.action_result = None

    tab1, tab2, tab3 = st.tabs(["📄 Text-Suche", "🖼 Bild-Suche", "⚙️ Verwaltung"])

    # ------------------ TAB 1: TEXT ------------------
    with tab1:
        st.subheader("📄 Dokumente durchsuchen")

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Suchbegriff",
                placeholder="z.B. Rechnung, Vertrag, Protokoll...",
                label_visibility="collapsed",
                key="text_query",
            )
        with col2:
            st.session_state["text_limit"] = st.selectbox(
                "Ergebnisse",
                [20, 50, 100, 200],
                index=1,
                key="text_limit_select"
            )

        if st.button("🔍 Suchen", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Suche läuft..."):
                    results = search_text(query.strip(), st.session_state["text_limit"])
                st.session_state["text_results"] = results
            else:
                st.warning("Bitte Suchbegriff eingeben")

        results = st.session_state.get("text_results", [])
        if results:
            st.success(f"✅ {len(results)} Treffer gefunden")

            for idx, (path, mtime, snip) in enumerate(results):
                dt = datetime.fromtimestamp(mtime).strftime("%d.%m.%Y %H:%M")
                filename = Path(path).name

                st.markdown(f"""
                <div class="result-card">
                    <h3 style="margin: 0 0 10px 0;">📄 {filename}</h3>
                    <p style="margin: 5px 0; opacity: 0.9;">📅 {dt}</p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📋 Details", expanded=False):
                    st.markdown(f'<div class="path-display">{path}</div>', unsafe_allow_html=True)
                    if snip:
                        st.markdown("**Vorschau:**")
                        st.code(snip, language=None)

                c1, c2, _ = st.columns([1, 1, 4])
                with c1:
                    if st.button("📂 Öffnen", key=f"open_txt_{idx}"):
                        execute_action(path, "open")
                        st.rerun()
                with c2:
                    if st.button("📁 Explorer", key=f"show_txt_{idx}"):
                        execute_action(path, "show")
                        st.rerun()

                st.markdown("---")
        else:
            st.info("Noch keine Ergebnisse. Such oben nach einem Begriff.")

    # ------------------ TAB 2: BILDER ------------------
    with tab2:
        st.subheader("🖼 Bilder durchsuchen")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            query_img = st.text_input(
                "Suchbegriff",
                placeholder="z.B. sunset, person, cat...",
                label_visibility="collapsed",
                key="img_query",
            )
        with col2:
            st.session_state["img_min_score"] = st.slider(
                "Min Score",
                0.0,
                1.0,
                0.22,
                0.02,
                key="img_min_score_slider"
            )
        with col3:
            st.session_state["img_top_k"] = st.selectbox(
                "Max",
                [24, 48, 100],
                index=1,
                key="img_top_k_select"
            )

        # zusätzlicher Ordner-Filter
        base_filter = st.text_input(
            "Optional: nur in diesem Ordner/Pfad suchen",
            "",
            key="img_folder_filter",
            help="z.B. atelier, C:\\Users\\alexj\\Bilder oder 'kunst'"
        )

        if st.button("🔍 Suchen", type="primary", use_container_width=True, key="search_img_btn"):
            if query_img.strip():
                with st.spinner("🔍 Suche läuft..."):
                    hits = search_images(
                        query_img.strip(),
                        top_k=st.session_state["img_top_k"],
                        min_score=st.session_state["img_min_score"],
                        base_filter=base_filter
                    )
                st.session_state["image_results"] = hits
            else:
                st.warning("Bitte Suchbegriff eingeben")

        hits = st.session_state.get("image_results", [])
        if hits:
            st.success(f"✅ {len(hits)} Bilder gefunden")

            for i, (path, score) in enumerate(hits):
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    try:
                        if os.path.exists(path):
                            img = Image.open(path)
                            st.image(img, width=250)
                    except:
                        st.error("❌ Fehler beim Laden")

                with col_info:
                    if score >= 0.4:
                        badge_class = "score-high"
                        emoji = "🟢"
                    elif score >= 0.28:
                        badge_class = "score-med"
                        emoji = "🟡"
                    else:
                        badge_class = "score-low"
                        emoji = "🟠"

                    st.markdown(
                        f'<span class="score-badge {badge_class}">{emoji} Score: {score:.3f}</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"**{Path(path).name}**")

                    with st.expander("📋 Pfad", expanded=False):
                        st.markdown(f'<div class="path-display">{path}</div>', unsafe_allow_html=True)

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("📂 Öffnen", key=f"open_img_{i}"):
                            execute_action(path, "open")
                            st.rerun()
                    with b2:
                        if st.button("📁 Explorer", key=f"show_img_{i}"):
                            execute_action(path, "show")
                            st.rerun()

                st.markdown("---")
        else:
            st.info("Noch keine Bilder gefunden. Such oben nach einem Motiv.")

    # ------------------ TAB 3: VERWALTUNG ------------------
    with tab3:
        st.subheader("⚙️ Verwaltung")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Dokumente indexieren")
            doc_dirs = st.text_area(
                "Ordner",
                "\n".join(DEFAULT_DOC_DIRS),
                height=80,
                key="doc_dirs_input"
            )
            force_doc = st.checkbox("Alle neu indexieren", key="force_doc_new")
            if st.button("📄 INDEXIEREN", type="primary", use_container_width=True):
                dirs = [d.strip() for d in doc_dirs.split('\n') if d.strip()]
                if dirs:
                    index_text_docs(dirs, force_doc)

        with col2:
            st.markdown("### 🖼 Bilder indexieren")
            photo_dirs = st.text_area(
                "Ordner",
                "\n".join(DEFAULT_PHOTO_DIRS),
                height=80,
                key="photo_dirs_input"
            )
            force_img = st.checkbox("Alle neu indexieren", key="force_img_new")
            if st.button("🖼 INDEXIEREN", type="primary", use_container_width=True):
                dirs = [d.strip() for d in photo_dirs.split('\n') if d.strip()]
                if dirs:
                    index_images_batch(dirs, 32, force_img)

        st.markdown("---")
        st.markdown("### 🔬 Diagnose")

        if st.button("🔍 Datenbank prüfen", use_container_width=True):
            conn = get_conn()
            if conn:
                try:
                    img_count = conn.execute("SELECT COUNT(*) FROM image_embeds").fetchone()[0]
                    doc_count = conn.execute("SELECT COUNT(*) FROM text_docs").fetchone()[0]

                    c1, c2 = st.columns(2)
                    c1.markdown(
                        f'<div class="stat-box"><h2>{doc_count:,}</h2><p>Dokumente</p></div>',
                        unsafe_allow_html=True
                    )
                    c2.markdown(
                        f'<div class="stat-box"><h2>{img_count:,}</h2><p>Bilder</p></div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"❌ DB-Fehler: {e}")
                finally:
                    conn.close()

        st.markdown("---")
        st.markdown("### 🧪 System-Test")

        if st.button("🎯 Datei-Aktionen testen", use_container_width=True):
            test_file = r"C:\Windows\System32\notepad.exe"

            if os.path.exists(test_file):
                st.success("✅ Testdatei gefunden")

                st.markdown("**Test 1: Datei öffnen**")
                result1 = open_file_windows(test_file)
                if result1:
                    st.success("✅ Notepad sollte sich öffnen")
                else:
                    st.error("❌ Fehler")

                st.markdown("**Test 2: Explorer**")
                result2 = show_in_explorer(test_file)
                if result2:
                    st.success("✅ Explorer sollte sich öffnen")
                else:
                    st.error("❌ Fehler")

                if os.path.exists(LOG_PATH):
                    st.markdown("**📝 Log:**")
                    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        st.code("\n".join(lines[-15:]), language=None)
            else:
                st.error("❌ Testdatei nicht gefunden")

        st.markdown("---")
        st.markdown("""
        ### 💡 Version 12.0 - MODERN EDITION

        **Features:**
        - ✅ Moderne Gradient-UI mit Cards
        - ✅ Funktionierende Datei-Aktionen
        - ✅ Optimierte Bildsuche (clip-ViT-B-32)
        - ✅ Score-Badges mit Farbcodierung
        - ✅ Schnelle Volltext-Suche (FTS5)

        **Tipps:**
        - **Bildsuche:** Englische Begriffe verwenden
        - **Min Score 0.20-0.25:** Viele Ergebnisse
        - **Min Score 0.30+:** Sehr präzise
        - **Batch Size 32:** Optimal für die meisten PCs

        Made with ❤️ • Powered by Sentence Transformers
        """)


if __name__ == "__main__":
    init_db()
    main()
