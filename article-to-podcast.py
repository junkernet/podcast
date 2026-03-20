#!/usr/bin/env python3
"""
article-to-podcast.py — Convert articles/text to podcast episodes (v2).

Generates MP3 via OpenAI TTS (voice: nova), with rich metadata, spoken intro,
Whisper transcription (WebVTT), chapter markers, Telegram notification,
and Obsidian sync. Commits to git repo and pushes to GitHub Pages.

Usage:
    python3 article-to-podcast.py --url "https://example.com/article"
    python3 article-to-podcast.py --text "Some text to read aloud"
    python3 article-to-podcast.py --file /path/to/article.txt
    python3 article-to-podcast.py --list
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from email.utils import formatdate
from pathlib import Path
from urllib.parse import quote, urlparse

import time

import requests
from openai import OpenAI

# --- Configuration ---
REPO_DIR = Path.home() / "podcast"
EPISODES_DIR = REPO_DIR / "episodes"
FEED_FILE = REPO_DIR / "feed.xml"
INDEX_FILE = REPO_DIR / "index.html"
ARTWORK_PATH = REPO_DIR / "artwork.jpg"

BASE_URL = "https://junkernet.github.io/podcast"
FEED_URL = f"{BASE_URL}/feed.xml"
EPISODES_URL = f"{BASE_URL}/episodes"

TTS_MODEL = "tts-1"
TTS_VOICE = "nova"
TTS_CHUNK_LIMIT = 4096

GIT_AUTHOR_NAME = "Claw"
GIT_AUTHOR_EMAIL = "claw@junkernet.github"


def _retry(fn, *, retries=3, backoff=2.0, label="API call"):
    """Retry a callable with exponential backoff on transient errors."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            is_transient = any(x in err_str.lower() for x in [
                "429", "rate_limit", "500", "502", "503", "timeout",
                "connection", "server_error",
            ])
            if not is_transient or attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"  ⚠️ {label} failed (attempt {attempt + 1}/{retries}): {e}")
            print(f"  Retrying in {wait:.0f}s...")
            time.sleep(wait)

TELEGRAM_CHAT_ID = "8660670549"
TELEGRAM_BOT_TOKEN = None  # loaded at runtime

WPM = 140  # Nova's approximate words per minute

# Garbage detection phrases
GARBAGE_PHRASES = [
    "Subscribe to continue reading",
    "Enable JavaScript",
    "Please turn off your ad blocker",
    "Please enable cookies",
    "Access denied",
    "403 Forbidden",
    "You need to enable JavaScript",
    "This content is available to subscribers",
]

OBS_BIN = Path.home() / ".local" / "bin" / "obs"


def get_telegram_token():
    """Load Telegram bot token from openclaw config."""
    global TELEGRAM_BOT_TOKEN
    if TELEGRAM_BOT_TOKEN:
        return TELEGRAM_BOT_TOKEN
    # Try env first
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        TELEGRAM_BOT_TOKEN = token
        return token
    # Read from openclaw.json
    try:
        cfg_path = Path.home() / ".openclaw" / "openclaw.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        token = cfg.get("channels", {}).get("telegram", {}).get("botToken", "")
        if token:
            TELEGRAM_BOT_TOKEN = token
            return token
    except Exception:
        pass
    return None


def send_telegram(message: str):
    """Send a message to Jeff via Telegram."""
    token = get_telegram_token()
    if not token:
        print("  ⚠️ No Telegram bot token found, skipping notification")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.ok
    except Exception as e:
        print(f"  ⚠️ Telegram notification failed: {e}")
        return False


def atomic_write(path: Path, content: str):
    """Write content to file atomically using temp file + rename."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix='.xml')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(content)
        os.rename(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm for WebVTT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _find_ttf_font() -> str | None:
    """Find a good sans-serif TTF font on the system."""
    # Prefer DejaVu Sans for clean, readable episode art
    preferred = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in preferred:
        if os.path.exists(p):
            return p
    fonts = glob.glob("/usr/share/fonts/truetype/**/*.ttf", recursive=True)
    return fonts[0] if fonts else None


def _find_ttf_font_bold() -> str | None:
    """Find a bold sans-serif TTF font on the system."""
    preferred = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(preferred):
        return preferred
    return _find_ttf_font()


def generate_episode_art(source: str, title: str, output_path: Path):
    """Generate per-episode artwork using Pillow (3000x3000 for Apple Podcasts).

    Brand: dark navy (#1a1a2e) background with claw pattern texture from
    banner.jpg, white text. Matches feed artwork aesthetic.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  ⚠️ Pillow not installed, skipping episode art")
        return False

    W, H = 3000, 3000
    img = Image.new("RGB", (W, H), "#1a1a2e")

    # Try to overlay banner.jpg as a subtle background pattern
    script_dir = Path(__file__).resolve().parent
    # Check both the podcast repo and script directory for banner
    banner_paths = [
        Path.home() / "podcast" / "banner.jpg",
        script_dir.parent / "banner.jpg",
    ]
    for bp in banner_paths:
        if bp.exists():
            try:
                banner = Image.open(bp).convert("RGB")
                # Tile the banner pattern across the canvas at low opacity
                bw, bh = banner.size
                overlay = Image.new("RGB", (W, H), "#1a1a2e")
                for y in range(0, H, bh):
                    for x in range(0, W, bw):
                        overlay.paste(banner, (x, y))
                # Blend at 30% opacity for subtle texture
                img = Image.blend(img, overlay, 0.3)
            except Exception:
                pass  # Fall back to solid background
            break

    draw = ImageDraw.Draw(img)

    font_path = _find_ttf_font()
    font_bold = _find_ttf_font_bold()

    # Scale fonts for 3000x3000 canvas
    try:
        if font_bold:
            font_source = ImageFont.truetype(font_bold, 100)
            font_title = ImageFont.truetype(font_bold, 130)
        elif font_path:
            font_source = ImageFont.truetype(font_path, 100)
            font_title = ImageFont.truetype(font_path, 130)
        else:
            raise IOError("no font")
        font_date = ImageFont.truetype(font_path or font_bold, 72)
    except (IOError, OSError):
        font_source = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_date = ImageFont.load_default()

    # Source: top center, light gray, all caps
    source_text = source.upper() if source else "UNKNOWN"
    bbox = draw.textbbox((0, 0), source_text, font=font_source)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) / 2, 280), source_text, fill="#aaaaaa", font=font_source)

    # Thin separator line below source
    line_y = 440
    draw.line([(W // 4, line_y), (3 * W // 4, line_y)], fill="#444444", width=3)

    # Title: center, white, word-wrapped
    wrapped = textwrap.wrap(title, width=22)[:5]
    line_height = 160
    total_title_h = len(wrapped) * line_height
    y_start = (H - total_title_h) / 2 + 60
    for i, line in enumerate(wrapped):
        bbox = draw.textbbox((0, 0), line, font=font_title)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) / 2, y_start + i * line_height), line, fill="#ffffff", font=font_title)

    # Date: bottom center, gray
    date_text = datetime.now().strftime("%B %d, %Y").replace(" 0", " ")
    bbox = draw.textbbox((0, 0), date_text, font=font_date)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) / 2, H - 340), date_text, fill="#666666", font=font_date)

    img.save(str(output_path), "JPEG", quality=90, dpi=(72, 72))
    print(f"  Episode art saved: {output_path}")
    return True


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:60].strip("-")


def extract_metadata(url: str) -> dict:
    """Extract rich metadata from a URL: title, author, date, source, text, word_count."""
    metadata = {
        "title": "",
        "author": "",
        "source": "",
        "published_date": "",
        "url": url,
        "text": "",
        "word_count": 0,
    }

    # Step 1: Get clean article text via Tavily
    tavily_script = Path.home() / ".openclaw/skills/tavily/scripts/tavily.py"
    tavily_text = ""
    tavily_title = ""
    if tavily_script.exists():
        try:
            result = subprocess.run(
                ["python3", str(tavily_script), "extract", url],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = result.stdout.strip()
                # Tavily extract outputs plain text (not JSON)
                # Strip the header line "── URL ──" if present
                lines = raw.split('\n')
                if lines and '──' in lines[0]:
                    lines = lines[1:]
                tavily_text = '\n'.join(lines).strip()
        except Exception:
            pass

    # Step 2: Fetch raw HTML for metadata extraction
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("beautifulsoup4 not installed. pip3 install --user beautifulsoup4")
        sys.exit(1)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; PodcastBot/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  Warning: Could not fetch HTML for metadata: {e}")
        soup = None

    if soup:
        # Title: og:title > h1 > <title>
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            metadata["title"] = og_title["content"].strip()
        elif soup.find("h1"):
            metadata["title"] = soup.find("h1").get_text(strip=True)
        elif soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        # Source: og:site_name > domain
        og_site = soup.find("meta", property="og:site_name")
        if og_site and og_site.get("content"):
            metadata["source"] = og_site["content"].strip()
        else:
            # Derive from domain
            domain = urlparse(url).netloc.replace("www.", "")
            # Remove TLD and title-case
            name = domain.rsplit(".", 1)[0]
            # Split camelCase/concatenated words by inserting spaces before capitals
            name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
            # Also try to split common concatenated words (e.g. "gemstatechronicle")
            name = name.replace("-", " ").replace("_", " ")
            # Known domain mappings
            domain_map = {
                "gemstatechronicle": "Gem State Chronicle",
                "idahostatesman": "Idaho Statesman",
                "boisedev": "BoiseDev",
                "eastidahonews": "East Idaho News",
                "idahopress": "Idaho Press",
            }
            clean_name = name.lower().replace(" ", "")
            if clean_name in domain_map:
                metadata["source"] = domain_map[clean_name]
            else:
                metadata["source"] = name.title()

        # Author: meta author > article:author > byline patterns > JSON-LD
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta and author_meta.get("content"):
            metadata["author"] = author_meta["content"].strip()
        else:
            article_author = soup.find("meta", property="article:author")
            if article_author and article_author.get("content"):
                metadata["author"] = article_author["content"].strip()
            else:
                # Byline patterns
                for cls in ["author", "byline", "writer", "post-author"]:
                    el = soup.find(class_=re.compile(cls, re.I))
                    if el:
                        text_val = el.get_text(strip=True)
                        # Clean "By " prefix
                        text_val = re.sub(r"^[Bb]y\s+", "", text_val)
                        if 2 < len(text_val) < 80:
                            metadata["author"] = text_val
                            break

        # Published date: article:published_time > JSON-LD > time elements > meta
        pub_date = None
        og_pub = soup.find("meta", property="article:published_time")
        if og_pub and og_pub.get("content"):
            pub_date = og_pub["content"]
        else:
            # JSON-LD
            for script_tag in soup.find_all("script", type="application/ld+json"):
                try:
                    ld = json.loads(script_tag.string)
                    if isinstance(ld, list):
                        ld = ld[0]
                    if isinstance(ld, dict):
                        dp = ld.get("datePublished", "")
                        if dp:
                            pub_date = dp
                            break
                        # Check author from JSON-LD too
                        if not metadata["author"]:
                            a = ld.get("author", {})
                            if isinstance(a, dict):
                                metadata["author"] = a.get("name", "")
                            elif isinstance(a, list) and a:
                                metadata["author"] = a[0].get("name", "")
                except Exception:
                    pass

        if not pub_date:
            # time elements
            time_el = soup.find("time", attrs={"datetime": True})
            if time_el:
                pub_date = time_el["datetime"]

        if not pub_date:
            # Meta publish-date
            for name_attr in ["publish-date", "date", "pubdate"]:
                m = soup.find("meta", attrs={"name": name_attr})
                if m and m.get("content"):
                    pub_date = m["content"]
                    break

        if not pub_date:
            # URL pattern: /YYYY/MM/DD/ or /YYYY/MM/
            url_match = re.search(r"/(\d{4})/(\d{2})(?:/(\d{2}))?/", url)
            if url_match:
                y, m, d = url_match.group(1), url_match.group(2), url_match.group(3) or "01"
                pub_date = f"{y}-{m}-{d}"

        if pub_date:
            metadata["published_date"] = _parse_date(pub_date)

    # Use tavily title as fallback
    if not metadata["title"] and tavily_title:
        metadata["title"] = tavily_title

    # If metadata still missing (403 blocked HTML), extract from Tavily text
    if tavily_text and not metadata["title"]:
        lines = [l.strip() for l in tavily_text.split('\n') if l.strip()]
        # First non-image, non-short line is likely the title
        for line in lines:
            if not line.startswith('!') and not line.startswith('#####') and len(line) > 15:
                # Strip markdown heading markers
                title_candidate = re.sub(r'^#+\s*', '', line).strip()
                if len(title_candidate) > 15 and len(title_candidate) < 200:
                    metadata["title"] = title_candidate
                    break

    if tavily_text and not metadata["source"]:
        # Extract domain as source name
        domain = urlparse(metadata["url"]).netloc.replace('www.', '')
        known = {
            'idahocapitalsun.com': 'Idaho Capital Sun',
            'gemstatechronicle.com': 'Gem State Chronicle',
            'eastidahonews.com': 'East Idaho News',
            'postregister.com': 'Post Register',
            'idahostatesman.com': 'Idaho Statesman',
            'idahofreedom.org': 'Idaho Freedom Foundation',
        }
        metadata["source"] = known.get(domain, domain)

    if tavily_text and not metadata["author"]:
        # Try to find "By: Name" or "By Name" pattern in first 500 chars
        author_match = re.search(r'[Bb]y:?\s+([A-Z][a-z]+ [A-Z][a-z]+)', tavily_text[:500])
        if author_match:
            metadata["author"] = author_match.group(1)

    # Text: prefer tavily, fallback to BS4
    if tavily_text:
        metadata["text"] = tavily_text
    elif soup:
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        content = soup.find("article") or soup.find("main") or soup.find("body")
        if content:
            raw = content.get_text(separator="\n", strip=True)
        else:
            raw = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in raw.split("\n") if line.strip()]
        metadata["text"] = "\n".join(lines)

    metadata["word_count"] = len(metadata["text"].split())

    # Browser fallback: if content seems truncated or from a known paywalled site,
    # try authenticated browser extraction
    browser_script = Path(__file__).parent / "browser-extract.py"
    is_short = metadata["word_count"] < 400
    known_paywall = any(d in url for d in [
        "dailywire.com", "wsj.com", "nytimes.com", "washingtonpost.com",
        "theathletic.com", "bloomberg.com",
    ])
    if browser_script.exists() and (is_short or known_paywall):
        reason = "paywalled site" if known_paywall else f"short content ({metadata['word_count']} words)"
        print(f"  Trying browser extraction ({reason})...")
        try:
            result = subprocess.run(
                ["python3", str(browser_script), url],
                capture_output=True, text=True, timeout=90,
            )
            if result.returncode == 0 and result.stdout.strip():
                browser_data = json.loads(result.stdout)
                browser_wc = browser_data.get("word_count", 0)
                if browser_wc > metadata["word_count"]:
                    print(f"  Browser got {browser_wc} words (vs {metadata['word_count']} from Tavily)")
                    metadata["text"] = browser_data["text"]
                    metadata["word_count"] = browser_wc
                    if browser_data.get("title") and not metadata["title"]:
                        metadata["title"] = browser_data["title"]
                    if browser_data.get("author") and not metadata["author"]:
                        metadata["author"] = browser_data["author"]
                    if browser_data.get("source") and not metadata["source"]:
                        metadata["source"] = browser_data["source"]
                else:
                    print(f"  Browser got {browser_wc} words (not better), keeping Tavily result")
        except subprocess.TimeoutExpired:
            print("  Browser extraction timed out, using existing content")
        except Exception as e:
            print(f"  Browser extraction failed: {e}")

    return metadata


def _parse_date(date_str: str) -> str:
    """Parse various date formats into 'Month DD, YYYY' format."""
    if not date_str:
        return ""
    # Try ISO formats
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
    ]:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%B %d, %Y").replace(" 0", " ")
        except ValueError:
            continue
    # Handle timezone offset without colon (e.g. +0000)
    cleaned = re.sub(r"(\+\d{2}):(\d{2})$", r"\1\2", date_str.strip())
    for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"]:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%B %d, %Y").replace(" 0", " ")
        except ValueError:
            continue
    # Last resort: return as-is
    return date_str.strip()


def generate_summary(text: str, title: str) -> str:
    """Generate a 2-3 sentence summary using gpt-4o-mini."""
    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this article in 2-3 sentences, focusing on the key point and why it matters:\n\n{text[:3000]}",
                }
            ],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Warning: Summary generation failed: {e}")
        return ""


def build_description(metadata: dict, summary: str) -> str:
    """Build rich episode description (plain text, no HTML/markdown)."""
    parts = []
    if summary:
        parts.append(summary)
    parts.append("")
    parts.append(f"Source: {metadata.get('source', 'Unknown')}")
    parts.append(f"Author: {metadata.get('author') or 'Unknown'}")
    pub = metadata.get("published_date", "")
    parts.append(f"Published: {pub or 'Unknown'}")
    parts.append(f"Converted: {datetime.now().strftime('%B %d, %Y').replace(' 0', ' ')}")
    parts.append(f"Read the original: {metadata['url']}")
    return "\n".join(parts)


def build_episode_title(metadata: dict, duration_minutes: int) -> str:
    """Format: [Source] Title (Xm)"""
    source = metadata.get("source", "Unknown")
    title = metadata.get("title", "Untitled")
    return f"[{source}] {title} ({duration_minutes}m)"


def build_intro_text(metadata: dict) -> str:
    """Short spoken intro: 'From {source}, published {date}. {title}.'"""
    source = metadata.get("source", "Unknown source")
    date = metadata.get("published_date", "")
    title = metadata.get("title", "")
    date_part = f", published {date}" if date else ""
    return f"From {source}{date_part}. {title}."


KILL_TRIGGERS = [
    r'^about$',             # "About" alone as heading (bio section)
    r'about the author',
    r'about \w+ \w+\s*$',  # "About Brian Allman"
    r'related articles',
    r'related posts',
    r'you may also like',
    r'more from',
    r'read more',
    r'share this',
    r'follow us',
    r'subscribe to',
    r'sign up for',
    r'newsletter',
    r'comments?\s*\(',
    r'leave a (reply|comment)',
    r'filed under',
    r'tags?:',
    r'posted in',
]

STRIP_PATTERNS = [
    (r'https?://\S+', ''),                          # URLs
    (r'data:[a-zA-Z0-9/;,+=\-_.]+\S*', ''),        # data: URIs (base64 images, SVGs, etc.)
    (r'!\[.*?\]\(.*?\)', ''),                        # Markdown images
    (r'\[([^\]]+)\]\([^\)]+\)', r'\1'),              # Markdown links → keep text
    (r'\*{1,3}([^*]+)\*{1,3}', r'\1'),              # Bold/italic → keep text
    (r'@\w+', ''),                                   # @mentions
    (r'#\w+', ''),                                   # hashtags
    (r'^\s*[\|]\s*', ''),                            # Table pipes
    (r'\S+@\S+\.\S+', ''),                          # Email addresses
]

BOILERPLATE_LINES = [
    'subscribe', 'newsletter', 'follow us', 'share this', 'tweet',
    'facebook', 'instagram', 'twitter', 'linkedin', 'tiktok',
    'comment', 'related:', 'tags:', 'filed under', 'posted in',
    'cookie', 'javascript', 'ad blocker', 'support our work', 'donate',
    'click here', 'read more', 'load more', 'show more',
    'advertisement', 'sponsored', 'promoted', 'partner content',
    'photo credit', 'image credit', 'caption:', 'credit:',
    'updated:', 'correction:', 'editor\'s note',
    'share', 'save', 'print', 'email this',
    'web,', 'more posts', 'view all', 'see all',
]


def _collapse_whitespace(text: str) -> str:
    """Max 2 consecutive blank lines."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_for_tts(text: str) -> str:
    """Clean text for TTS: kill author bios, strip URLs/boilerplate, remove nav garbage."""
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Kill trigger check — stop processing entirely
        for trigger in KILL_TRIGGERS:
            if re.search(trigger, lower):
                result = '\n'.join(cleaned)
                result = _collapse_whitespace(result)
                # Limit to ~5000 words
                words = result.split()
                if len(words) > 5000:
                    result = ' '.join(words[:5000])
                return result

        # Skip boilerplate lines
        if any(bp in lower for bp in BOILERPLATE_LINES):
            continue

        # Skip very short lines (nav items, labels) unless they end a sentence
        if len(stripped) < 20 and not stripped.endswith(('.', '!', '?', ':')):
            continue

        # Apply strip patterns
        for pattern, replacement in STRIP_PATTERNS:
            stripped = re.sub(pattern, replacement, stripped)

        stripped = stripped.strip()
        if stripped:
            cleaned.append(stripped)

    result = '\n'.join(cleaned)
    result = _collapse_whitespace(result)
    # Limit to ~5000 words
    words = result.split()
    if len(words) > 5000:
        result = ' '.join(words[:5000])
    return result


def split_text(text: str, max_len: int = TTS_CHUNK_LIMIT) -> list[str]:
    """Split text into chunks respecting sentence boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    current = ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        if len(sentence) > max_len:
            words = sentence.split()
            for word in words:
                if len(current) + len(word) + 1 > max_len:
                    if current:
                        chunks.append(current.strip())
                    current = word
                else:
                    current = f"{current} {word}" if current else word
        elif len(current) + len(sentence) + 1 > max_len:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def concat_mp3_files(input_paths: list, output_path: Path) -> None:
    """Concatenate MP3 files using ffmpeg if available, else raw concat."""
    if shutil.which('ffmpeg'):
        list_file = output_path.parent / '_concat_list.txt'
        try:
            with open(list_file, 'w') as f:
                for p in input_paths:
                    f.write(f"file '{p}'\n")
            subprocess.run(
                ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(list_file),
                 '-c', 'copy', str(output_path), '-y', '-loglevel', 'error'],
                check=True
            )
        finally:
            list_file.unlink(missing_ok=True)
    else:
        # Fallback: raw concat (works for most MP3s from same source)
        with open(output_path, 'wb') as out:
            for p in input_paths:
                with open(p, 'rb') as f:
                    out.write(f.read())


def text_to_mp3(text: str, output_path: Path) -> None:
    """Convert text to MP3 using OpenAI TTS, handling chunking."""
    client = OpenAI()
    chunks = split_text(text)

    if len(chunks) == 1:
        response = _retry(
            lambda: client.audio.speech.create(
                model=TTS_MODEL, voice=TTS_VOICE, input=chunks[0]
            ),
            label="TTS"
        )
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        return

    temp_files = []
    try:
        for i, chunk in enumerate(chunks):
            print(f"  Generating audio chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            response = _retry(
                lambda c=chunk: client.audio.speech.create(
                    model=TTS_MODEL, voice=TTS_VOICE, input=c
                ),
                label=f"TTS chunk {i+1}"
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            with open(tmp.name, "wb") as f:
                for audio_chunk in response.iter_bytes():
                    f.write(audio_chunk)
            temp_files.append(tmp.name)

        concat_mp3_files(temp_files, output_path)
    finally:
        for tmp_path in temp_files:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def generate_intro_audio(intro_text: str, main_mp3: Path) -> None:
    """Generate intro TTS and prepend to main MP3."""
    client = OpenAI()
    intro_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    combined_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        response = _retry(
            lambda: client.audio.speech.create(
                model=TTS_MODEL, voice=TTS_VOICE, input=intro_text
            ),
            label="TTS intro"
        )
        with open(intro_tmp.name, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

        # Concat intro + main into temp, then move to main_mp3
        concat_mp3_files([intro_tmp.name, str(main_mp3)], Path(combined_tmp.name))
        shutil.move(combined_tmp.name, str(main_mp3))
    finally:
        for tmp in [intro_tmp.name, combined_tmp.name]:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def get_mp3_duration(mp3_path: Path) -> int:
    """Get MP3 duration in seconds using mutagen. Falls back to file size estimate."""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(str(mp3_path))
        return int(audio.info.length)
    except Exception as e:
        print(f"  ⚠️ mutagen failed ({e}), estimating from file size...")
        # Estimate: ~16KB/s for 128kbps MP3
        size = mp3_path.stat().st_size
        return max(1, size // 16000)


def generate_webvtt(mp3_path: Path) -> Path:
    """Transcribe MP3 via Whisper and generate WebVTT with word-level timestamps."""
    client = OpenAI()
    vtt_path = mp3_path.with_suffix(".vtt")

    print("  Transcribing with Whisper (word-level)...")

    def _whisper_call():
        with open(mp3_path, "rb") as f:
            return client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

    transcript = _retry(_whisper_call, label="Whisper")

    # Build cues from word-level timestamps: max ~32 chars or ~5 seconds per cue
    words = transcript.words if hasattr(transcript, 'words') and transcript.words else []

    if not words:
        # Fallback to segment-level if no words returned
        print("  ⚠️ No word-level timestamps, falling back to segments...")
        lines = ["WEBVTT", ""]
        for i, seg in enumerate(getattr(transcript, 'segments', []) or [], 1):
            if isinstance(seg, dict):
                start = format_vtt_timestamp(seg["start"])
                end = format_vtt_timestamp(seg["end"])
                text = seg["text"].strip()
            else:
                start = format_vtt_timestamp(seg.start)
                end = format_vtt_timestamp(seg.end)
                text = seg.text.strip()
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        vtt_path.write_text("\n".join(lines))
        return vtt_path

    # Group words into cues of max ~32 chars or ~5 seconds
    MAX_CUE_CHARS = 32
    MAX_CUE_DURATION = 5.0

    lines = ["WEBVTT", ""]
    cue_num = 0
    cue_words = []
    cue_chars = 0
    cue_start = None

    for w in words:
        word_text = w.word if hasattr(w, 'word') else w.get('word', '')
        word_start = w.start if hasattr(w, 'start') else w.get('start', 0)
        word_end = w.end if hasattr(w, 'end') else w.get('end', 0)
        word_text = word_text.strip()
        if not word_text:
            continue

        new_len = cue_chars + len(word_text) + (1 if cue_words else 0)
        duration = (word_end - cue_start) if cue_start is not None else 0

        if cue_words and (new_len > MAX_CUE_CHARS or duration > MAX_CUE_DURATION):
            # Flush current cue
            cue_num += 1
            last_w = cue_words[-1]
            last_end = last_w[2]
            lines.append(f"{cue_num}")
            lines.append(f"{format_vtt_timestamp(cue_start)} --> {format_vtt_timestamp(last_end)}")
            lines.append(" ".join(t for _, _, _, t in cue_words))
            lines.append("")
            cue_words = []
            cue_chars = 0
            cue_start = None

        if cue_start is None:
            cue_start = word_start
        cue_words.append((word_start, word_end, word_end, word_text))
        cue_chars += len(word_text) + (1 if len(cue_words) > 1 else 0)

    # Flush remaining
    if cue_words:
        cue_num += 1
        last_end = cue_words[-1][2]
        lines.append(f"{cue_num}")
        lines.append(f"{format_vtt_timestamp(cue_start)} --> {format_vtt_timestamp(last_end)}")
        lines.append(" ".join(t for _, _, _, t in cue_words))
        lines.append("")

    vtt_path.write_text("\n".join(lines))
    return vtt_path


# _format_vtt_time removed — use format_vtt_timestamp() at module level instead


def detect_chapters(text: str) -> list[dict]:
    """Detect section breaks in text and generate chapter markers."""
    chapters = [{"startTime": 0, "title": "Introduction"}]
    words_so_far = 0
    lines = text.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        is_heading = False
        title = stripped

        # Markdown ## headers
        if stripped.startswith("##"):
            is_heading = True
            title = stripped.lstrip("#").strip()
        # ALL CAPS lines (at least 3 words, not too long)
        elif (
            stripped.isupper()
            and len(stripped.split()) >= 2
            and len(stripped) < 100
        ):
            is_heading = True
        # Short line followed by blank line (possible section header)
        elif (
            len(stripped.split()) <= 8
            and len(stripped) < 80
            and i + 1 < len(lines)
            and not lines[i + 1].strip()
            and not stripped.endswith(".")
            and words_so_far > 50
        ):
            is_heading = True

        if is_heading and title and words_so_far > 30:
            start_seconds = int((words_so_far / WPM) * 60)
            chapters.append({"startTime": start_seconds, "title": title[:80]})

        words_so_far += len(stripped.split())

    return chapters if len(chapters) > 1 else []


def save_chapters(chapters: list[dict], chapters_path: Path):
    """Save chapters in Podcasting 2.0 JSON format."""
    data = {"version": "1.2.0", "chapters": chapters}
    chapters_path.write_text(json.dumps(data, indent=2))


def generate_artwork():
    """Generate feed artwork with DALL-E 3 if not present."""
    if ARTWORK_PATH.exists():
        print("  Artwork already exists, skipping generation")
        return

    print("  Generating feed artwork with DALL-E 3...")
    client = OpenAI()
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt="Minimalist podcast cover art: dark navy background, single white headphone icon centered, small text 'PRIVATE FEED' below in clean sans-serif. No faces, no people, no text other than 'PRIVATE FEED'. Professional, clean, modern.",
            size="1024x1024",
            quality="standard",
        )
        image_url = response.data[0].url
        img_resp = requests.get(image_url, timeout=60)
        img_resp.raise_for_status()
        ARTWORK_PATH.write_bytes(img_resp.content)
        print(f"  Saved artwork to {ARTWORK_PATH}")
    except Exception as e:
        print(f"  Warning: Artwork generation failed: {e}")


def init_repo():
    """Initialize the git repo and base files if needed."""
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    EPISODES_DIR.mkdir(exist_ok=True)

    if not (REPO_DIR / ".git").exists():
        subprocess.run(["git", "init"], cwd=REPO_DIR, capture_output=True)
        subprocess.run(
            ["git", "config", "user.name", GIT_AUTHOR_NAME],
            cwd=REPO_DIR,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", GIT_AUTHOR_EMAIL],
            cwd=REPO_DIR,
            capture_output=True,
        )

    if not FEED_FILE.exists():
        create_initial_feed()

    if not INDEX_FILE.exists():
        create_index_html()

    nojekyll = REPO_DIR / ".nojekyll"
    if not nojekyll.exists():
        nojekyll.touch()

    # Generate artwork if missing
    generate_artwork()


def create_initial_feed():
    """Create the initial RSS feed XML."""
    now = formatdate(localtime=True)
    artwork_url = f"{BASE_URL}/artwork.jpg"
    feed_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:atom="http://www.w3.org/2005/Atom"
     xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Claw Feed</title>
    <link>{BASE_URL}</link>
    <description>Claw Feed — private audio.</description>
    <language>en-us</language>
    <lastBuildDate>{now}</lastBuildDate>
    <atom:link href="{FEED_URL}" rel="self" type="application/rss+xml"/>
    <itunes:author>Private</itunes:author>
    <itunes:owner>
      <itunes:name>Private</itunes:name>
    </itunes:owner>
    <itunes:explicit>false</itunes:explicit>
    <itunes:category text="Technology"/>
    <itunes:type>episodic</itunes:type>
    <itunes:block>Yes</itunes:block>
    <podcast:locked>yes</podcast:locked>
    <itunes:image href="{artwork_url}"/>
    <image>
      <url>{artwork_url}</url>
      <title>Claw Feed</title>
      <link>{BASE_URL}</link>
    </image>
  </channel>
</rss>"""
    atomic_write(FEED_FILE, feed_xml)


def create_index_html():
    """Create a simple landing page."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claw Feed</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #e0e0e0;
        }}
        h1 {{ color: #ff6b35; }}
        a {{ color: #4ecdc4; }}
        .feed-url {{
            background: #16213e;
            padding: 12px;
            border-radius: 6px;
            font-family: monospace;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <h1>🎧 Claw Feed</h1>
    <p>Articles converted to audio, served as a private podcast.</p>
    <h2>Subscribe</h2>
    <p>Add this feed URL to your podcast app:</p>
    <div class="feed-url">{FEED_URL}</div>
</body>
</html>"""
    INDEX_FILE.write_text(html)


def add_episode_to_feed(
    title: str,
    description: str,
    filename: str,
    file_size: int,
    duration_seconds: int,
    vtt_filename: str = "",
    chapters_filename: str = "",
    author: str = "",
    art_filename: str = "",
):
    """Add a new episode entry to the RSS feed using string manipulation for reliability."""
    now = formatdate(localtime=True)

    # Build the item XML
    encoded_filename = quote(filename)
    episode_url = f"{EPISODES_URL}/{encoded_filename}"
    
    # CDATA-wrap description
    desc_cdata = f"<![CDATA[{description}]]>"
    
    itunes_ns = "http://www.itunes.com/dtds/podcast-1.0.dtd"
    podcast_ns = "https://podcastindex.org/namespace/1.0"

    episode_number = _count_feed_episodes() + 1

    item_lines = [
        "    <item>",
        f"      <title>{_xml_escape(title)}</title>",
        f"      <description>{desc_cdata}</description>",
        f"      <pubDate>{now}</pubDate>",
        f'      <guid isPermaLink="true">{episode_url}</guid>',
        f'      <enclosure url="{episode_url}" length="{file_size}" type="audio/mpeg"/>',
        f"      <itunes:duration>{duration_seconds}</itunes:duration>",
        f"      <itunes:summary>{_xml_escape(truncate_at_sentence(description))}</itunes:summary>",
        f"      <itunes:episodeType>full</itunes:episodeType>",
        f"      <itunes:episode>{episode_number}</itunes:episode>",
    ]

    if author:
        item_lines.append(f"      <itunes:author>{_xml_escape(author)}</itunes:author>")

    if art_filename:
        art_url = f"{EPISODES_URL}/{quote(art_filename)}"
        item_lines.append(f'      <itunes:image href="{art_url}"/>')

    if vtt_filename:
        vtt_url = f"{EPISODES_URL}/{quote(vtt_filename)}"
        item_lines.append(
            f'      <podcast:transcript url="{vtt_url}" type="text/vtt" language="en" rel="captions"/>'
        )

    if chapters_filename:
        ch_url = f"{EPISODES_URL}/{quote(chapters_filename)}"
        item_lines.append(
            f'      <podcast:chapters url="{ch_url}" type="application/json+chapters"/>'
        )

    item_lines.append("    </item>")
    item_xml = "\n".join(item_lines)

    # Read existing feed
    feed_text = FEED_FILE.read_text()

    # Ensure podcast namespace is in the rss tag
    if "xmlns:podcast" not in feed_text:
        feed_text = feed_text.replace(
            'xmlns:atom="http://www.w3.org/2005/Atom"',
            'xmlns:atom="http://www.w3.org/2005/Atom"\n     xmlns:podcast="https://podcastindex.org/namespace/1.0"',
        )

    # Ensure itunes:type is in channel
    if "<itunes:type>" not in feed_text:
        feed_text = feed_text.replace(
            '<itunes:category text="Technology"/>',
            '<itunes:category text="Technology"/>\n    <itunes:type>episodic</itunes:type>',
        )

    # Ensure artwork tags are in channel
    artwork_url = f"{BASE_URL}/artwork.jpg"
    if "<itunes:image" not in feed_text and ARTWORK_PATH.exists():
        feed_text = feed_text.replace(
            '<itunes:category text="Technology"/>',
            f'<itunes:category text="Technology"/>\n    <itunes:image href="{artwork_url}"/>\n    <image>\n      <url>{artwork_url}</url>\n      <title>Claw Feed</title>\n    </image>',
        )

    # Update lastBuildDate
    feed_text = re.sub(
        r"<lastBuildDate>.*?</lastBuildDate>",
        f"<lastBuildDate>{now}</lastBuildDate>",
        feed_text,
    )

    # Insert item before </channel>
    feed_text = feed_text.replace("  </channel>", f"{item_xml}\n  </channel>")

    atomic_write(FEED_FILE, feed_text)


def truncate_at_sentence(text: str, max_len: int = 255) -> str:
    """Truncate text at a sentence boundary, not mid-word."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    # Find last sentence boundary
    for punct in ['. ', '! ', '? ']:
        idx = truncated.rfind(punct)
        if idx > 50:  # Don't truncate too aggressively
            return truncated[:idx + 1]
    return truncated.rstrip() + '...'


def _xml_escape(text: str) -> str:
    """Escape XML special characters."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    return text


def _count_feed_episodes() -> int:
    """Count existing episodes in the feed for episode numbering."""
    if not FEED_FILE.exists():
        return 0
    content = FEED_FILE.read_text()
    return content.count('<item>')


def git_commit_and_push(title: str, files: list[str] | None = None):
    """Stage explicit files, commit, and push. Notifies on push failure."""
    if files:
        subprocess.run(["git", "add"] + files, cwd=REPO_DIR, capture_output=True)
    else:
        subprocess.run(["git", "add", str(FEED_FILE)], cwd=REPO_DIR, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"Add episode: {title}"],
        cwd=REPO_DIR,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "push"], cwd=REPO_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ⚠️ git push failed: {result.stderr}")
        send_telegram(
            f"⚠️ Episode created but git push failed: <b>{_xml_escape(title)}</b>\n"
            f"Manual push needed: cd ~/podcast && git push"
        )
        return False
    return True


def sync_to_obsidian(metadata: dict):
    """Save article text to Obsidian inbox."""
    if not OBS_BIN.exists():
        print("  ⚠️ obs CLI not found, skipping Obsidian sync")
        return

    title = metadata.get("title", "Untitled")
    safe_title = re.sub(r'[/\\:*?"<>|]', "-", title)
    note_path = f"-Inbox/Podcast - {safe_title}.md"

    frontmatter = "---\n"
    frontmatter += f"title: \"{title}\"\n"
    frontmatter += f"source: {metadata['url']}\n"
    if metadata.get("author"):
        frontmatter += f"author: {metadata['author']}\n"
    if metadata.get("published_date"):
        frontmatter += f"published: {metadata['published_date']}\n"
    frontmatter += "tags: [podcast, AI/Claw]\n"
    frontmatter += "---\n\n"
    content = frontmatter + metadata.get("text", "")

    try:
        result = subprocess.run(
            [str(OBS_BIN), "write", note_path, content],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            print(f"  Saved to Obsidian: {note_path}")
        else:
            print(f"  ⚠️ Obsidian sync failed: {result.stderr}")
    except Exception as e:
        print(f"  ⚠️ Obsidian sync error: {e}")


def save_episode_metadata(metadata: dict, episode_info: dict, json_path: Path):
    """Save episode metadata JSON."""
    data = {
        "title": metadata.get("title", ""),
        "source": metadata.get("source", ""),
        "author": metadata.get("author", ""),
        "published": metadata.get("published_date", ""),
        "url": metadata.get("url", ""),
        "summary": episode_info.get("summary", ""),
        "word_count": metadata.get("word_count", 0),
        "duration_seconds": episode_info.get("duration_seconds", 0),
        "mp3": episode_info.get("mp3_filename", ""),
        "vtt": episode_info.get("vtt_filename", ""),
        "chapters": episode_info.get("chapters_filename", ""),
        "created": datetime.now().isoformat(),
    }
    json_path.write_text(json.dumps(data, indent=2))


def list_episodes():
    """List all episodes in the feed."""
    if not FEED_FILE.exists():
        print("No feed found.")
        return
    import xml.etree.ElementTree as ET

    tree = ET.parse(FEED_FILE)
    root = tree.getroot()
    items = root.findall(".//item")
    if not items:
        print("No episodes yet.")
        return
    print(f"📻 Claw Feed — {len(items)} episode(s)\n")
    for item in items:
        title = item.findtext("title", "Untitled")
        date = item.findtext("pubDate", "Unknown date")
        print(f"  • {title}")
        print(f"    {date}\n")


def check_garbage(text: str, url: str) -> bool:
    """Check if extracted text is garbage. Returns True if garbage detected."""
    if len(text) < 200:
        send_telegram(
            f"⚠️ Couldn't extract article from {url} — text too short ({len(text)} chars). Try a different URL."
        )
        return True
    text_lower = text.lower()
    for phrase in GARBAGE_PHRASES:
        if phrase.lower() in text_lower:
            send_telegram(
                f"⚠️ Couldn't extract article from {url} — looks like a paywall or JS-required page. Try a different URL."
            )
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert articles to podcast episodes (v2)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--url", help="URL of article to convert")
    group.add_argument("--text", help="Text content to convert")
    group.add_argument("--file", help="Path to text file to convert")
    group.add_argument("--list", action="store_true", help="List all episodes")

    parser.add_argument("--title", help="Override episode title")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no generation")

    args = parser.parse_args()

    if args.list:
        list_episodes()
        return

    if not args.url and not args.text and not args.file:
        parser.print_help()
        sys.exit(1)

    # --- Extract content ---
    metadata = {
        "title": "",
        "author": "",
        "source": "",
        "published_date": "",
        "url": args.url or "",
        "text": "",
        "word_count": 0,
    }

    if args.url:
        # Dedup check — don't re-convert same URL
        episodes_dir = REPO_DIR / "episodes"
        existing_urls = []
        for jf in episodes_dir.glob("*.json"):
            try:
                import json as _json
                d = _json.loads(jf.read_text())
                if d.get("url"):
                    existing_urls.append(d["url"])
            except Exception:
                pass
        if args.url in existing_urls:
            print(f"⏭️  Already converted: {args.url}")
            print("   Use --force to reconvert (not yet implemented)")
            sys.exit(0)

        print(f"📰 Fetching article from: {args.url}")
        metadata = extract_metadata(args.url)
        if not metadata["text"]:
            print("Error: Could not extract article text.")
            send_telegram(
                f"⚠️ Couldn't extract article from {args.url} — no text found."
            )
            sys.exit(1)
        print(f"  Extracted {len(metadata['text'])} chars, {metadata['word_count']} words")
        print(f"  Title: {metadata['title']}")
        print(f"  Source: {metadata['source']}")
        print(f"  Author: {metadata['author'] or 'Unknown'}")
        print(f"  Published: {metadata['published_date'] or 'Unknown'}")

        # Check for garbage
        if check_garbage(metadata["text"], args.url):
            sys.exit(1)

    elif args.text:
        metadata["text"] = args.text
        metadata["title"] = args.text[:80].split("\n")[0]
        metadata["word_count"] = len(args.text.split())
    elif args.file:
        metadata["text"] = Path(args.file).read_text()
        metadata["title"] = Path(args.file).stem.replace("-", " ").replace("_", " ").title()
        metadata["word_count"] = len(metadata["text"].split())

    if args.title:
        metadata["title"] = args.title

    if not metadata["title"]:
        metadata["title"] = "Untitled Episode"

    # Estimate duration
    est_duration_min = max(1, round(metadata["word_count"] / WPM))

    # Build episode title
    if metadata.get("source"):
        episode_title = build_episode_title(metadata, est_duration_min)
    else:
        episode_title = f"{metadata['title']} ({est_duration_min}m)"

    print(f"\n🎙️  Episode: {episode_title}")
    print(f"📝 {metadata['word_count']} words, ~{est_duration_min} min estimated")

    if args.dry_run:
        print("\n[Dry run — no audio generated]")
        summary = generate_summary(metadata["text"], metadata["title"])
        desc = build_description(metadata, summary)
        print(f"\nDescription:\n{desc}")
        return

    # --- Generate episode ---
    init_repo()

    # Generate summary
    print("\n📝 Generating summary...")
    summary = generate_summary(metadata["text"], metadata["title"])
    if summary:
        print(f"  {summary[:120]}...")

    # Build description
    description = build_description(metadata, summary)

    # Clean text for TTS
    clean_text = clean_for_tts(metadata["text"])

    # Build spoken intro
    intro_text = build_intro_text(metadata)
    print(f"🎤 Intro: {intro_text}")

    # Generate filenames
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(metadata["title"])
    mp3_filename = f"{date_str}-{slug}.mp3"
    vtt_filename = f"{date_str}-{slug}.vtt"
    chapters_filename = f"{date_str}-{slug}-chapters.json"
    json_filename = f"{date_str}-{slug}.json"

    mp3_path = EPISODES_DIR / mp3_filename
    vtt_path = EPISODES_DIR / vtt_filename
    chapters_path = EPISODES_DIR / chapters_filename
    json_path = EPISODES_DIR / json_filename

    # Generate main audio
    print(f"\n🔊 Generating audio ({len(split_text(clean_text))} chunks)...")
    text_to_mp3(clean_text, mp3_path)
    print(f"  Main audio: {mp3_path.stat().st_size / 1024:.1f} KB")

    # Prepend intro
    print("🎤 Prepending spoken intro...")
    generate_intro_audio(intro_text, mp3_path)

    # Get actual duration
    duration_seconds = get_mp3_duration(mp3_path)
    duration_minutes = max(1, round(duration_seconds / 60))
    file_size = mp3_path.stat().st_size
    print(f"  Final audio: {file_size / 1024:.1f} KB, {duration_seconds}s ({duration_minutes}m)")

    # Update episode title with actual duration
    if metadata.get("source"):
        episode_title = build_episode_title(metadata, duration_minutes)
    else:
        episode_title = f"{metadata['title']} ({duration_minutes}m)"

    # Generate WebVTT
    print("📜 Generating WebVTT transcript...")
    try:
        generate_webvtt(mp3_path)
        print(f"  Saved: {vtt_path}")
    except Exception as e:
        print(f"  ⚠️ WebVTT generation failed: {e}")
        vtt_filename = ""

    # Generate episode artwork
    art_filename = f"{date_str}-{slug}-art.jpg"
    art_path = EPISODES_DIR / art_filename
    print("🎨 Generating episode artwork...")
    art_ok = generate_episode_art(
        metadata.get("source", "Unknown"),
        metadata.get("title", "Untitled"),
        art_path,
    )
    if not art_ok:
        art_filename = ""

    # Chapter markers for long articles
    actual_chapters_filename = ""
    if metadata["word_count"] > 2000:
        print("📑 Generating chapter markers...")
        chapters = detect_chapters(metadata["text"])
        if chapters:
            save_chapters(chapters, chapters_path)
            actual_chapters_filename = chapters_filename
            print(f"  {len(chapters)} chapters saved")
        else:
            print("  No clear sections detected")
    
    # Update RSS feed
    print("📻 Updating RSS feed...")
    add_episode_to_feed(
        title=episode_title,
        description=description,
        filename=mp3_filename,
        file_size=file_size,
        duration_seconds=duration_seconds,
        vtt_filename=vtt_filename,
        chapters_filename=actual_chapters_filename,
        author=metadata.get("author", ""),
        art_filename=art_filename,
    )

    # Save episode metadata JSON
    print("💾 Saving metadata...")
    save_episode_metadata(
        metadata,
        {
            "summary": summary,
            "duration_seconds": duration_seconds,
            "mp3_filename": mp3_filename,
            "vtt_filename": vtt_filename,
            "chapters_filename": actual_chapters_filename,
        },
        json_path,
    )

    # Commit and push — explicit file list
    print("📤 Committing and pushing...")
    git_files = [
        str(FEED_FILE),
        str(mp3_path),
        str(json_path),
    ]
    if vtt_filename:
        git_files.append(str(EPISODES_DIR / vtt_filename))
    if actual_chapters_filename:
        git_files.append(str(EPISODES_DIR / actual_chapters_filename))
    if art_filename:
        git_files.append(str(art_path))
    pushed = git_commit_and_push(episode_title, files=git_files)

    # Obsidian sync
    print("📓 Syncing to Obsidian...")
    sync_to_obsidian(metadata)

    # Telegram notification
    if pushed:
        pub_date = metadata.get("published_date", "Unknown")
        source = metadata.get("source", "Unknown")
        msg = (
            f"🎙️ New episode ready: {episode_title}\n"
            f"⏱️ {duration_minutes}m · {source}\n"
            f"Published: {pub_date}\n"
            f"Feed: {FEED_URL}"
        )
        print("📱 Sending Telegram notification...")
        send_telegram(msg)

    print(f"\n✅ Episode created: {episode_title}")
    print(f"   File: {mp3_path}")
    print(f"   Duration: {duration_minutes}m ({duration_seconds}s)")
    if pushed:
        print(f"   URL: {EPISODES_URL}/{quote(mp3_filename)}")
    print(f"   Feed: {FEED_URL}")


if __name__ == "__main__":
    main()
