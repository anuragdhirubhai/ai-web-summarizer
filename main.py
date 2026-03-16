import re
import json
import requests
import argparse
import os
from bs4 import BeautifulSoup
import validators


OLLAMA_URL = os.getenv("OLLAMA_URL","http://ollama:11434/api/generate")


# ---------------- Fetch & Clean Webpage ----------------

def fetch_clean(url: str, timeout: int = 20) -> tuple[str, str]:

    r = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0 (AI-Web-Summarizer)",
            "Accept-Language": "en"
        },
    )

    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    title = (soup.title.string or "").strip() if soup.title else ""

    # remove noisy tags
    for tag in soup([
        "footer", "header", "script", "style", "nav", "aside",
        "noscript", "form", "svg", "img", "video", "audio",
        "iframe", "canvas"
    ]):
        tag.decompose()

    parts = []

    for t in soup.find_all(["h1", "h2", "h3", "p", "li", "blockquote"]):

        txt = t.get_text(" ", strip=True)

        if txt and len(txt.split()) >= 3:
            parts.append(txt)

    text = re.sub(r"\s+", " ", "\n".join(parts)).strip()

    if not text:
        raise RuntimeError("Could not extract readable text")

    return title, text


# ---------------- Chunking ----------------

def chunks(text: str, n: int = 4000, overlap: int = 300):

    if len(text) <= n:
        yield text
        return

    i = 0

    while i < len(text):

        seg = text[i:i+n]

        yield seg

        i += n - overlap


# ---------------- JSON Parsing ----------------

def parse_json(s: str) -> dict:

    try:
        return json.loads(s)

    except Exception:

        match = re.search(r"\{.*\}", s, re.S)

        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        return {
            "abstract": s.strip(),
            "bullets": []
        }


# ---------------- LLM call ----------------

def summarize_block(text: str, model="phi3:mini", bullets=6, max_words=300) -> dict:

    prompt = f"""
You are a precise neutral summarizer.

Return STRICT JSON with keys:
abstract
bullets

Format:
{{
"abstract": "<= {max_words} words",
"bullets": ["up to {bullets} key points"]
}}

CONTENT:
{text[:4000]}
"""

    try:

        response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx": 2048,
                "num_thread": 6
            }
        },
        stream=True
    )
        

        response.raise_for_status()

        data = response.json()

        return parse_json(data["response"])

    except Exception as e:

        return {
            "abstract": f"Model error occurred: {str(e)}",
            "bullets": []
        }
#------------------stream summary----------------
def summarize_block(text: str, model="phi3:mini", bullets=6, max_words=300) -> dict:

    prompt = f"""
You are a precise neutral summarizer.

Return STRICT JSON with keys:
abstract
bullets

Format:
{{
"abstract": "<= {max_words} words",
"bullets": ["up to {bullets} key points"]
}}

CONTENT:
{text[:2000]}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 2048,
                "num_thread": 6
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return parse_json(data["response"])
#-------------------frequency based extractions-------------
def extract_key_sentences(text, max_sentences=10):

    sentences = re.split(r'(?<=[.!?]) +', text)

    word_freq = {}

    words = re.findall(r'\w+', text.lower())

    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

    sentence_scores = {}

    for s in sentences:
        for w in re.findall(r'\w+', s.lower()):
            if w in word_freq:
                sentence_scores[s] = sentence_scores.get(s, 0) + word_freq[w]

    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    return " ".join(ranked[:max_sentences])

# ---------------- URL Summarization ----------------

def summarize_url(url: str, model="phi3:mini", bullets=6, max_words=300) -> dict:

    title, text = fetch_clean(url)

    # prevent huge pages
    text = text[:20000]
    # extract important sentences first
    text = extract_key_sentences(text, max_sentences=12)

    sections = [
        summarize_block(
            c,
            model=model,
            bullets=min(7, bullets),
            max_words=min(220, max_words)
        )
        for c in chunks(text)
    ]

    merged_abs = " ".join(
        s.get("abstract", "") for s in sections
    )[:10000]

    merged_bul = [
        b for s in sections for b in s.get("bullets", [])
    ][:bullets]

    final = summarize_block(
        merged_abs,
        model=model,
        bullets=bullets,
        max_words=max_words
    )

    final.setdefault("abstract", merged_abs[:max_words*8])
    final["bullets"] = final.get("bullets", []) or merged_bul
    final["title"] = title
    final["url"] = url

    return final


# ------------- CLI ---------

def main():

    p = argparse.ArgumentParser(description="AI Web Page Summarizer (Ollama + BS4)")

    p.add_argument("--url", required=True)

    p.add_argument(
        "--model",
        default="phi3:mini",
        help="Model to use (phi3:mini / gemma3:12b / deepseek-r1)"
    )

    p.add_argument("--bullets", type=int, default=6)

    p.add_argument("--max-words", type=int, default=300)

    args = p.parse_args()

    try:

        res = summarize_url(
            args.url,
            model=args.model,
            bullets=args.bullets,
            max_words=args.max_words
        )

        print(json.dumps(res, ensure_ascii=False, indent=2))

    except requests.HTTPError:
        raise SystemExit("HTTP error")

    except Exception as e:
        raise SystemExit(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()