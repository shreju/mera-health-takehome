"""Extract structured lab results from a lab report.

Usage:
    python extract.py reports/report_01.pdf

Prints JSON matching schema.json to stdout.
"""
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
DPI = int(os.environ.get("EXTRACTION_DPI", "150"))

PROMPT = """You are an expert clinical medical laboratory data extraction engine.
Extract all diagnostic test results printed on this lab report into structured JSON matching the target schema.

Return a JSON object:
{"tests": [{"test_name": "...", "value": "...", "unit": "...", "reference_range": "..."}]}

Rules:
1. Extract every test result row in the exact order printed across all pages.
2. test_name: Use the exact test parameter name printed on the report. Do not paraphrase or expand acronyms.
3. value: Extract the exact result value as printed. Can be numeric (e.g. 13.5, 9200) or qualitative text (e.g. "Negative", "Present (+)", "Clear", "Nil", "2-3", "< 0.5").
4. unit: Extract the exact unit printed (e.g. "g/dL", "mg/dL", "%", "mmol/L"). If no unit is printed, output null.
5. reference_range: Extract the reference interval exactly as printed on the report (e.g. "13.0 - 17.0", "< 200"). If no reference range is printed for a test, output null. NEVER invent or hallucinate a reference range.
6. Do not include section headers, patient details, or method notes as tests.
"""

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def load_images(path: Path) -> list[tuple[str, str]]:
    """Return list of (media_type, base64 data) for all pages of the report."""
    suffix = path.suffix.lower()
    if suffix in MEDIA_TYPES:
        return [(MEDIA_TYPES[suffix], base64.standard_b64encode(path.read_bytes()).decode())]

    if suffix == ".pdf":
        # Prefer pypdfium2 for cross-platform rendering of all pages
        try:
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(str(path))
            pages = []
            scale = DPI / 72.0
            for page in doc:
                img = page.render(scale=scale).to_pil()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                pages.append(("image/png", base64.standard_b64encode(buf.getvalue()).decode()))
            if pages:
                return pages
        except ImportError:
            pass

        # Fallback to pdftoppm for all pages
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "page"
            try:
                subprocess.run(["pdftoppm", "-r", str(DPI), "-png", str(path), str(prefix)],
                               check=True, capture_output=True)
            except FileNotFoundError:
                raise RuntimeError("Install Poppler (pdftoppm) or pypdfium2; see README.md.") from None
            pages = []
            for png_file in sorted(Path(directory).glob("page-*.png")):
                pages.append(("image/png", base64.standard_b64encode(png_file.read_bytes()).decode()))
            if pages:
                return pages
            single = prefix.with_suffix(".png")
            if single.exists():
                return [("image/png", base64.standard_b64encode(single.read_bytes()).decode())]

    raise ValueError(f"Unsupported file format: {path}")


def call_model(images: list[tuple[str, str]]) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY; see README.md. No API call made.")

    parts = [{"inlineData": {"mimeType": m, "data": d}} for m, d in images]
    parts.append({"text": PROMPT})

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 16000,
            "responseMimeType": "application/json",
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    req_data = json.dumps(body).encode()
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}

    # Resilient calling with exponential backoff on 429/503
    max_retries = 4
    delay = 3.0
    for attempt in range(max_retries):
        request = urllib.request.Request(url, data=req_data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            candidate = result.get("candidates", [{}])[0]
            return "".join(part["text"] for part in candidate.get("content", {}).get("parts", [])
                           if "text" in part and not part.get("thought"))
        except urllib.error.HTTPError as error:
            if error.code in (429, 503) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"Gemini HTTP {error.code}; check API access, model and quota.") from None
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError("Gemini connection failed; check network access.") from None


def parse(text: str) -> list[dict]:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tests" in data:
            return data["tests"]
        if isinstance(data, list):
            return data
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)).get("tests", [])
            except Exception:
                pass
    return []


def clean_val(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    if not s or s.lower() == "null":
        return None
    # Keep as number if numeric
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def clean(tests: list[dict]) -> list[dict]:
    out = []
    for t in tests:
        if not isinstance(t, dict):
            continue
        name = str(t.get("test_name", "")).strip()
        if not name:
            continue
        unit = t.get("unit")
        if unit is not None:
            unit = str(unit).strip()
            if not unit or unit.lower() == "null":
                unit = None
        rng = t.get("reference_range")
        if rng is not None:
            rng = str(rng).strip()
            if not rng or rng.lower() == "null":
                rng = None
        out.append({
            "test_name": name,
            "value": clean_val(t.get("value")),
            "unit": unit,
            "reference_range": rng,
        })
    return out


def extract(path: Path) -> dict:
    images = load_images(Path(path))
    return {"tests": clean(parse(call_model(images)))}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    try:
        print(json.dumps(extract(Path(sys.argv[1])), indent=2, ensure_ascii=False))
    except RuntimeError as error:
        sys.exit(str(error))
