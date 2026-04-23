import urllib.request
import re


def _strip_html(html: str) -> str:
    """
    Basic HTML → readable text conversion (no external deps).
    Good enough for MVP.

    Removes:
    - <script>, <style>
    - all tags
    - excessive whitespace
    """

    # Remove script and style blocks
    html = re.sub(r"<script.*?>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style.*?>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)

    # Decode common HTML entities (very basic)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def fetch(url: str) -> str:
    """
    Fetch content from a URL and return readable text.

    Notes:
    - Returns cleaned text (not raw HTML)
    - Output is truncated for safety
    """

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode("utf-8", errors="ignore")

        text = _strip_html(content)

        # Bound output size (important for LLM context)
        return text[:1000]

    except Exception as e:
        return f"ERROR: {str(e)}"
