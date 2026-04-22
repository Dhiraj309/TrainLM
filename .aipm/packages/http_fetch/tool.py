import urllib.request


def fetch(url: str) -> str:
    """
    Fetch content from a URL using HTTP GET.
    """
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read()

        # Return only first 1000 chars to avoid huge outputs
        return content.decode("utf-8", errors="ignore")[:1000]

    except Exception as e:
        return f"ERROR: {str(e)}"
