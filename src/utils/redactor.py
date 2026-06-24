import re

# api_key = "...", "secret_token": '...', DB_PASSWORD="...", etc.
API_KEY_ASSIGNMENT = re.compile(
    r'(?i)((?:api[_-]?key|secret|token|password|auth|passwd|credential|private[_-]?key)\s*[:=]\s*["\'])([a-zA-Z0-9_\-\.\=\+\/]{8,128})(["\'])'
)

# Authorization: Bearer sk-12345...
BEARER_TOKEN = re.compile(r"(?i)(bearer\s+)([a-zA-Z0-9_\-\.\=\+\/]{12,128})")

# standard Email address structures
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

# Network IP Addresses (IPv4 & standard IPv6)
IP_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    r"|"
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
)


def redact_python_text(text: str) -> str:
    """
    Scans and redacts ONLY high-risk API keys, tokens, and credentials.
    """
    text = API_KEY_ASSIGNMENT.sub(r"\1[REDACTED_SECRET]\3", text)
    text = BEARER_TOKEN.sub(r"\1[REDACTED_SECRET]", text)
    return text


def redact_markdown_text(text: str) -> str:
    """
    Scans and aggressively redacts API keys, tokens, emails, and IP addresses.
    """
    text = API_KEY_ASSIGNMENT.sub(r"\1[REDACTED_SECRET]\3", text)
    text = BEARER_TOKEN.sub(r"\1[REDACTED_SECRET]", text)
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = IP_PATTERN.sub("[REDACTED_IP]", text)
    return text
