#!/usr/bin/env python3
"""
comparison/deidentify.py - replace residual identifiers in text before it is sent to a cloud API (D-192 section 4).

The corpus is de-identified, but residual identifiers remain (user folder names in paths, e-mail addresses, host
names). Every text field is passed through the replacements below, in order, and the result is screened again with
the coding packets' own identifier screen (coding/build_direct_packet.py SCREENS) plus the extra patterns used here.
A text is fit to send only when that second screen finds nothing outside the replacement markers.

Replacements:
  shared chat link            -> [shared chat link removed]   (as in the coding packets)
  adopted rules               -> their own replacement         (the 16 October de-identification file; applied to every
                                                                 conversation, not only the one each rule names)
  scp or git remote           -> [url]                         (git@github.com:<user>/<repo> and user@host:path)
  web address                 -> [url], except public documentation domains, which are kept
  e-mail address              -> [email]
  shell prompt user@host      -> [user]@[host]
  ssh/scp target user@        -> [user]@
  user folder in a path       -> /Users/[user], /home/[user], C:\\Users\\[user]
  Windows or Mac host name    -> [host]
  SSH greeting name           -> Hi [user]!
  owner and group in ls -l    -> [user] [group]
  unikey-like token           -> [unikey]
  nine-digit number           -> [sid]
  Australian mobile number    -> [phone]

Conversation scope (D-206, consequence 3; added 2 November 2026 before the reruns):
  host after a replaced user  -> [user]@[host]                  (for example ssh-keygen output "[user]@<host>")
  any token replaced above    -> [redacted]                     wherever else it appears in the same conversation
     (user folder, shell prompt user or host, ssh target, SSH greeting, ls -l owner or group, e-mail user part, host
     after a replaced user; tokens shorter than 3 characters, generic names and the tokens D-206 confirmed as not
     identifying are left alone)

Names written in free text are not screened (as in the coding packets). Nothing here prints text; callers report
counts only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "coding"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import build_direct_packet as packet  # noqa: E402  (SCREENS: the coding packets' identifier screen)
import select_materials_v2 as materials  # noqa: E402  (SHARE_URL)

ADOPTED_RULES = ROOT / "data/maintenance/archive/20261016-coding-release/deidentification-adopted.json"

MARKERS = ("[shared chat link removed]", "[email removed]", "[email]", "[user]", "[host]", "[group]", "[sid]",
           "[unikey]", "[phone]", "[url]", "[redacted]")

# Public documentation and reference sites: links to them carry no student identifier and are kept.
PUBLIC_DOMAINS = (
    "man7.org", "cppreference.com", "cplusplus.com", "gnu.org", "sourceware.org", "opengroup.org", "die.net",
    "archlinux.org", "manpages.ubuntu.com", "manpages.debian.org", "freebsd.org", "kernel.org", "llvm.org",
    "valgrind.org", "git-scm.com", "docs.github.com", "docs.python.org", "cmake.org", "wikipedia.org",
    "stackoverflow.com", "stackexchange.com", "learn.microsoft.com", "developer.apple.com", "open-std.org",
    "iso.org", "ietf.org", "rfc-editor.org", "w3.org", "json.org", "gitlab.com/gitlab-org", "geeksforgeeks.org",
    "tutorialspoint.com", "programiz.com", "w3schools.com", "beej.us",
)

_URL = re.compile(r"https?://[^\s<>\"'`)\]}]+")
_TRAIL = ".,;:!?"
_SCP = re.compile(r"\b[A-Za-z0-9._-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}:[^\s<>\"'`]+")
_PROMPT = re.compile(r"(?m)^(\s*(?:\([\w.-]+\)\s*)?)([A-Za-z0-9._-]+)@([A-Za-z0-9._-]+)(?=[ :][^\n]{0,80}?[%$#>](?: |$))")
_SSH_TARGET = re.compile(r"\b(ssh|scp|sftp|rsync)((?:\s+-\S+(?:\s+\d+)?)*\s+)([A-Za-z0-9._-]+)@(?!\[host\])")
_FOLDER = re.compile(r"((?:[A-Za-z]:[\\/]+Users[\\/]+)|/Users/|/home/)(?!\[user\])([^\\/\s\"'<>:]+)")
_HOST = re.compile(r"\b(?:DESKTOP|LAPTOP)-[A-Z0-9]{5,10}\b|\b[\w-]+-(?:MacBook|MBP|iMac|Mac-mini)(?:-[\w-]+)?(?:\.local)?\b",
                   re.IGNORECASE)
_SSH_GREETING = re.compile(r"Hi (?!\[user\]!)([^!\s]+)! You've successfully authenticated")
_LISTING = re.compile(r"(?m)^([-dlcbps][-rwxsStT@+.]{9,10}\s+\d+\s+)([A-Za-z0-9._-]+)(\s+)([A-Za-z0-9._-]+)(?=\s)")


def _public(url: str) -> bool:
    rest = url.split("://", 1)[1].lower()
    host = rest.split("/", 1)[0].split(":", 1)[0].split("@")[-1]
    for d in PUBLIC_DOMAINS:
        dom, _, path = d.partition("/")
        if host == dom or host.endswith("." + dom):
            if not path or rest.split("/", 1)[-1].startswith(path):
                return True
    return False


def _url_sub(m: re.Match) -> str:
    s = m.group()
    core = s.rstrip(_TRAIL)
    tail = s[len(core):]
    return s if _public(core) else "[url]" + tail


def load_adopted_rules(path: Path = ADOPTED_RULES) -> list[dict]:
    """The adopted conversation-scoped rules of the coding packets (names and personal addresses)."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("status") != "adopted":
        raise ValueError("the de-identification file is not marked adopted")
    rules = []
    for r in spec.get("rules", []):
        rules.append({"kind": r["kind"], "pattern": re.compile(r["pattern"], re.IGNORECASE),
                      "replacement": r["replacement"]})
    return rules


def replacements(rules: list[dict]) -> list[tuple[str, re.Pattern, object]]:
    """Ordered (name, pattern, replacement) passes."""
    passes: list[tuple[str, re.Pattern, object]] = [
        ("shared chat link", materials.SHARE_URL, "[shared chat link removed]")]
    passes += [(f"adopted rule: {r['kind']}", r["pattern"], r["replacement"]) for r in rules]
    passes += [
        ("scp or git remote", _SCP, "[url]"),
        ("web address", _URL, _url_sub),
        ("e-mail address", packet.SCREENS["email address"], "[email]"),
        ("shell prompt user@host", _PROMPT, r"\1[user]@[host]"),
        ("ssh target user", _SSH_TARGET, r"\1\2[user]@"),
        ("user folder name", _FOLDER, r"\1[user]"),
        ("host name", _HOST, "[host]"),
        ("SSH greeting name", _SSH_GREETING, "Hi [user]! You've successfully authenticated"),
        ("owner and group in listing", _LISTING, r"\1[user]\3[group]"),
        ("unikey-like token", packet.SCREENS["unikey-like token"], "[unikey]"),
        ("nine-digit number", packet.SCREENS["nine-digit number"], "[sid]"),
        ("Australian mobile number", packet.SCREENS["Australian mobile number"], "[phone]"),
    ]
    return passes


def redact(text: str, passes) -> tuple[str, Counter]:
    counts: Counter = Counter()
    for name, pattern, repl in passes:
        new, n = pattern.subn(repl, text)
        if name == "web address":
            n = sum(1 for m in pattern.finditer(text) if not _public(m.group().rstrip(_TRAIL)))
        if n:
            counts[name] += n
        text = new
    return text, counts


def _is_marker(value: str) -> bool:
    return any(mk in value for mk in MARKERS)


def residual(text: str, rules: list[dict]) -> Counter:
    """Second screen on redacted text. Every count here must be zero before anything is sent."""
    hits: Counter = Counter()
    for kind, pattern in packet.SCREENS.items():
        for m in pattern.finditer(text):
            if kind == "other web address":
                full = _URL.match(text, m.start())
                if full is None or not _public(full.group().rstrip(_TRAIL)):
                    hits[kind] += 1
                continue
            value = m.group(1) if m.groups() else m.group()
            if not _is_marker(value):
                hits[kind] += 1
    for r in rules:
        hits["adopted rule: " + r["kind"]] += sum(1 for m in r["pattern"].finditer(text) if not _is_marker(m.group()))
    for name, pattern in (("host name", _HOST), ("shared chat link", materials.SHARE_URL), ("scp or git remote", _SCP)):
        hits[name] += sum(1 for _ in pattern.finditer(text))
    return +hits


# ----------------------------------------------------------------------------- conversation scope (D-206)

GENERIC = {"root", "user", "users", "ubuntu", "admin", "git", "localhost", "student", "pi", "vscode", "codespace",
           "codespaces", "runner", "node", "host", "example", "username", "yourname", "me", "test", "name", "home",
           "desktop", "documents", "downloads", "shared", "public", "default", "guest"}
# Confirmed in D-206 as not identifying (path components, the macOS default group, the default Multipass machine, the
# Pro Git documentation's example user).
NOT_IDENTIFYING = {"src", "libs", "unit.txt", "markdown_unit.c", "staff", "primary", "schacon"}
_AFTER_USER = re.compile(r"\[user\]@(?!\[)([A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]|[A-Za-z0-9])")


def _scoped(token: str) -> bool:
    t = token.lower()
    return len(t) >= 3 and not t.startswith("[") and t not in GENERIC and t not in NOT_IDENTIFYING


def replaced_tokens(text: str) -> set[str]:
    """Tokens that the replacement passes take out of this text and that are worth replacing elsewhere."""
    out = set()
    for pat, groups in ((_FOLDER, (2,)), (_PROMPT, (2, 3)), (_SSH_TARGET, (3,)), (_SSH_GREETING, (1,)),
                        (_LISTING, (2, 4))):
        for m in pat.finditer(text):
            out.update(m.group(g) for g in groups)
    for m in packet.SCREENS["email address"].finditer(text):
        out.add(m.group().split("@")[0])
    return {t for t in out if _scoped(t)}


def conversation_tokens(texts, passes) -> set[str]:
    """Every token replaced in any text of one conversation, including hosts written after a replaced user."""
    tokens = set()
    for t in texts:
        tokens |= replaced_tokens(t)
        red, _ = redact(t, passes)
        tokens |= {m.group(1) for m in _AFTER_USER.finditer(red) if _scoped(m.group(1))}
    return tokens


def _token_pattern(tokens):
    if not tokens:
        return None
    alts = sorted({t.lower() for t in tokens}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(t) for t in alts) + r")(?![A-Za-z0-9])",
                      re.IGNORECASE)


def redact_scoped(text: str, passes, tokens) -> tuple[str, Counter]:
    """redact(), then the two conversation-scope rules."""
    new, counts = redact(text, passes)
    n = 0

    def host(m):
        nonlocal n
        if _scoped(m.group(1)):
            n += 1
            return "[user]@[host]"
        return m.group()
    new = _AFTER_USER.sub(host, new)
    if n:
        counts["host after replaced user"] += n
    pat = _token_pattern(tokens)
    if pat is not None:
        new, k = pat.subn("[redacted]", new)
        if k:
            counts["conversation token"] += k
    return new, counts


def scoped_residual(text: str, tokens) -> int:
    """Occurrences of the conversation's replaced tokens, and of hosts after a replaced user, left in a text."""
    pat = _token_pattern(tokens)
    left = len(pat.findall(text)) if pat is not None else 0
    return left + sum(1 for m in _AFTER_USER.finditer(text) if _scoped(m.group(1)))
