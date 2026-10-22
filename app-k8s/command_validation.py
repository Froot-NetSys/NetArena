"""Allowlist validation for the LLM-generated commands run by the harness.

A command may run only when it parses as a single `kubectl get` or
`kubectl patch networkpolicy`, optionally piped into read-only text
filters, with redirects only to /dev/null or 2>&1. Everything else,
including anything this module does not recognize, is rejected: the
validator fails closed.
"""

import bashlex

ALLOWED_VERBS = ("get", "patch")
PATCH_RESOURCES = ("networkpolicy", "networkpolicies", "netpol", "netpols")
PIPELINE_FILTERS = ("cat", "cut", "grep", "head", "jq", "sort", "tail", "tr", "uniq", "wc")
MARKER_WORDS = ("bash", "sh", "shell", "yaml", "json", "kubectl")
_NULL_REDIRECT = "/dev/null"
_ANSI_C_PLACEHOLDER = "ANSI_C_STRING_0"
_VALUE_FLAGS = ("-n", "--namespace", "--context", "--cluster", "--user", "--kubeconfig", "-p", "--patch", "--patch-file", "--type")


def validate_command(command: str) -> str | None:
    """None when the command may run, else a short reason safe to show the agent."""
    stripped = command.strip()
    if not stripped:
        return "no command found"
    first_line, newline, rest = stripped.partition("\n")
    if newline and first_line.strip().lower() in MARKER_WORDS:
        stripped = rest
    try:
        for tree in bashlex.parse(_replace_ansi_c_strings(stripped)):
            if (reason := _validate_tree(tree)) is not None:
                return reason
    except Exception:
        return "the command could not be parsed as a shell command"
    return None


def _validate_tree(node) -> str | None:
    if node.kind == "command":
        return _validate_segment(node, first=True)
    if node.kind == "pipeline":
        for index, part in enumerate(node.parts):
            if part.kind == "pipe":
                continue
            if (reason := _validate_segment(part, first=index == 0)) is not None:
                return reason
        return None
    if node.kind == "list":
        return "run one command at a time; chaining (';', '&&', '||', '&') is not allowed"
    return f"shell construct '{node.kind}' is not allowed"


def _validate_segment(node, first: bool) -> str | None:
    if node.kind != "command":
        return f"shell construct '{node.kind}' is not allowed"
    words = []
    for part in node.parts:
        if part.kind == "word":
            if part.parts:
                return f"shell expansion is not allowed in the argument '{part.word[:60]}'"
            words.append(part.word)
        elif part.kind == "redirect":
            if reason := _validate_redirect(part):
                return reason
        else:
            return f"shell construct '{part.kind}' is not allowed"
    return _validate_kubectl(words) if first else _validate_filter(words)


def _validate_redirect(redirect) -> str | None:
    if getattr(redirect, "heredoc", None) is not None:
        return "heredocs are not allowed"
    if redirect.type == ">&":
        if redirect.input in (None, 1, 2) and redirect.output == 1:
            return None
        return "redirecting between file descriptors is not allowed"
    if redirect.type not in (">", ">>") or redirect.input not in (None, 1, 2):
        return "redirection is only allowed for stdout and stderr, to /dev/null"
    if getattr(redirect.output, "word", redirect.output) != _NULL_REDIRECT:
        return "redirection is only allowed to /dev/null"
    return None


def _validate_kubectl(words: list[str]) -> str | None:
    if not words or words[0] != "kubectl":
        return "only kubectl commands are allowed"
    if len(words) < 2 or words[1].startswith("-"):
        return "expected a kubectl verb (get, patch) directly after 'kubectl'"
    if words[1] not in ALLOWED_VERBS:
        return f"kubectl '{words[1]}' is not allowed; only kubectl get and kubectl patch"
    if words[1] == "patch" and not _targets_network_policy(words):
        return "kubectl patch must target a networkpolicy"
    return None


def _validate_filter(words: list[str]) -> str | None:
    if not words:
        return "an empty pipeline segment is not allowed"
    if words[0] not in PIPELINE_FILTERS:
        return f"piping into '{words[0]}' is not allowed"
    return None


def _targets_network_policy(words: list[str]) -> bool:
    index = 2
    while index < len(words):
        word = words[index]
        if word in _VALUE_FLAGS:
            index += 2
        elif word.startswith("-"):
            index += 1
        else:
            return word.split("/")[0].split(".")[0] in PATCH_RESOURCES
    return False


def _replace_ansi_c_strings(command: str) -> str:
    """Replace complete $'...' spans with an inert placeholder word: bashlex
    does not track ANSI-C quoting, and the span content cannot mean anything
    at execution time (ANSI-C quoting only decodes backslash escapes)."""
    out = []
    i = 0
    length = len(command)
    while i < length:
        char = command[i]
        if char == "\\":
            out.append(command[i:i + 2])
            i += 2
        elif char == "'":
            end = command.find("'", i + 1)
            if end == -1:
                out.append(command[i:])
                break
            out.append(command[i:end + 1])
            i = end + 1
        elif char == '"':
            j = i + 1
            while j < length and command[j] != '"':
                if command[j] == "\\":
                    j += 1
                j += 1
            out.append(command[i:j + 1])
            i = j + 1
        elif char == "$" and command[i + 1:i + 2] == "'":
            j = i + 2
            end = -1
            while j < length:
                if command[j] == "\\":
                    j += 2
                    continue
                if command[j] == "'":
                    end = j
                    break
                j += 1
            if end == -1:
                out.append(command[i:])
                break
            out.append(_ANSI_C_PLACEHOLDER)
            i = end + 1
        else:
            out.append(char)
            i += 1
    return "".join(out)