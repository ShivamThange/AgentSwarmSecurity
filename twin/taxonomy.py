from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RiskType:
    risk_id: str
    key: str
    name: str
    owasp_ref: str
    tier: int
    description: str

RT1 = [
    RiskType("RT1.1", "prompt_injection", "Prompt Injection", "LLM01", 1,
             "Manipulating agent behavior via malicious input"),
    RiskType("RT1.2", "jailbreak", "Jailbreak Attack", "LLM01", 1,
             "Bypassing safety guidelines and ethical constraints"),
    RiskType("RT1.3", "sensitive_disclosure", "Sensitive Info Disclosure", "LLM02", 1,
             "Leaking system prompts, API keys, or private data"),
    RiskType("RT1.4", "excessive_agency", "Excessive Agency", "LLM06", 1,
             "Executing actions beyond intended scope"),
    RiskType("RT1.5", "unauthorized_code_exec", "Unauthorized Code Execution", "ASI05", 1,
             "Running malicious code or commands"),
    RiskType("RT1.6", "hallucination", "Hallucination", "LLM09", 1,
             "Fabricating false information"),
    RiskType("RT1.7", "memory_poisoning", "Memory Poisoning", "ASI06", 1,
             "Injecting malicious content into agent memory"),
    RiskType("RT1.8", "tool_misuse", "Tool Misuse", "ASI02", 1,
             "Improper use of external tools or APIs"),
]

RT2 = [
    RiskType("RT2.1", "malicious_propagation", "Malicious Propagation", "ASI07", 2,
             "Malicious instructions spreading through the agent network"),
    RiskType("RT2.2", "misinformation_amplification", "Misinformation Amplification", "LLM09", 2,
             "Errors amplified and reinforced across agents"),
    RiskType("RT2.3", "insecure_output_handling", "Insecure Output Handling", "LLM05", 2,
             "Unvalidated upstream output consumed by a downstream agent"),
    RiskType("RT2.4", "goal_drift", "Goal Drift", "ASI01", 2,
             "Deviation from the original objective over multi-turn interaction"),
    RiskType("RT2.5", "message_tampering", "Message Tampering", "ASI07", 2,
             "Message content modified during transmission"),
    RiskType("RT2.6", "identity_spoofing", "Identity Spoofing", "ASI03", 2,
             "Impersonation of a trusted agent"),
]

RT3 = [
    RiskType("RT3.1", "cascading_failure", "Cascading Failure", "ASI08", 3,
             "A single-point failure triggering system-wide collapse"),
    RiskType("RT3.2", "sandbox_escape", "Sandbox Escape", "ASI05", 3,
             "Agents accessing unauthorized resources"),
    RiskType("RT3.3", "insufficient_monitoring", "Insufficient Monitoring", "ASI09", 3,
             "Lack of effective behavioral monitoring and audit"),
    RiskType("RT3.4", "group_hallucination", "Group Hallucination", "LLM09", 3,
             "Collective fabrication of false information"),
    RiskType("RT3.5", "malicious_emergence", "Malicious Emergence", "ASI01", 3,
             "Emergence of unanticipated harmful behaviors"),
    RiskType("RT3.6", "rogue_agent", "Rogue Agent", "ASI10", 3,
             "Agent deviating from system objectives"),
]

ALL_RISKS = RT1 + RT2 + RT3
BY_KEY = {r.key: r for r in ALL_RISKS}
BY_ID = {r.risk_id: r for r in ALL_RISKS}

_ALIASES = {
    "goal_misgeneralization": "goal_drift",
    "context_rot": "insecure_output_handling",
    "confidence_collapse": "hallucination",
}

def lookup(key: str | None) -> RiskType | None:
    if not key:
        return None
    return BY_KEY.get(key) or BY_KEY.get(_ALIASES.get(key, ""))

def canonical_key(key: str | None) -> str | None:
    r = lookup(key)
    return r.key if r else key
