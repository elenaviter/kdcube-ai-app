from typing import Dict, Any, List, Optional
import json

from kdcube_ai_app.apps.chat.sdk.context.policy.policy import PreferenceExtractionOut
from kdcube_ai_app.infra.service_hub.inventory import ModelServiceBase
from kdcube_ai_app.apps.chat.sdk.util import _today_str


async def preference_extractor(svc: ModelServiceBase, user_text: str, history_hint: str = "") -> Dict[str, Any]:
    today = _today_str()
    sys = (
        "You extract structured user preferences. "
        "OUTPUT STRICT JSON: {\"assertions\":[],\"exceptions\":[]}. "
        "Rules:\n"
        "1) LOWERCASE dotted keys (e.g., 'budget.max', 'suggest.chemical_fertilizer', 'watering.plan').\n"
        "2) ASSERTIONS: desired=true (prefer/include) or desired=false (avoid/exclude). Use scope='conversation'.\n"
        "3) EXCEPTIONS: carve-outs for a rule_key. Use scope='conversation'.\n"
        "4) Keep output minimal; no commentary.\n"
        f"Assume today={today}."
    )

    fewshots = [
        ("Keep pesticide spend under $200 and avoid chemical fertilizer.",
         {"assertions":[
             {"key":"budget.max","value":{"category":"pesticide","amount":200},"desired":True,"scope":"conversation","confidence":0.9,"reason":"user-cap"},
             {"key":"suggest.chemical_fertilizer","value":False,"desired":False,"scope":"conversation","confidence":0.85,"reason":"user-exclusion"}
         ],"exceptions":[]}),
        ("Don't mulch paths — except around the roses.",
         {"assertions":[
             {"key":"mulch.apply","value":False,"desired":False,"scope":"conversation","confidence":0.8,"reason":"user-exclusion"}
         ],
             "exceptions":[
                 {"rule_key":"mulch.apply","value":{"except_for":["roses"]},"scope":"conversation","confidence":0.8,"reason":"user-exception"}
             ]}),
        ("Emphasize drip irrigation and native plants this season.",
         {"assertions":[
             {"key":"focus.drip_irrigation","value":True,"desired":True,"scope":"conversation","confidence":0.8,"reason":"user-preference"},
             {"key":"focus.native_plants","value":True,"desired":True,"scope":"conversation","confidence":0.8,"reason":"user-preference"}
         ],"exceptions":[]})
    ]
    fewshot_block = "\n\n".join(f"User: {u}\nJSON:\n{json.dumps(j, ensure_ascii=False)}" for u,j in fewshots)

    msg = (
        f"Examples:\n{fewshot_block}\n\n"
        f"Message:\n{user_text}\n\n"
        f"Conversation hint (may be empty):\n{(history_hint or '')[:400]}\n\n"
        "Return ONLY JSON with shape: {\"assertions\":[],\"exceptions\":[]}."
    )

    res = await svc.call_model_with_structure(
        svc.classifier_client, sys, msg, PreferenceExtractionOut,
        client_cfg=svc.describe_client(svc.classifier_client, role="preference_extractor")
    )
    if res.get("success"):
        return res["data"]

    # Heuristic fallback (conversation scope)
    import re
    text = user_text.strip()
    assertions: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []

    NEG_PAT = re.compile(r"(?:no|avoid|don['’]t|do not|skip|exclude)\s+([^.,;:]+)", re.I)
    for m in NEG_PAT.finditer(text):
        phrase = m.group(1).strip().strip(".")
        if not phrase:
            continue
        key = "suggest." + re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
        assertions.append({"key": key, "value": False, "desired": False, "scope": "conversation", "confidence": 0.65, "reason": "heuristic-negation"})

    MONEY_PAT = re.compile(r"(?:under|below|capped\s+at|cap(?:ped)?\s+at|within)\s*\$?\s*([0-9][0-9,\.]*\s*[kKmM]?)", re.I)
    CAT_PAT = re.compile(r"(?:for|on|in)\s+([a-zA-Z][\w\s\-]{2,32})")
    m = MONEY_PAT.search(text)
    if m:
        amt_raw = m.group(1).replace(",", "").strip().lower()
        mult = 1
        if amt_raw.endswith("k"): mult, amt_raw = 1_000, amt_raw[:-1]
        elif amt_raw.endswith("m"): mult, amt_raw = 1_000_000, amt_raw[:-1]
        try:
            amount = int(float(amt_raw) * mult)
            cat = None
            after = text[m.end(): m.end() + 40]
            mcat = CAT_PAT.search(after)
            if mcat:
                cat = mcat.group(1).strip().lower()
            assertions.append({
                "key": "budget.max",
                "value": ({"amount": amount, "category": cat} if cat else {"amount": amount}),
                "desired": True, "scope": "conversation", "confidence": 0.9, "reason": "heuristic-budget"
            })
        except Exception:
            pass

    return {"assertions": assertions, "exceptions": exceptions}
