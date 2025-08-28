# What’s an assertion vs. an exception?

* **Assertion** = a preference about *what to do or avoid* keyed by a normalized dotted key.

    * `desired=True` → “prefer/include” (e.g., `budget.max = {"amount":450000}`)
    * `desired=False` → “avoid/exclude” (e.g., `suggest.phishing_training = false`)
* **Exception** = a *carve-out* to a rule (usually to an assertion or a policy rule) keyed by `rule_key`. Its `value` encodes the condition under which the rule doesn’t apply.

    * Example: “avoid phishing training **except executives**”

        * Assertion (neg): `{"key":"suggest.phishing_training","value":false,"desired":false,...}`
        * Exception: `{"rule_key":"suggest.phishing_training","value":{"except_for":["executives"]}}`

In policy resolution:

* `exception` outranks everything for that key (your precedence already encodes that).
* `desired=False` is a negation (avoid). It’s not the same as an exception: the negation is the baseline; the exception *scopes* a permitted condition.