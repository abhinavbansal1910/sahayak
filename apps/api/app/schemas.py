
# ──────────────────────────────────────────────────────────────
# Sahayak API — Pydantic schemas (the "shape" of our data)
# ──────────────────────────────────────────────────────────────
# This file defines WHAT our data looks like as it flows through the
# pipeline. Every agent will speak these types to each other.
#
# Think of each class as a FORM: it lists the fields, their types, and
# which are required. Pydantic is the "form validator" — if data doesn't
# fit the form, it gets rejected immediately with a clear error.

from enum import Enum

from pydantic import BaseModel, Field


# ── The set of clause types we recognize ──
# An Enum ("enumeration") = a fixed MENU of allowed values. We use it so a
# clause's type can never be a nonsense string like "banana" — only these.
class ClauseType(str, Enum):
    termination = "termination"
    payment = "payment"
    liability = "liability"
    confidentiality = "confidentiality"
    intellectual_property = "intellectual_property"
    governing_law = "governing_law"
    other = "other"


# ── A single clause — the atomic unit of the whole product ──
# A clause is one numbered paragraph in a contract. By the END of the
# pipeline, each clause will ALSO carry a risk score + a counter-draft.
# For Sprint 1 (extraction) we start small: just text + type + position.
# ── Risk verdict (Day 4: LLM+RAG judge output) ──
# The judge says WHO the clause favors; the asymmetry engine turns that
# into the −100…+100 score: direction sign × confidence × 100.
class RiskDirection(str, Enum):
    favors_them = "favors_them"    # lopsided toward the other party → negative
    balanced = "balanced"          # roughly even → ~0
    favors_you = "favors_you"      # lopsided toward OUR reader → positive


class RiskVerdict(BaseModel):
    direction: RiskDirection = Field(..., description="Who this clause favors.")
    confidence: float = Field(..., ge=0, le=1, description="Judge's certainty, 0-1.")
    reason: str = Field(default="", description="One-sentence plain-English justification.")


class Clause(BaseModel):
    # str            → must be text.
    # Field(..., ...) → the "..." means REQUIRED (no default given).
    # description=   → shows up in the auto-generated Swagger /docs!
    text: str = Field(..., description="The actual clause text from the contract.")

    # Has a DEFAULT (other), so it's OPTIONAL when you create a Clause.
    clause_type: ClauseType = Field(
        default=ClauseType.other,
        description="What kind of clause this is.",
    )

    # int → must be a whole number. Required (no default).
    index: int = Field(..., description="Position of this clause in the document (0-based).")
