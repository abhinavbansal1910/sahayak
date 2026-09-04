# ──────────────────────────────────────────────────────────────
# Sahayak — Dataset prep (Day 2.3): the classifier's training data
# ──────────────────────────────────────────────────────────────
# Produces a labeled, split, reproducible dataset under model/data/:
#   train.jsonl / val.jsonl  — one JSON object per line: {"text", "label"}
#
# LABEL SCHEMA (3 classes, DIRECTION of favor — feeds the asymmetry
# engine directly: direction × confidence × 100 = −100…+100):
#   favors_them  — the clause disproportionately protects the other party
#   balanced     — roughly even obligations/benefits
#   favors_you   — disproportionately protects the reader (our user)
#
# PROVENANCE (be honest about this): a hand-curated SEED set of realistic
# Indian freelance / employment / gig-platform contract language, written
# from standard boilerplate patterns. Upgrade path: CUAD (Atticus Project)
# mapped onto this 3-class scheme for scale. Seed quality > seed size for
# a LoRA demo — but labels ARE the ceiling on classifier quality.
#
# Run (stdlib only — no venv needed):
#   python model/prepare_dataset.py

import json
import random
from pathlib import Path

SEED = 42
VAL_FRACTION = 0.2
OUT_DIR = Path(__file__).parent / "data"

# ── The seed set ──────────────────────────────────────────────
# (label, clause text) — every line is a plausible contract sentence.

SEED_CLAUSES = [
    # ── favors_them: one-sided in the OTHER party's favor ──
    ("favors_them", "The Company shall pay the Contractor within 90 days of invoice acceptance, at the Company's sole discretion."),
    ("favors_them", "No interest or penalty shall accrue on payments delayed by the Company for any reason whatsoever."),
    ("favors_them", "The Company may withhold any portion of the fees for any perceived defect in the Services, without prior notice to the Contractor."),
    ("favors_them", "Final payment is contingent upon the Company's sole and subjective satisfaction with the deliverables."),
    ("favors_them", "The Company may terminate this Agreement at any time, for any reason, without notice or compensation."),
    ("favors_them", "Termination by the Company forfeits all payment for Services performed but not yet invoiced."),
    ("favors_them", "The Contractor may terminate this Agreement only by providing sixty (60) days' prior written notice."),
    ("favors_them", "The Contractor shall indemnify and hold harmless the Company against all claims arising from the Services, without limitation."),
    ("favors_them", "The Company's total liability under this Agreement shall not exceed Rs. 100 (Rupees One Hundred only)."),
    ("favors_them", "The Contractor shall be liable for all consequential and indirect damages arising from the Services."),
    ("favors_them", "All work product, including the Contractor's pre-existing background IP, vests exclusively in the Company upon creation."),
    ("favors_them", "Assignments of inventions conceived by the Contractor within one year after termination vest in the Company at no additional compensation."),
    ("favors_them", "The Contractor assigns all intellectual property worldwide, in perpetuity, without further consideration."),
    ("favors_them", "The Contractor shall maintain confidentiality in perpetuity, including after termination of this Agreement."),
    ("favors_them", "Any breach of confidentiality entitles the Company to liquidated damages of Rs. 10,00,000 per breach, at the Company's discretion."),
    ("favors_them", "Only the Contractor is bound by confidentiality obligations under this Agreement."),
    ("favors_them", "The Contractor shall not engage with any business competing with the Company for twenty-four (24) months, anywhere in India."),
    ("favors_them", "The Company shall have sole discretion to determine what constitutes a competing business."),
    ("favors_them", "The courts at the Company's registered office shall have exclusive jurisdiction over all disputes."),
    ("favors_them", "The Company may amend these terms at any time by posting an updated version on its website, without individual notice."),
    ("favors_them", "The Platform may deactivate the Partner's account at any time, with or without cause, and without liability."),
    ("favors_them", "The Partner shall bear all costs of fuel, maintenance, insurance, and repairs of the vehicle used to provide Services."),
    ("favors_them", "The Platform may revise payout rates, incentives, and fees at any time, effective immediately upon app notification."),
    ("favors_them", "Any dispute shall be resolved by final and binding arbitration at the Company's chosen venue, to the exclusion of all courts and remedies."),
    ("favors_them", "The Company may assign this Agreement freely; the Contractor may not assign any rights or obligations whatsoever."),

    # ── balanced: roughly even obligations ──
    ("balanced", "The Client shall pay the Contractor's invoices within thirty (30) days of receipt."),
    ("balanced", "Interest at one percent (1%) per month shall accrue on any overdue amount payable by either party."),
    ("balanced", "Either party may terminate this Agreement with fifteen (15) days' prior written notice to the other."),
    ("balanced", "Upon termination, the Client shall pay for all Services performed up to the effective date of termination within fifteen (15) days."),
    ("balanced", "Each party shall be liable for losses caused by its own negligence or wilful misconduct."),
    ("balanced", "Each party's aggregate liability shall be capped at the fees paid or payable in the three (3) months preceding the claim."),
    ("balanced", "Each party shall indemnify the other against losses arising from its own breach of this Agreement."),
    ("balanced", "Ownership of work product shall transfer to the Client upon receipt of full and final payment."),
    ("balanced", "The Contractor retains ownership of pre-existing background IP and grants the Client a licence to use it as embedded in the deliverables."),
    ("balanced", "Subject to the confidentiality obligations below, the Contractor may display the deliverables in a professional portfolio."),
    ("balanced", "Both parties shall keep confidential information secret for three (3) years from disclosure."),
    ("balanced", "The confidentiality obligations of both parties shall survive termination of this Agreement for two (2) years."),
    ("balanced", "Confidentiality obligations apply equally to information exchanged by either party under this Agreement."),
    ("balanced", "Disputes shall first be escalated to senior representatives of both parties for good-faith resolution for thirty (30) days."),
    ("balanced", "Any dispute not resolved by negotiation shall be referred to arbitration under the Arbitration and Conciliation Act, 1996."),
    ("balanced", "The seat of arbitration shall be mutually agreed between the parties; failing agreement, it shall be Delhi."),
    ("balanced", "This Agreement is governed by the laws of India, and the courts at the place of contracting shall have jurisdiction."),
    ("balanced", "This Agreement may be amended only by a written instrument signed by both parties."),
    ("balanced", "The Platform's commission shall be fifteen percent (15%) of the fare, as set out in Schedule A."),
    ("balanced", "The Platform may deactivate a Partner's account only after three (3) documented violations and written notice."),
    ("balanced", "Neither party shall assign this Agreement without the prior written consent of the other party."),
    ("balanced", "Neither party shall be liable for delay caused by events beyond its reasonable control, provided prompt notice is given."),
    ("balanced", "The Contractor shall provide the Services with reasonable skill and care; the Client shall provide timely access and materials."),
    ("balanced", "Either party may request a change in scope, effective only upon a signed written change order."),

    # ── favors_you: one-sided in OUR reader's favor ──
    ("favors_you", "Fifty percent (50%) of the fee shall be paid in advance upon signing, and the balance within seven (7) days of delivery."),
    ("favors_you", "Overdue payments shall accrue interest at two percent (2%) per month, compounded monthly, automatically and without notice."),
    ("favors_you", "If the Client cancels the project after commencement, the Contractor shall be paid a kill fee of fifty percent (50%) of the remaining contract value."),
    ("favors_you", "The Client shall pay for all Services performed up to the date of termination within seven (7) days, regardless of the reason for termination."),
    ("favors_you", "The Contractor may terminate this Agreement with seven (7) days' written notice; the Client must provide thirty (30) days' notice."),
    ("favors_you", "Termination by the Client for convenience shall not affect the Contractor's entitlement to fees for work already performed."),
    ("favors_you", "The Client shall indemnify and hold harmless the Contractor against all claims arising from materials or instructions supplied by the Client."),
    ("favors_you", "Delays caused by the Client shall extend every affected deadline day-for-day, and additional effort shall be billed at the stated hourly rate."),
    ("favors_you", "The Client shall bear the cost of all third-party licences, fonts, and stock assets requested by the Client for the project."),
    ("favors_you", "Ownership of the work product shall vest in the Contractor until receipt of full and final payment."),
    ("favors_you", "The licence granted to the Client shall commence only upon full payment and shall terminate upon any chargeback."),
    ("favors_you", "The Contractor retains the right to display the work in a portfolio and to reuse generic techniques and know-how, notwithstanding the licence."),
    ("favors_you", "Confidentiality obligations apply only to information expressly marked as confidential and expire one (1) year from disclosure."),
    ("favors_you", "Information that is publicly available or independently developed by the Contractor is excluded from confidentiality obligations."),
    ("favors_you", "The Client shall not solicit or engage the Contractor's employees or subcontractors for twelve (12) months following termination."),
    ("favors_you", "Nothing in this Agreement restricts the Contractor from providing services to other clients during the term."),
    ("favors_you", "Any work outside the signed scope of work shall require a written change order with adjusted fees and timeline before commencement."),
    ("favors_you", "The Client shall respond to deliverables within five (5) business days; failing which the deliverables shall be deemed accepted."),
    ("favors_you", "All payouts to the Partner shall be settled weekly, no later than every Monday, for trips completed in the preceding week."),
    ("favors_you", "Incentive amounts advertised for the contract term are guaranteed and shall not be reduced unilaterally during the term."),
    ("favors_you", "The Platform shall provide and maintain accident insurance coverage for the Partner for the entire term at the Platform's cost."),
    ("favors_you", "Deactivation of the Partner's account shall require prior written notice and an opportunity to be heard, except in cases of fraud."),
    ("favors_you", "Any amendment adverse to the Contractor shall require thirty (30) days' prior written notice and the Contractor's explicit consent."),
    ("favors_you", "In any dispute, the prevailing party's reasonable legal costs shall be borne by the losing party."),
]


def stratified_split(rows: list, val_fraction: float, seed: int) -> tuple[list, list]:
    """Split per class so train and val keep the SAME class mix.

    A plain shuffle can land zero examples of a class in val — then val
    accuracy is a lie (missing class looks like perfect performance).
    Stratify: split within each class, then join.
    """
    rng = random.Random(seed)
    train, val = [], []
    by_label: dict[str, list] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)  # mix classes within each split too
    rng.shuffle(val)
    return train, val


def main() -> None:
    rows = [{"text": text, "label": label} for label, text in SEED_CLAUSES]
    train, val = stratified_split(rows, VAL_FRACTION, SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train.jsonl", train), ("val.jsonl", val)):
        path = OUT_DIR / name
        with path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Report — and VERIFY the stratification actually worked.
    print(f"Dataset: {len(rows)} clauses -> {len(train)} train / {len(val)} val")
    for name, split in (("train", train), ("val", val)):
        counts = {}
        for row in split:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        assert set(counts) == {"favors_them", "balanced", "favors_you"}, f"{name} split lost a class!"
    print(f"Wrote {OUT_DIR / 'train.jsonl'} and {OUT_DIR / 'val.jsonl'}")


if __name__ == "__main__":
    main()
