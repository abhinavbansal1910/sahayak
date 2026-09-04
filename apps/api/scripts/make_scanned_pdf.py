# ──────────────────────────────────────────────────────────────
# Sahayak — Test fixture: fabricate a "scanned" PDF
# ──────────────────────────────────────────────────────────────
# A scanned PDF is just photos of pages inside a PDF wrapper. We fake
# one: draw contract text onto an image with Pillow, then save that
# image AS a PDF. Result — a PDF with ZERO text layer, perfect for
# testing the OCR fallback (Day 1.3) without owning a scanner.
#
# Run INSIDE the container (it has Pillow + DejaVu fonts):
#   docker compose exec -T api python scripts/make_scanned_pdf.py
# → writes /tmp/scanned_contract.pdf
#
# Then fire it at the pipeline:
#   docker compose exec -T api python - <<'PY'
#   import httpx
#   pdf = open("/tmp/scanned_contract.pdf", "rb")
#   r = httpx.post("http://localhost:8000/analyze",
#                  files={"file": ("scanned.pdf", pdf, "application/pdf")},
#                  timeout=120)
#   print(r.status_code, r.json()["used_ocr"])
#   PY

from PIL import Image, ImageDraw, ImageFont

PAGE = (1240, 1754)  # ~A4 at 150 DPI, in pixels
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Deliberately lopsided clauses — this file doubles as demo input for
# the Extraction + Risk agents in later days.
CONTRACT_LINES = [
    ("FREELANCE SERVICES AGREEMENT", "bold"),
    ("", None),
    ("This Agreement is made on 4 September 2026 between QuickHire", None),
    ('Technologies Pvt. Ltd. ("the Company") and the undersigned', None),
    ('freelancer ("the Contractor").', None),
    ("", None),
    ("1. PAYMENT. The Company shall pay the Contractor a fee of", None),
    ("Rs. 25,000 per project. Payment will be made within 90 days of", None),
    ("invoice acceptance, at the Company's sole discretion.", None),
    ("", None),
    ("2. TERMINATION. The Company may terminate this Agreement at any", None),
    ("time, for any reason, without notice or compensation. The", None),
    ("Contractor may terminate only with 60 days written notice.", None),
    ("", None),
    ("3. INTELLECTUAL PROPERTY. All work product, including background", None),
    ("IP, vests exclusively in the Company upon creation.", None),
    ("", None),
    ("4. LIABILITY. The Contractor shall indemnify the Company for all", None),
    ("claims arising from the services, without limitation.", None),
]


def main() -> None:
    image = Image.new("RGB", PAGE, "white")
    draw = ImageDraw.Draw(image)
    y = 140
    for text, style in CONTRACT_LINES:
        size = 34 if style == "bold" else 26
        try:
            font = ImageFont.truetype(FONT_BOLD if style == "bold" else FONT, size)
        except OSError:
            font = ImageFont.load_default(size)
        draw.text((110, y), text, fill="black", font=font)
        y += 48
    image.save("/tmp/scanned_contract.pdf", "PDF", resolution=150)
    print("Wrote /tmp/scanned_contract.pdf (image-only — no text layer)")


if __name__ == "__main__":
    main()
