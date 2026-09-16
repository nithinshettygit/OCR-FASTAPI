"""
Amount Extraction Engine
========================

Standalone VS Code / Python version of the notebook.

Supported inputs:
    PDF, PNG, JPG, JPEG, WEBP

Main API:
    extract_max_amount(path, return_debug=False)

CLI:
    python amount_extraction_engine.py "path/to/invoice.pdf"
 python amount_extraction_engine.py "D:\xfcatr\FASTAPI-OCR-API\data\100250858-MARUTHI-DN 1.pdf"
Requirements:
    pip install opencv-python pymupdf numpy pytesseract

System requirement:
    Tesseract OCR must also be installed and available in PATH.
"""

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from pytesseract import Output


# ============================================================
# CONFIGURATION
# ============================================================

PDF_DPI = 200
MIN_EMBEDDED_IMAGE_AREA = 500 * 500
OCR_CONFIG = "--psm 6"
MIN_AMOUNT = 0.01
AMOUNT_COLUMN_MARGIN_X = 45
HEADER_UPSCALE = 2

# Matches "Amount", "Amount (Rs.)", "Amount(INR)", "Amt", etc.
AMOUNT_HEADER_RE = re.compile(r"amou?nt", re.IGNORECASE)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


# ============================================================
# AMOUNT PATTERN + PARSER
# ============================================================

AMOUNT_RE = re.compile(
    r"""
    ^[+-]?
    (?:
        # Western / International grouping
        # 31,746.00
        # 3,174,600.00
        # 12,345,678.90
        \d{1,3}(?:,\d{3})+(?:\.\d{1,2})?

        |

        # Indian grouping
        # 1,00,000.00
        # 31,74,600.00
        # 3,17,46,000.00
        # 12,34,56,789.00
        \d{1,3}(?:,\d{2})+,\d{3}(?:\.\d{1,2})?

        |

        # Plain number
        # 31746.00
        # 31746000.00
        \d+(?:\.\d{1,2})?
    )
    $
    """,
    re.VERBOSE,
)


def normalize_amount_token(text: Any) -> str:
    """Normalize OCR text before amount parsing."""
    if text is None:
        return ""

    s = str(text).strip()
    s = re.sub(r"\s+", "", s)

    # Common high-confidence OCR corrections.
    s = s.translate(
        str.maketrans(
            {
                "O": "0",
                "o": "0",
                "I": "1",
                "l": "1",
                "|": "1",
            }
        )
    )

    # Keep only characters that can belong to an amount.
    s = re.sub(r"[^0-9,.\-+]", "", s)

    return s


def parse_amount(text: Any) -> Optional[float]:
    """
    Convert OCR text into a signed float.

    Returns:
        float: Valid amount.
        None: If the value is invalid or below MIN_AMOUNT.
    """
    s = normalize_amount_token(text)

    if not s:
        return None

    negative = False

    # Trailing minus: 31,746.00-
    if s.endswith("-"):
        negative = True
        s = s[:-1]

    # Leading minus: -31,746.00
    if s.startswith("-"):
        negative = True
        s = s[1:]

    # Leading plus.
    if s.startswith("+"):
        s = s[1:]

    if not s:
        return None

    # OCR correction:
    # 31.746.00 -> 31,746.00
    if s.count(".") > 1 and "," not in s:
        parts = s.split(".")

        if len(parts[-1]) in (1, 2) and all(part.isdigit() for part in parts):
            s = ",".join(parts[:-1]) + "." + parts[-1]

    if not AMOUNT_RE.fullmatch(s):
        return None

    try:
        value = float(s.replace(",", ""))

        if abs(value) < MIN_AMOUNT:
            return None

        return -value if negative else value

    except (ValueError, TypeError):
        return None


# ============================================================
# PDF RENDERING
# ============================================================

def render_pdf(path: Union[str, Path], dpi: int = PDF_DPI) -> List[Tuple[int, np.ndarray]]:
    """
    Convert PDF pages into OpenCV BGR images.

    Per page:
    1. Use the largest embedded image when it is large enough.
    2. Otherwise render the complete PDF page.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    pages: List[Tuple[int, np.ndarray]] = []

    try:
        doc = fitz.open(str(path))

        try:
            for page_index, page in enumerate(doc):
                page_no = page_index + 1
                best_xref = None
                best_area = 0

                # Embedded-image fast path.
                for img_info in page.get_images(full=True):
                    xref = img_info[0]

                    try:
                        pix = fitz.Pixmap(doc, xref)
                        area = pix.width * pix.height

                        if area > best_area and area >= MIN_EMBEDDED_IMAGE_AREA:
                            best_area = area
                            best_xref = xref

                    except Exception:
                        continue

                if best_xref is not None:
                    pix = fitz.Pixmap(doc, best_xref)

                    if pix.colorspace and pix.colorspace.n not in (1, 3):
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    if pix.n >= 4:
                        arr = np.frombuffer(
                            pix.samples,
                            dtype=np.uint8,
                        ).reshape(pix.height, pix.width, pix.n)

                        img = cv2.cvtColor(
                            arr[:, :, :4],
                            cv2.COLOR_RGBA2BGR,
                        )

                    elif pix.n == 3:
                        arr = np.frombuffer(
                            pix.samples,
                            dtype=np.uint8,
                        ).reshape(pix.height, pix.width, 3)

                        img = cv2.cvtColor(
                            arr,
                            cv2.COLOR_RGB2BGR,
                        )

                    else:
                        arr = np.frombuffer(
                            pix.samples,
                            dtype=np.uint8,
                        ).reshape(pix.height, pix.width)

                        img = cv2.cvtColor(
                            arr,
                            cv2.COLOR_GRAY2BGR,
                        )

                    pages.append((page_no, img))
                    continue

                # Fallback: render the complete page.
                scale = dpi / 72.0
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    alpha=False,
                )

                arr = np.frombuffer(
                    pix.samples,
                    dtype=np.uint8,
                ).reshape(pix.height, pix.width, 3)

                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                pages.append((page_no, img))

        finally:
            doc.close()

        return pages

    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Unable to process PDF '{path.name}': {exc}"
        ) from exc


# ============================================================
# INPUT LOADER
# ============================================================

def load_input(
    path: Union[str, Path],
) -> List[Tuple[int, np.ndarray]]:
    """Load PDF/image input and return a list of (page_number, image)."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        pages = render_pdf(path)

        if not pages:
            raise RuntimeError(f"No readable pages found in: {path.name}")

        return pages

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Unable to read image: {path.name}")

    return [(1, img)]


# ============================================================
# BLANK PAGE DETECTION
# ============================================================

def is_blank_page(img: Optional[np.ndarray]) -> bool:
    """Return True when the image is likely blank."""
    if img is None or img.size == 0:
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return bool(
        np.mean(gray) > 245
        and np.mean(gray < 245) < 0.02
    )


def remove_blank_pages(
    pages: List[Tuple[int, np.ndarray]]
) -> List[Tuple[int, np.ndarray]]:
    """Remove blank pages while keeping pages when detection itself fails."""
    valid: List[Tuple[int, np.ndarray]] = []

    for page_no, img in pages:
        try:
            if not is_blank_page(img):
                valid.append((page_no, img))
        except Exception as exc:
            print(
                f"Warning: blank-page check failed on page "
                f"{page_no}: {exc}"
            )
            valid.append((page_no, img))

    return valid


# ============================================================
# OCR
# ============================================================

def run_ocr(
    img: Optional[np.ndarray],
    config: str = OCR_CONFIG,
) -> str:
    """Run Tesseract OCR on an image with basic preprocessing."""
    if img is None or img.size == 0:
        return ""

    try:
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.ndim == 3
            else img
        )

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

        processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        return pytesseract.image_to_string(
            processed,
            config=config,
        )

    except Exception as exc:
        print(f"OCR warning: {exc}")
        return ""


# ============================================================
# AMOUNT COLUMN DETECTION
# ============================================================

def find_amount_header(
    img: np.ndarray,
) -> Optional[Dict[str, int]]:
    """Locate the topmost OCR match for the Amount header."""
    try:
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.ndim == 3
            else img
        )

        scaled = cv2.resize(
            gray,
            None,
            fx=HEADER_UPSCALE,
            fy=HEADER_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        data = pytesseract.image_to_data(
            scaled,
            output_type=Output.DICT,
        )

    except Exception as exc:
        print(f"Header detection OCR warning: {exc}")
        return None

    best = None

    for i in range(len(data.get("text", []))):
        word = (data["text"][i] or "").strip()

        if not word or not AMOUNT_HEADER_RE.search(word):
            continue

        box = {
            "left": int(data["left"][i] // HEADER_UPSCALE),
            "top": int(data["top"][i] // HEADER_UPSCALE),
            "width": int(data["width"][i] // HEADER_UPSCALE),
            "height": int(data["height"][i] // HEADER_UPSCALE),
        }

        if best is None or box["top"] < best["top"]:
            best = box

    return best


def crop_amount_column(
    img: np.ndarray,
    header_box: Dict[str, int],
    margin_x: int = AMOUNT_COLUMN_MARGIN_X,
) -> np.ndarray:
    """Crop a full-height vertical strip around the detected header."""
    h, w = img.shape[:2]

    left = max(0, header_box["left"] - margin_x)
    right = min(
        w,
        header_box["left"] + header_box["width"] + margin_x,
    )
    top = max(0, header_box["top"] - 5)

    return img[top:h, left:right]


# ============================================================
# CANDIDATE EXTRACTION
# ============================================================

# Requires comma/dot so invoice IDs, dates and quantities
# represented as bare digit runs are not picked up.
_CANDIDATE_RE = re.compile(
    r"""
    (?<![0-9])
    [+-]?
    (?:
        \d{1,3}(?:[,.]\d{3})*
        |
        \d+
    )
    (?:[.,]\d{1,2})?
    -?
    (?![0-9])
    """,
    re.VERBOSE,
)


def extract_amount_candidates_from_text(
    text: str,
) -> List[Dict[str, Any]]:
    """Find and parse amount-shaped tokens from OCR text."""
    if not text:
        return []

    candidates: List[Dict[str, Any]] = []

    for token in _CANDIDATE_RE.findall(text):
        # Ignore plain digit runs.
        if "," not in token and "." not in token:
            continue

        value = parse_amount(token)

        if value is not None:
            candidates.append(
                {
                    "raw": token,
                    "value": value,
                }
            )

    return candidates


# ============================================================
# PER-PAGE EXTRACTION
# ============================================================

def extract_from_page(
    img: np.ndarray,
    page_no: int,
) -> Dict[str, Any]:
    """
    Extract amounts from one page.

    Returns:
        {
            "rows": [{"page": ..., "raw": ..., "value": ...}],
            "column_detected": bool
        }
    """
    rows: List[Dict[str, Any]] = []

    header_box = find_amount_header(img)

    if header_box is not None:
        crop = crop_amount_column(img, header_box)
        text = run_ocr(crop, config=OCR_CONFIG)

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            # Skip the header line itself.
            normalized_header = re.sub(r"[^A-Za-z]", "", line)

            if AMOUNT_HEADER_RE.fullmatch(normalized_header):
                continue

            parsed = parse_amount(line)

            if parsed is not None:
                rows.append(
                    {
                        "page": page_no,
                        "raw": line,
                        "value": line,
                    }
                )
                continue

            # Line has extra OCR noise around the number.
            for candidate in extract_amount_candidates_from_text(line):
                rows.append(
                    {
                        "page": page_no,
                        "raw": line,
                        "value": candidate["raw"],
                    }
                )

        return {
            "rows": rows,
            "column_detected": True,
        }

    # Fallback: no Amount header found.
    text = run_ocr(img)

    for candidate in extract_amount_candidates_from_text(text):
        rows.append(
            {
                "page": page_no,
                "raw": candidate["raw"],
                "value": candidate["raw"],
            }
        )

    return {
        "rows": rows,
        "column_detected": False,
    }


# ============================================================
# COMPLETE EXTRACTION ENGINE
# ============================================================

def extract_max_amount(
    path: Union[str, Path],
    return_debug: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    Extract the maximum valid amount from a PDF/image.

    If return_debug=False:
        Returns float on success, or a structured failure dict.

    If return_debug=True:
        Always returns the complete structured result.

    Failure example:
        {
            "success": False,
            "amount": None,
            "error": "NO_AMOUNT_DETECTED",
            ...
        }
    """
    path = Path(path)
    total_start = time.perf_counter()

    timing: Dict[str, float] = {
        "load_input": 0.0,
        "blank_page_detection": 0.0,
        "amount_extraction": 0.0,
        "amount_selection": 0.0,
        "total": 0.0,
    }

    def fail(
        error: str,
        message: str,
        rows: Optional[List[Dict[str, Any]]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        timing["total"] = time.perf_counter() - total_start

        return {
            "success": False,
            "amount": None,
            "error": error,
            "message": message,
            "rows": rows or [],
            "timing": timing,
            **extra,
        }

    try:
        # 1. LOAD INPUT
        start = time.perf_counter()
        pages = load_input(path)
        timing["load_input"] = time.perf_counter() - start

        if not pages:
            return fail(
                "NO_PAGES",
                f"No pages found in {path.name}",
            )

        # 2. REMOVE BLANK PAGES
        start = time.perf_counter()
        non_blank_pages = remove_blank_pages(pages)
        timing["blank_page_detection"] = (
            time.perf_counter() - start
        )

        if not non_blank_pages:
            return fail(
                "ALL_PAGES_BLANK",
                f"All pages are blank in {path.name}",
            )

        # 3. AMOUNT EXTRACTION
        start = time.perf_counter()

        all_rows: List[Dict[str, Any]] = []
        page_errors: List[Dict[str, Any]] = []
        any_column_detected = False

        for page_no, img in non_blank_pages:
            try:
                page_result = extract_from_page(img, page_no)

                if page_result.get("column_detected"):
                    any_column_detected = True

                all_rows.extend(
                    page_result.get("rows", [])
                )

            except Exception as exc:
                page_errors.append(
                    {
                        "page": page_no,
                        "error": str(exc),
                    }
                )

                print(
                    f"Warning: extraction failed on page "
                    f"{page_no}: {exc}"
                )

        timing["amount_extraction"] = (
            time.perf_counter() - start
        )

        if not all_rows:
            return fail(
                "NO_AMOUNT_DETECTED",
                f"No valid Amount values found in: {path.name}",
                pages_processed=len(non_blank_pages),
                column_detected=any_column_detected,
                page_errors=page_errors,
            )

        # 4. FINAL AMOUNT SELECTION
        start = time.perf_counter()

        values = [
            parse_amount(row.get("value"))
            for row in all_rows
            if isinstance(row, dict)
        ]

        values = [
            value for value in values
            if value is not None
        ]

        timing["amount_selection"] = (
            time.perf_counter() - start
        )

        if not values:
            return fail(
                "NO_VALID_AMOUNT",
                (
                    "Amount column detected but no valid "
                    f"numeric amount could be parsed in: {path.name}"
                ),
                rows=all_rows,
                pages_processed=len(non_blank_pages),
                column_detected=any_column_detected,
                page_errors=page_errors,
            )

        # Select largest positive amount.
        # If all values are negative, select largest magnitude.
        positive_values = [
            value for value in values
            if value > 0
        ]

        final_amount = (
            max(positive_values)
            if positive_values
            else max(values, key=abs)
        )

        timing["total"] = (
            time.perf_counter() - total_start
        )

        result: Dict[str, Any] = {
            "success": True,
            "amount": final_amount,
            "error": None,
            "message": "Amount extracted successfully",
            "rows": all_rows,
            "pages_processed": len(non_blank_pages),
            "column_detected": any_column_detected,
            "page_errors": page_errors,
            "timing": timing,
        }

        return result if return_debug else final_amount

    except FileNotFoundError as exc:
        return fail(
            "FILE_NOT_FOUND",
            str(exc),
        )

    except ValueError as exc:
        return fail(
            "INVALID_INPUT",
            str(exc),
        )

    except Exception as exc:
        return fail(
            "ENGINE_ERROR",
            str(exc),
        )


# ============================================================
# OPTIONAL DEBUG / DISPLAY HELPERS
# ============================================================

def print_result(result: Union[float, Dict[str, Any]]) -> None:
    """Print a clean result summary."""
    print("=" * 60)
    print("AMOUNT EXTRACTION RESULT")
    print("=" * 60)

    if isinstance(result, (int, float)):
        print("Status          : SUCCESS")
        print("Amount          :", f"{result:,.2f}")
        return

    if result.get("success"):
        print("Status          : SUCCESS")
        print("Amount          :", f"{result['amount']:,.2f}")
    else:
        print("Status          : FAILED")
        print("Error           :", result.get("error"))
        print("Message         :", result.get("message"))

    if "pages_processed" in result:
        print("Pages Processed :", result["pages_processed"])

    if "column_detected" in result:
        print("Column Detected :", result["column_detected"])

    if result.get("page_errors"):
        print("Page Errors     :", result["page_errors"])

    timing = result.get("timing", {})
    if timing:
        print(
            "Execution Time  :",
            f"{timing.get('total', 0.0):.4f} sec",
        )

    print("=" * 60)


def print_timing(result: Dict[str, Any]) -> None:
    """Print timing breakdown from a debug result."""
    timing = result.get("timing", {})

    print("\nENGINE EXECUTION BREAKDOWN")
    print("-" * 45)

    for name, value in timing.items():
        print(f"{name:<30}: {value:.4f} sec")


def print_detected_rows(result: Dict[str, Any]) -> None:
    """Print all detected amount rows without requiring pandas."""
    rows = result.get("rows", [])

    if not rows:
        print("\nNo rows were detected for this document.")
        return

    print("\nDETECTED AMOUNT ROWS")
    print("-" * 70)

    for index, row in enumerate(rows, start=1):
        parsed = parse_amount(row.get("value"))

        print(
            f"{index:>4}. "
            f"page={row.get('page')} | "
            f"raw={row.get('raw')!r} | "
            f"value={row.get('value')!r} | "
            f"parsed={parsed}"
        )

    print("-" * 70)


# ============================================================
# PARSER SELF-TEST
# ============================================================

def run_parser_test() -> None:
    """Run the original parser test values."""
    test_values = [
        "31,746.00",
        "31,746.00-",
        "31.746.00-",
        "31746.00",
        "2,054.00-",
        "41,085.00",
        "O1,746.00",
        "43,139",
        "1,234,567.89",
        "-31746.00",
        "+31746.00",
        "abc",
        "",
        "0.00",
        "3,17,46,000.00",
        "31.746.000.00",
    ]

    print("PARSER TEST")
    print("-" * 45)

    for value in test_values:
        print(
            f"{value!r:<20} -> {parse_amount(value)}"
        )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the maximum amount from a PDF or image."
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Path to PDF / PNG / JPG / JPEG / WEBP",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show timing, detected rows and debug information.",
    )

    parser.add_argument(
        "--test-parser",
        action="store_true",
        help="Run the amount parser self-test and exit.",
    )

    args = parser.parse_args()

    if args.test_parser:
        run_parser_test()
        return

    if not args.input:
        parser.error(
            "Please provide an input file path. "
            'Example: python amount_extraction_engine.py "invoice.pdf"'
        )

    input_path = Path(args.input)

    result = extract_max_amount(
        input_path,
        return_debug=True,
    )

    print_result(result)

    if args.debug:
        print_timing(result)
        print_detected_rows(result)


if __name__ == "__main__":
    main()
