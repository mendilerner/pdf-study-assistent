"""PDF text extraction with page attribution.

Uses PyMuPDF with RTL-aware span reconstruction: the PDF stores
each Hebrew word as a separate span, so we group spans by vertical
position into visual lines, then sort right-to-left within each
line to recover correct logical reading order.

Handles:
- Two-column layouts: detects spans on both sides of the page
  midpoint and splits them into separate lines (right column
  first for RTL reading order).
- Heading detection: uses font size and bold flags to prefix
  headings with "## ".
- Watermark removal: filters out the repeated watermark text.

Interface:
    parse(pdf_path) -> list of {pdf_page: int, text: str}
    detect_page_offset(pdf_path) -> int
"""

import pymupdf

Y_TOLERANCE = 4
COLUMN_GAP_MIN = 30
WATERMARK_PATTERNS = ("File #", "mendi3266", "702620212", "__-_-__--")


def _collect_spans(page: pymupdf.Page) -> list[tuple]:
    """Collect all text spans with position and font metadata.

    Returns list of (x0, y0, x1, y1, text, font_size, is_bold).
    """
    data = page.get_text("dict", sort=True)
    spans = []
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                text_lower = text.lower()
                if any(text_lower.startswith(p.lower()) for p in WATERMARK_PATTERNS):
                    continue
                bbox = span["bbox"]
                font_size = span["size"]
                is_bold = bool(span["flags"] & 16)
                spans.append((bbox[0], bbox[1], bbox[2], bbox[3],
                              text, font_size, is_bold))
    return spans


def _detect_body_font_size(spans: list[tuple]) -> float:
    """Find the most common font size (= body text size)."""
    size_counts: dict[float, int] = {}
    for _, _, _, _, text, font_size, _ in spans:
        rounded = round(font_size, 0)
        size_counts[rounded] = size_counts.get(rounded, 0) + len(text)
    if not size_counts:
        return 10.0
    return max(size_counts, key=size_counts.get)


def _group_into_visual_lines(spans: list[tuple]) -> list[list[tuple]]:
    """Group spans sharing the same approximate y-coordinate."""
    if not spans:
        return []

    spans = sorted(spans, key=lambda s: (s[1], s[0]))

    lines = []
    current = [spans[0]]
    current_y = spans[0][1]

    for span in spans[1:]:
        if abs(span[1] - current_y) <= Y_TOLERANCE:
            current.append(span)
        else:
            lines.append(current)
            current = [span]
            current_y = span[1]
    lines.append(current)
    return lines


def _detect_column_boundary(spans: list[tuple], page_width: float) -> float | None:
    """Detect if the page has a two-column layout and return the x-boundary.

    Builds a text-density histogram across the page width and looks
    for a valley in the middle zone where density drops to less than
    20% of the surrounding peaks. Returns the center of the valley,
    or None for single-column pages.
    """
    if not spans:
        return None

    num_buckets = 100
    bucket_width = page_width / num_buckets
    buckets = [0] * num_buckets

    for x0, _, x1, _, text, _, _ in spans:
        sb = max(0, int(x0 / bucket_width))
        eb = min(num_buckets - 1, int(x1 / bucket_width))
        for b in range(sb, eb + 1):
            buckets[b] += len(text)

    mid_start = int(num_buckets * 0.35)
    mid_end = int(num_buckets * 0.65)

    left_peak = max(buckets[10:mid_start], default=0)
    right_peak = max(buckets[mid_end:90], default=0)

    if left_peak == 0 or right_peak == 0:
        return None

    threshold = min(left_peak, right_peak) * 0.20

    best_valley_start = -1
    best_valley_len = 0
    cur_start = -1
    cur_len = 0

    for b in range(mid_start, mid_end + 1):
        if buckets[b] < threshold:
            if cur_start == -1:
                cur_start = b
            cur_len += 1
        else:
            if cur_len > best_valley_len:
                best_valley_start = cur_start
                best_valley_len = cur_len
            cur_start = -1
            cur_len = 0

    if cur_len > best_valley_len:
        best_valley_start = cur_start
        best_valley_len = cur_len

    if best_valley_len < 2:
        return None

    return (best_valley_start + best_valley_len / 2) * bucket_width


def _split_at_boundary(vline: list[tuple], boundary: float) -> list[list[tuple]]:
    """Split a visual line at a column boundary.

    Returns [right_column, left_column] for RTL reading order.
    If all spans are on one side, returns [vline].
    """
    left = [s for s in vline if (s[0] + s[2]) / 2 < boundary]
    right = [s for s in vline if (s[0] + s[2]) / 2 >= boundary]

    if not left or not right:
        return [vline]

    return [right, left]


def _is_heading(spans: list[tuple], body_size: float) -> bool:
    """Check if a line's spans represent a heading."""
    if not spans:
        return False
    total_chars = sum(len(s[4]) for s in spans)
    bold_chars = sum(len(s[4]) for s in spans if s[6])
    large_chars = sum(len(s[4]) for s in spans if s[5] > body_size + 1)
    if total_chars == 0:
        return False
    return (bold_chars / total_chars > 0.5) and (large_chars / total_chars > 0.5)


def _render_line(spans: list[tuple], body_size: float) -> str:
    """Sort spans RTL and join into a text line, with heading prefix if needed."""
    sorted_spans = sorted(spans, key=lambda s: s[0], reverse=True)
    text = " ".join(s[4] for s in sorted_spans)
    if _is_heading(spans, body_size):
        text = "## " + text
    return text


def _extract_page_text(page: pymupdf.Page) -> str:
    """Extract text from a single page with correct RTL word order."""
    spans = _collect_spans(page)
    if not spans:
        return ""

    body_size = _detect_body_font_size(spans)
    visual_lines = _group_into_visual_lines(spans)
    page_width = page.rect.width
    col_boundary = _detect_column_boundary(spans, page_width)

    if col_boundary is not None:
        right_lines = []
        left_lines = []
        for vline in visual_lines:
            cols = _split_at_boundary(vline, col_boundary)
            if len(cols) == 2:
                right_lines.append(cols[0])
                left_lines.append(cols[1])
            else:
                x_center = sum((s[0] + s[2]) / 2 for s in vline) / len(vline)
                if x_center >= col_boundary:
                    right_lines.append(vline)
                else:
                    left_lines.append(vline)

        output_lines = []
        for col_lines in [right_lines, left_lines]:
            for col in col_lines:
                line_text = _render_line(col, body_size)
                if line_text:
                    output_lines.append(line_text)
    else:
        output_lines = []
        for vline in visual_lines:
            line_text = _render_line(vline, body_size)
            if line_text:
                output_lines.append(line_text)

    return "\n".join(output_lines)


def parse(pdf_path: str) -> list[dict]:
    """Extract text from every page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts, one per page:
          - pdf_page (int): 1-based position in the PDF file
          - text (str): Extracted text content

        Pages with no extractable text are included with text="".
    """
    doc = pymupdf.open(pdf_path)
    results = []
    for page_idx in range(len(doc)):
        text = _extract_page_text(doc[page_idx])
        results.append({
            "pdf_page": page_idx + 1,
            "text": text,
        })
    doc.close()
    return results


def parse_pages(pdf_path: str, pages: list[int]) -> list[dict]:
    """Extract text from specific pages only.

    Args:
        pdf_path: Path to the PDF file.
        pages: List of 1-based page numbers.

    Returns:
        Same format as parse(), but only for the requested pages.
    """
    doc = pymupdf.open(pdf_path)
    total = len(doc)
    results = []
    for page_num in pages:
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= total:
            continue
        text = _extract_page_text(doc[page_idx])
        results.append({
            "pdf_page": page_num,
            "text": text,
        })
    doc.close()
    return results


def detect_page_offset(pdf_path: str) -> int:
    """Detect how many PDF pages precede the page printed as '1'.

    Finds the first page with a standalone number in the header/footer
    at a margin position (left/right edge), then calculates the offset.
    Returns the offset such that:
        printed_page = pdf_page - offset

    Returns 0 if detection fails (caller should ask the user).
    """
    doc = pymupdf.open(pdf_path)
    max_scan = min(40, len(doc))

    for page_idx in range(max_scan):
        page = doc[page_idx]
        page_height = page.rect.height
        page_width = page.rect.width
        data = page.get_text("dict", sort=True)

        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text.isdigit():
                        continue
                    bbox = span["bbox"]
                    y_center = (bbox[1] + bbox[3]) / 2
                    x_center = (bbox[0] + bbox[2]) / 2

                    in_header = y_center < page_height * 0.10
                    in_footer = y_center > page_height * 0.90
                    at_edge = x_center < page_width * 0.15 or x_center > page_width * 0.85

                    if (in_header or in_footer) and at_edge:
                        printed = int(text)
                        offset = page_idx + 1 - printed
                        if offset >= 0:
                            doc.close()
                            return offset

    doc.close()
    return 0


def get_page_count(pdf_path: str) -> int:
    """Return the total number of pages in the PDF."""
    doc = pymupdf.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def has_text_layer(pdf_path: str) -> bool:
    """Check whether the PDF has extractable text (not just scanned images)."""
    doc = pymupdf.open(pdf_path)
    sample_indices = [0, len(doc) // 2, min(len(doc) - 1, 20)]
    for idx in sample_indices:
        if idx < len(doc):
            text = doc[idx].get_text().strip()
            if len(text) > 50:
                doc.close()
                return True
    doc.close()
    return False
