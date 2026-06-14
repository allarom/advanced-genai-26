"""Extract PDF text using PyPDF2, with optional page range and language hint.

Usage:
    python scripts/extract_pdf.py "<path>" [output.md] [--pages 1-10] [--lang en]
"""

import sys, os, re, pathlib

def extract_pdf(path: str, pages=None):
    try:
        import PyPDF2
    except ImportError:
        print("PyPDF2 not installed. Install with: pip install PyPDF2")
        sys.exit(1)

    reader = PyPDF2.PdfReader(path)
    total = len(reader.pages)
    selected = range(total) if pages is None else pages

    parts = []
    for i in selected:
        if 0 <= i < total:
            text = reader.pages[i].extract_text() or ""
            parts.append(text)
    return "\n\n".join(parts)

def parse_pages(arg: str):
    """Parse strings like '1-5', '1,3,5', '1-3,7'."""
    pages = set()
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a) - 1, int(b)))
        else:
            pages.add(int(part) - 1)
    return sorted(pages)

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    pdf_path = args[0]
    output_path = None
    pages = None
    lang = "en"

    i = 1
    while i < len(args):
        if args[i].lower() == "--pages" and i + 1 < len(args):
            pages = parse_pages(args[i + 1])
            i += 2
        elif args[i].lower() == "--lang" and i + 1 < len(args):
            lang = args[i + 1]
            i += 2
        elif output_path is None and not args[i].startswith("--"):
            output_path = args[i]
            i += 1
        else:
            i += 1

    text = extract_pdf(pdf_path, pages=pages)

    if output_path:
        pathlib.Path(output_path).write_text(text, encoding="utf-8")
        print(f"Extracted {len(text)} characters to {output_path}")
    else:
        print(text)

if __name__ == "__main__":
    main()
