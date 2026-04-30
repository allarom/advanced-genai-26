from docx import Document
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/read_docx.py <input.docx> [output.txt]")
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    doc = Document(str(input_path))
    lines = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            lines.append(text)

    out_text = "\n".join(lines)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2]).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(out_text, encoding="utf-8")
        print(f"Wrote: {output_path}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
