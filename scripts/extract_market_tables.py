import re
import argparse
from pathlib import Path


def sanitize_filename(caption: str) -> str:
    """Sanitize a string to be used as a filename."""
    # Remove LaTeX commands, replace spaces with underscores, and remove special characters
    caption = re.sub(r"\\[a-zA-Z]+", "", caption)  # Remove LaTeX commands like \toprule
    caption = re.sub(r"\$.*?\$", "", caption)  # Remove math mode content
    caption = re.sub(
        r"[^a-zA-Z0-9_ -]", "", caption
    )  # Remove most non-alphanumeric chars
    caption = caption.strip().replace(" ", "_")
    caption = re.sub(r"_+", "_", caption)  # Replace multiple underscores with single
    return f"table_{caption.lower()}"


def extract_tables(input_tex_path: Path, output_dir: Path) -> None:
    with open(input_tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find all table environments and capture the caption.
    # Literal backslashes in LaTeX commands must be escaped as \\ in the pattern.
    table_pattern = re.compile(
        r"(\\begin\{table\}.*?\\caption\{([^}]+)\}.*?\\end\{table\})",
        re.DOTALL,
    )

    found_tables = table_pattern.findall(content)

    if not found_tables:
        print(f"No tables found in {input_tex_path}")
        return

    for i, (table_block, caption) in enumerate(found_tables):
        sanitized_name = sanitize_filename(caption)
        output_filepath = output_dir / f"{sanitized_name}.tex"

        with open(output_filepath, "w", encoding="utf-8") as out_f:
            out_f.write(table_block + "\n")  # Add a newline for safety

        print(f"Extracted table '{caption}' to {output_filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract tables from a .tex file into separate .tex files.",
    )
    parser.add_argument(
        "input_tex_path",
        type=Path,
        help="Path to the input .tex file (e.g., results/market_outcomes.tex)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the extracted table .tex files (e.g., results/)",
    )
    args = parser.parse_args()

    if not args.input_tex_path.exists():
        print(f"Error: Input .tex file not found at {args.input_tex_path}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    extract_tables(args.input_tex_path, args.output_dir)


if __name__ == "__main__":
    main()
