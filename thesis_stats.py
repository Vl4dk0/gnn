#!/usr/bin/env python
"""
Typst Thesis Word and Character Counter

This script counts the number of words and characters in Typst (.typ) files,
stripping comments, code blocks, math equations, and formatting markup to
estimate the count of actual rendered text. It also compares the counts against
the official UK/FMFI guidelines (30-40 norm pages / 54k-72k characters).
"""

import os
import re
import sys
import argparse

def clean_typst_content(content: str) -> str:
    """
    Cleans Typst content by removing code blocks, mathematical formulas,
    comments, and formatting syntax, leaving only the text that would
    be rendered as body text.
    """
    # 1. Strip comments
    # Multiline /* ... */
    content = re.sub(r'/\*[\s\S]*?\*/', ' ', content)
    # Single-line // ...
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    
    # 2. Strip code blocks
    # Multiline ``` ... ```
    content = re.sub(r'```[\s\S]*?```', ' ', content)
    # Inline ` ... `
    content = re.sub(r'`[^`]*`', ' ', content)
    
    # 3. Handle math equations
    # Block equations: Typst defines block math if it's on a line by itself, wrapped in $
    # Example:
    # $ M(k, g) = ... $
    content = re.sub(r'^\s*\$\s+[\s\S]*?\s+\$\s*$', ' ', content, flags=re.MULTILINE)
    # Inline math: keep the text inside the dollars but remove the $ symbols
    content = re.sub(r'\$([^$]+)\$', r'\1', content)
    
    # 4. Replace citations like @label with [1] so they count as 1 word and 3 characters
    content = re.sub(r'@[a-zA-Z0-9_\-:]+', '[1]', content)
    
    # 5. Remove Typst label definitions like <label>
    content = re.sub(r'<[a-zA-Z0-9_\-:]+>', ' ', content)
    
    # 6. Parse Typst expressions starting with #
    # We do a stack-based parser to strip the function names, arguments in parens (),
    # and code block braces {}, but KEEP text inside content brackets []
    output: list[str] = []
    i = 0
    n = len(content)
    stack: list[str] = ['markup']
    
    while i < n:
        char = content[i]
        curr_state = stack[-1] if stack else 'markup'
        
        if curr_state == 'markup':
            if char == '#':
                # Switch to code mode and skip the identifier (e.g. strong, let, set, table, etc.)
                stack.append('code')
                i += 1
                while i < n and (content[i].isalnum() or content[i] in ('-', '_', '.')):
                    i += 1
                continue
            elif char == ']':
                # Closing bracket of a content block
                if len(stack) > 1 and stack[-2] == 'code':
                    _ = stack.pop()  # pop 'markup'
                    _ = stack.pop()  # pop 'code'
                    i += 1
                    continue
                else:
                    output.append(char)
                    i += 1
            else:
                output.append(char)
                i += 1
                
        elif curr_state == 'code':
            if char == '(':
                stack.append('parens')
                i += 1
            elif char == '{':
                stack.append('braces')
                i += 1
            elif char == '[':
                # Open bracket inside code mode starts nested markup
                stack.append('markup')
                i += 1
            elif char == '"':
                stack.append('string')
                i += 1
            elif char.isspace() or char in (';', ',', '.'):
                # End of simple identifier/expression
                _ = stack.pop()
                # Let markup parser handle the space or punctuation
            else:
                i += 1
                
        elif curr_state == 'parens':
            if char == ')':
                _ = stack.pop()
                i += 1
            elif char == '(':
                stack.append('parens')
                i += 1
            elif char == '"':
                stack.append('string')
                i += 1
            else:
                i += 1
                
        elif curr_state == 'braces':
            if char == '}':
                _ = stack.pop()
                i += 1
            elif char == '{':
                stack.append('braces')
                i += 1
            elif char == '"':
                stack.append('string')
                i += 1
            else:
                i += 1
                
        elif curr_state == 'string':
            if char == '"' and content[i-1] != '\\':
                _ = stack.pop()
                i += 1
            else:
                i += 1
                
    cleaned = "".join(output)
    
    # 7. Additional cleanup of markup markers
    # Strip asterisks * and underscores _ used for formatting
    cleaned = cleaned.replace('*', '')
    cleaned = cleaned.replace('_', '')
    
    # Strip list item prefixes (e.g. "- ", "+ ", "1. ")
    cleaned = re.sub(r'^\s*[-+]\s+', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)
    
    # Strip heading markers "= " or "== "
    cleaned = re.sub(r'^\s*=+ ', '', cleaned, flags=re.MULTILINE)
    
    # Clean up empty lines or multiple consecutive spaces
    # But preserve single spacing between words.
    lines: list[str] = []
    for line in cleaned.splitlines():
        # Remove extra whitespace within the line
        line_clean = " ".join(line.split())
        if line_clean:
            lines.append(line_clean)
            
    return "\n".join(lines)

def count_file_stats(filepath: str) -> tuple[int, int, float]:
    """
    Reads a file, cleans it, and returns (word_count, char_count_with_spaces, norm_pages).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return 0, 0, 0.0

    cleaned = clean_typst_content(content)
    words = len(cleaned.split())
    chars = len(cleaned)
    norm_pages = chars / 1800.0
    return words, chars, norm_pages

def main() -> None:
    parser = argparse.ArgumentParser(description="Count words and characters in Typst thesis files.")
    _ = parser.add_argument(
        "--dir",
        type=str,
        default="./thesis",
        help="Path to the thesis directory containing .typ files (default: ./thesis)"
    )
    args = parser.parse_args()
    dir_val = getattr(args, "dir", "./thesis")
    dir_path = str(dir_val) if dir_val is not None else "./thesis"

    thesis_dir = os.path.abspath(dir_path)
    if not os.path.isdir(thesis_dir):
        print(f"Error: Directory '{thesis_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning directory: {thesis_dir}")

    # Gather .typ files
    typ_files: list[str] = []
    for root, _, files in os.walk(thesis_dir):
        root_str = str(root)
        for file in files:
            file_str = str(file)
            if file_str.endswith('.typ'):
                typ_files.append(os.path.join(root_str, file_str))

    if not typ_files:
        print("No .typ files found in the specified directory.")
        sys.exit(0)

    # Sort files so main.typ is first, then chapters in order
    typ_files.sort(key=lambda x: (0 if "main.typ" in x else 1, x))

    # Calculate statistics
    stats: dict[str, tuple[int, int, float]] = {}
    
    chapter_words = 0
    chapter_chars = 0
    
    total_words = 0
    total_chars = 0

    for filepath in typ_files:
        rel_path = os.path.relpath(filepath, thesis_dir)
        w, c, p = count_file_stats(filepath)
        stats[rel_path] = (w, c, p)
        
        # Accumulate totals
        total_words += w
        total_chars += c
        
        # If it's a chapter file (usually under chapters/), track it separately for the core body text
        if "chapters" in rel_path:
            chapter_words += w
            chapter_chars += c

    # Print results
    print("\n" + "=" * 80)
    print(f"{'Typst File':<40} | {'Words':<10} | {'Characters':<12} | {'Norm Pages':<10}")
    print("-" * 80)
    
    for rel_path, (w, c, p) in sorted(stats.items(), key=lambda item: (0 if "main.typ" in item[0] else 1, item[0])):
        print(f"{rel_path:<40} | {w:<10,} | {c:<12,} | {p:<10.2f}")
        
    print("=" * 80)
    
    # Guidelines comparison
    # From guidelines.md:
    # Recommended length:
    # - 30 to 40 norm pages
    # - 54,000 to 72,000 characters including spaces
    # One norm page is 1,800 characters.
    min_chars = 54000
    max_chars = 72000
    min_pages = 30.0
    max_pages = 40.0

    chapter_pages = chapter_chars / 1800.0
    total_pages = total_chars / 1800.0

    print("\nSummary and Comparison with Guidelines:")
    print("-" * 80)
    print(f"Core Body Chapters (under chapters/):")
    print(f"  - Words: {chapter_words:,}")
    print(f"  - Characters (inc. spaces): {chapter_chars:,}")
    print(f"  - Norm Pages: {chapter_pages:.2f}")
    
    print(f"\nWhole Thesis (including main.typ wrapper):")
    print(f"  - Words: {total_words:,}")
    print(f"  - Characters (inc. spaces): {total_chars:,}")
    print(f"  - Norm Pages: {total_pages:.2f}")

    print("-" * 80)
    print(f"UK/FMFI Guideline Recommendation: {min_chars:,} to {max_chars:,} characters ({min_pages:.1f} to {max_pages:.1f} norm pages)")
    print("-" * 80)
    
    # Comparison for the whole thesis
    if total_chars < min_chars:
        diff = min_chars - total_chars
        print(f"Status: BELOW RECOMMENDED LENGTH")
        print(f"The whole thesis is short by {diff:,} characters (approx. {diff / 1800.0:.2f} norm pages) to reach the minimum recommendation.")
    elif total_chars > max_chars:
        diff = total_chars - max_chars
        print(f"Status: ABOVE RECOMMENDED LENGTH")
        print(f"The whole thesis exceeds the maximum recommendation by {diff:,} characters (approx. {diff / 1800.0:.2f} norm pages).")
    else:
        print(f"Status: WITHIN RECOMMENDED RANGE")
        print("The whole thesis is perfectly within the 54,000 to 72,000 character range.")

if __name__ == "__main__":
    main()
