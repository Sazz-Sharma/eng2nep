from pathlib import Path
import re
import unicodedata
import hashlib
from collections import defaultdict

def normalize_line(s: str) -> str:
    # Unicode normalize, trim, collapse whitespace
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def analyze_file_quality(
    file_path: str,
    max_lines_to_show: int = 10,
    max_len: int = 2000,
    min_len: int = 1,
    track_duplicates: bool = True,
    max_unique_hashes: int = 2_000_000,  # stop tracking if corpus is huge
):
    stats = {
        "lines": 0,
        "chars": 0,
        "empty": 0,
        "short": 0,
        "long": 0,
        "whitespace_only": 0,
        "non_ascii_lines": 0,
        "control_char_lines": 0,
    }
    sample_lines = []
    # Duplicate tracking by normalized line hash (memory aware)
    seen_hashes = set()
    dup_counts = defaultdict(int)
    dup_samples = {}  # hash -> example text

    def line_has_controls(s: str) -> bool:
        return bool(re.search(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", s))

    def ascii_ratio(s: str) -> float:
        if not s:
            return 1.0
        ascii_chars = sum(1 for ch in s if ord(ch) < 128)
        return ascii_chars / len(s)

    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            for line in f:
                stats["lines"] += 1
                stats["chars"] += len(line)

                raw = line.rstrip("\n")
                if not raw:
                    stats["empty"] += 1

                norm = normalize_line(raw)
                if not norm:
                    stats["whitespace_only"] += 1

                if len(norm) < min_len:
                    stats["short"] += 1
                if len(norm) > max_len:
                    stats["long"] += 1

                if ascii_ratio(norm) < 0.5:
                    stats["non_ascii_lines"] += 1

                if line_has_controls(norm):
                    stats["control_char_lines"] += 1

                if len(sample_lines) < max_lines_to_show:
                    sample_lines.append(norm[:200])

                if track_duplicates:
                    # Stop if unique set grows too big to avoid OOM
                    if len(seen_hashes) <= max_unique_hashes:
                        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                        if h in seen_hashes:
                            dup_counts[h] += 1
                            if h not in dup_samples:
                                dup_samples[h] = norm
                        else:
                            seen_hashes.add(h)

                if stats["lines"] % 100_000 == 0:
                    print(f"Processed {stats['lines']:,} lines...")

        print(f"\nFile: {file_path}")
        print(f"Total lines: {stats['lines']:,}")
        print(f"Total characters: {stats['chars']:,}")
        if stats["lines"] > 0:
            print(f"Average line length: {stats['chars']/stats['lines']:.1f} chars")

        print("\nQuality flags:")
        print(f"- Empty lines:           {stats['empty']:,}")
        print(f"- Whitespace-only:       {stats['whitespace_only']:,}")
        print(f"- Too short (<{min_len}): {stats['short']:,}")
        print(f"- Too long  (>{max_len}): {stats['long']:,}")
        print(f"- Non-ASCII heavy lines: {stats['non_ascii_lines']:,}")
        print(f"- Control char lines:    {stats['control_char_lines']:,}")

        if track_duplicates:
            total_dupe_lines = sum(dup_counts.values())
            unique_lines_tracked = len(seen_hashes)
            print("\nDuplicates:")
            print(f"- Unique lines tracked:  {unique_lines_tracked:,}")
            print(f"- Duplicate line count:  {total_dupe_lines:,}")
            if dup_counts:
                top = sorted(dup_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                print("- Top repeated lines (normalized):")
                for i, (h, c) in enumerate(top, 1):
                    print(f"  {i:2d}. x{c+1} -> {dup_samples.get(h, '')[:160]}")

        print(f"\nFirst {len(sample_lines)} sample lines:")
        for i, s in enumerate(sample_lines, 1):
            suffix = "..." if len(s) == 200 else ""
            print(f"{i:2d}: {s}{suffix}")

        print("-" * 60)

    except UnicodeDecodeError as e:
        print(f"Decode error in {file_path}: {e}. Try opening with errors='replace' or cleaning the file.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    data_corpus = ["dataset/archive/1_Eng.txt", "dataset/archive/1_Nepali.txt"]
    for fp in data_corpus:
        analyze_file_quality(fp, max_lines_to_show=5, max_len=2000, min_len=1, track_duplicates=True)