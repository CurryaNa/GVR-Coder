import json
import argparse
import re
from typing import Dict, List, Any


def extract_svg_from_text(text: str) -> str:
    """
    Extracts the first <svg> block found in the provided text.
    """
    if not isinstance(text, str):
        return ""
    # Use re.DOTALL to ensure the dot matches newlines within the SVG block
    svg_pattern = re.compile(r"<svg [^>]*>.*?</svg>", re.DOTALL)
    match = svg_pattern.search(text)
    return match.group() if match else ""


def process_jsonl(input_path: str, output_path: str, output_dir: str) -> None:
    """
    Parses a JSONL file, extracts SVG code, and saves the structured results to a JSON file.
    """
    result_list: List[Dict[str, Any]] = []
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    line_data = json.loads(line)
                    data_id = line_data.get("data_id", "")
                    md_content = line_data.get("md_content", "")
                    dataset_name = line_data.get("dataset_name", "")

                    svg_code = ""
                    raw_answers = line_data.get("answers", [])
                    if raw_answers:
                        # Extract SVG from the first model response
                        svg_code = extract_svg_from_text(raw_answers[0])

                    # Construct a standardized image path for the next stage
                    image_path = f"{output_dir}/jpg0/{data_id}.png" if data_id else ""
                    
                    result_list.append(
                        {
                            "data_id": data_id,
                            "dataset_name": dataset_name,
                            "md_content": md_content,
                            "svg_code": svg_code,
                            "image_path": image_path,
                        }
                    )
                except json.JSONDecodeError:
                    print(f"Warning: JSON decode error at line {line_num}, skipping.")
                except Exception as e:
                    print(f"Warning: Failed to process line {line_num}. Error: {str(e)}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"Processing complete! Extracted {len(result_list)} valid records to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SVGs from repair JSONL output and format as input for the next critic iteration."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path for the output JSON file")
    parser.add_argument(
        "--output-dir", 
        "-d", 
        required=True, 
        help="Current iteration directory (e.g., .../dataset_rp1)"
    )
    
    args = parser.parse_args()
    process_jsonl(args.input, args.output, args.output_dir)


if __name__ == "__main__":
    main()