import json
import argparse
from typing import Dict, List, Any


def parse_answers(answers: List[str]) -> Dict[str, Any]:
    """
    Parses the first element of the answers list as JSON and extracts specific fields.
    """
    parsed_answers = {"attributes": [], "problems": "", "needs_repair": None}
    if not answers:
        return parsed_answers
    try:
        answer_str = answers[0].strip()
        answer_json = json.loads(answer_str)
        parsed_answers["attributes"] = answer_json.get("attributes", [])
        parsed_answers["problems"] = answer_json.get("problems", "")
        parsed_answers["needs_repair"] = answer_json.get("needs_repair", None)
    except (json.JSONDecodeError, IndexError, TypeError):
        print(f"Warning: Failed to parse 'answers' field. Content: {answers}")
    return parsed_answers


def process_jsonl(input_path: str, output_path: str) -> None:
    """
    Reads a JSONL file, extracts critic fields, and saves the formatted list to a JSON file.
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
                    svg_code = line_data.get("svg_code", "")
                    raw_answers = line_data.get("answers", [])
                    
                    parsed = parse_answers(raw_answers)
                    
                    result_list.append(
                        {
                            "data_id": data_id,
                            "dataset_name": dataset_name,
                            "md_content": md_content,
                            "svg_code": svg_code,
                            "attributes": parsed["attributes"],
                            "problems": parsed["problems"],
                            "needs_repair": parsed["needs_repair"],
                        }
                    )
                except json.JSONDecodeError:
                    print(f"Warning: JSON format error at line {line_num}, skipping.")
                except Exception as e:
                    print(f"Warning: Failed to process line {line_num}. Error: {str(e)}, skipping.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"Processing complete! Parsed {len(result_list)} valid records. Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract critic fields from JSONL and output a structured JSON file.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path for the output JSON file")
    args = parser.parse_args()
    process_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()