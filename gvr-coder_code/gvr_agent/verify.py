import os
import json
import pandas as pd
import argparse
from pprint import pprint
from pydantic import BaseModel
from loguru import logger


from infer import BatchInferOpenAI 


os.environ.pop("no_proxy", None)
os.environ.pop("NO_PROXY", None)

def process_eval_infer_example(example, output_dir: str):
    data_id = example["data_id"]
    svg = example["svg_code"]
    image_path = os.path.join(output_dir, "images", f"{data_id}.jpg")
    caption = example["md_content"]

    user_content = f"""
<caption>
{caption}
</caption>

Initial SVG Code:
<svg_code>
{svg}
</svg_code>
This image is the rendered result of the initial code:
<image>"""

    messages = [{"role": "user", "content": user_content}]
    example["messages"] = messages
    example["images"] = [image_path]
    return example

class VerifierOutput(BaseModel):
    attributes: list[str] = None
    problems: str = ""
    needs_repair: bool
    scores: int = -1

    def model_post_init(self, __context):
        self.scores = self.calculate_score()

    def calculate_score(self) -> int:
        if not self.attributes:
            return 6
        score = 6 - len(self.attributes)
        return max(0, score)

def process_eval_extract_example(example: dict) -> dict[str, str | int]:
    answers = example["answers"]
    verifier_outputs = []
    for answer in answers:
        answer_dict = json.loads(answer) if isinstance(answer, str) else answer
        answer_dict["attributes"] = answer_dict.get("attributes", []) or []
        
        result = VerifierOutput.model_validate(answer_dict)
        verifier_outputs.append(result)
    
    example["verifier_outputs"] = verifier_outputs
    for i, verifier_output in enumerate(verifier_outputs):
        example[f"verify_{i}"] = verifier_output.scores
    return example

def check_and_format_verifier_output(answer: str) -> str:
    try:
        verifier_output = VerifierOutput.model_validate_json(answer)
        return verifier_output.model_dump_json(ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Verifier output parsing error: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="SVG Quality Evaluation Script")
    parser.add_argument("--output-dir", required=True, help="Directory for storing intermediate assets and logs")
    parser.add_argument("--data-path", required=True, help="Path to the input JSON dataset")
    parser.add_argument("--predict-path", required=True, help="Path to save the model predictions (JSONL)")
    parser.add_argument("--config-path", required=True, help="Path to the model inference configuration")
    parser.add_argument("--prompt-key", default="xxx", help="Prompt template identifier")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--enable-infer", action="store_true", help="Run model inference")
    parser.add_argument("--excel-path", default=None, help="Optional: Path to save evaluation statistics in Excel")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.predict_path), exist_ok=True)

    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    dataset = raw_data if isinstance(raw_data, list) else [raw_data]
    dataset = [process_eval_infer_example(ex, args.output_dir) for ex in dataset]

    if args.enable_infer:
        client = BatchInferOpenAI(
            config_path=args.config_path,
            prompt_key=args.prompt_key,
            n_workers=args.n_workers,
            max_retries=args.max_retries,
            check_and_format_func=check_and_format_verifier_output,
        )
        client.batch_infer(dataset=dataset, output_path=args.predict_path)

    if not os.path.exists(args.predict_path):
        logger.error(f"Prediction file {args.predict_path} not found.")
        return

    dataset_eval = [json.loads(line) for line in open(args.predict_path, "r", encoding="utf-8").readlines()]
    dataset_eval = [process_eval_extract_example(ex) for ex in dataset_eval]
    dataset_eval.sort(key=lambda x: x.get("data_id", 0))
    
    n_generation = len(dataset_eval[0]["answers"])
    bon_list = [bon for bon in [1, 3, 5, 10] if bon <= n_generation]

    bon_records = []
    for bon in bon_list:
        score_count = [0] * 7
        total_score = 0
        for example in dataset_eval:
            selected = example["verifier_outputs"][:bon]
            max_score = max([o.scores for o in selected])
            score_count[max_score] += 1
            total_score += max_score

        avg_score = (total_score / len(dataset_eval)) / 6
        accept_rate = score_count[6] / len(dataset_eval)

        record = {
            "dataset": os.path.basename(args.data_path),
            "BoN": bon,
            "avg_score_normalized": round(avg_score, 4),
            "pass_rate": round(accept_rate, 4),
            "sample_size": len(dataset_eval),
        }
        for s in range(7):
            record[f"count_score_{s}"] = score_count[s]
        bon_records.append(record)

    df_bon_stat = pd.DataFrame(bon_records)
    print("\n--- BoN Statistics ---")
    print(df_bon_stat)

    if args.excel_path:
        os.makedirs(os.path.dirname(args.excel_path), exist_ok=True)
        with pd.ExcelWriter(args.excel_path) as writer:
            df_bon_stat.to_excel(writer, sheet_name="Summary", index=False)
        logger.info(f"Statistics saved to {args.excel_path}")

if __name__ == "__main__":
    main()