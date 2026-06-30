import os
import json
import argparse
from pprint import pprint
from loguru import logger


from infer import BatchInferOpenAI 

from utils.svg_handler import check_and_format_svg


os.environ.pop("no_proxy", None)
os.environ.pop("NO_PROXY", None)

def process_example(example: dict) -> dict:
    example.pop("images", None)
    
    messages = [
        {
            "role": "user",
            "content": f'''Description:
<caption>
{example["md_content"]}
</caption>

Initial SVG Code:
<svg_code>
{example["svg_code"]}
</svg_code>

Identified Issues (Attributes):
<attribute>
{example["attributes"]}
</attribute>

Analysis and Repair Suggestions:
<problems>
{example["problems"]}
</problems>
''',
        }
    ]
    example["messages"] = messages
    return example

def main():
    parser = argparse.ArgumentParser(description="SVG Repair Agent - Inference Script")
    parser.add_argument("--input-critic-json", required=True, help="Path to the JSON file containing critic feedback")
    parser.add_argument("--output-dir", required=True, help="Directory for output assets (e.g., rendered images)")
    parser.add_argument("--predict-path", required=True, help="Path to save the repaired results (JSONL)")
    parser.add_argument("--config-path", required=True, help="Inference configuration path")
    parser.add_argument("--prompt-key", default="xxx", help="Prompt template identifier")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--enable-infer", action="store_true", help="Execute model inference")
    parser.add_argument("--render", action="store_true", help="Whether to render the output SVG to images")
    parser.add_argument("--render-width", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.dirname(args.predict_path):
        os.makedirs(os.path.dirname(args.predict_path), exist_ok=True)

    with open(args.input_critic_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Original dataset size: {len(dataset)}")

    dataset_filtered = [ex for ex in dataset if ex.get("needs_repair") is True]
    logger.info(f"Filtered dataset size (needs_repair=True): {len(dataset_filtered)}")

    dataset_processed = [process_example(ex) for ex in dataset_filtered]

    if dataset_processed:
        logger.debug("Sample processed message:")
        pprint(dataset_processed[0]["messages"])
    else:
        logger.warning("No samples found that require repair.")

    if args.enable_infer and dataset_processed:
        client = BatchInferOpenAI(
            config_path=args.config_path,
            prompt_key=args.prompt_key,
            n_workers=args.n_workers,
            max_retries=args.max_retries,
            check_and_format_func=check_and_format_svg,
        )
        client.batch_infer(dataset=dataset_processed, output_path=args.predict_path)

    if args.render and os.path.exists(args.predict_path):
        logger.info("Starting rendering process...")

        render_cmd = (
            f"render -i {args.predict_path} -o {args.output_dir} "
            f"--width {args.render_width} --disable-valid --enable-svg"
        )
        os.system(render_cmd)

if __name__ == "__main__":
    main()