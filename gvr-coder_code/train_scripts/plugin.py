import os
import json
import uuid
import time
import threading
import re
import math
import base64
import datetime
import concurrent.futures
from typing import Optional, Dict, Any, List, Tuple

# --- Third-party Libraries ---
from loguru import logger
from bs4 import BeautifulSoup
from pydantic import BaseModel
from openai import OpenAI

# --- Swift Libraries ---
from swift.plugin import ORM, orms

# ==========================================
# Part 1: Configuration / Constants
# ==========================================


os.environ.pop("no_proxy", None)
os.environ.pop("NO_PROXY", None)

MAX_WORKERS = 60
IMAGE_WIDTH = 1080
DEFAULT_IMAGE_WIDTH = 512

OUTPUT_DIR = "./output/svg_eval"
CONFIG_PATH = "./config.yaml"
PROMPT_KEY = "xxx" 
INFER_WORKERS = 32
MAX_RETRIES = 5

RETRY_LOG_FILENAME = "critic_retry_stats.jsonl"

# ==========================================
# Part 1.5: Retry Logger 
# ==========================================

class CriticRetryLogger:
    def __init__(self, output_dir: str, filename: str):
        self.filepath = os.path.join(output_dir, filename)
        self.lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(os.path.dirname(self.filepath)):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def log(self, data_id: str, status: str, score: float, elapsed: float, error_msg: str = ""):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data_id": data_id,
            "status": status,
            "score": score,
            "elapsed_seconds": round(elapsed, 4),
            "max_retries_config": MAX_RETRIES,
            "error_msg": error_msg
        }
        json_line = json.dumps(entry, ensure_ascii=False)
        try:
            with self.lock:
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
        except Exception as e:
            logger.error(f"Failed to write retry log: {e}")

_retry_logger = CriticRetryLogger(OUTPUT_DIR, RETRY_LOG_FILENAME)

# ==========================================
# Part 2: SVG Verifier Base
# ==========================================

class SVGVerifierOutput(BaseModel):
    attributes: list[str] | None = None
    problems: str = ""
    needs_repair: bool
    score: int = -1

    def model_post_init(self, __context):
        self.score = self.calculate_score()
        if self.attributes is None:
            self.attributes = []
        if not self.problems:
            self.problems = ""

    def calculate_score(self) -> int:
        if not self.attributes:
            return 6
        score = 6 - len(self.attributes)
        return max(0, score)

def check_and_format_verifier_output(answer: str) -> dict | None:
    try:
        verifier_output = SVGVerifierOutput.model_validate_json(answer)
        return verifier_output.model_dump()
    except Exception as e:
        logger.warning(f"Verifier output parsing error: {e=} {answer=}")
        return None

# ==========================================
# Part 3: SVG Utils & Render Mock
# ==========================================

def svg2html(svg: str) -> str:
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8" />
    <style>
        html, body {{ margin: 0; padding: 0; overflow: hidden; }}
        svg {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
    </style>
</head>
<body>
    {svg}
</body>
</html>"""
    return html

def check_and_format_svg(svg: str) -> str:
    svgs = re.findall(r"<svg [^>]*>.*?</svg>", svg, re.DOTALL)
    if not svgs or len(svgs) != 1:
        return ""
    
    soup = BeautifulSoup(svgs[0], "xml")
    svg_tag = soup.find("svg")
    return svg_tag.prettify() if svg_tag else ""


class HtmlRenderer:
    def __init__(self, width: int = 512):
        self.width = width

    def start(self):
        pass # Initialize headless browser here

    def render_to_file(self, html: str, output_path: str):
        # TODO: Implement actual rendering logic 
        # For demonstration, creating a dummy file to bypass error checks.
        with open(output_path, 'wb') as f:
            f.write(b"dummy_image_data")

# ==========================================
# Part 4: OpenAI VLM Client & Scorer Logic
# ==========================================

_thread_local = threading.local()
_init_lock = threading.Lock()

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)

def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Failed to remove existing file: {path}. err={e}")

def get_renderer(image_width: int = 512) -> HtmlRenderer:
    if not hasattr(_thread_local, "renderer"):
        logger.info(f"Initializing generic HtmlRenderer for thread {threading.get_ident()}")
        _thread_local.renderer = HtmlRenderer(width=image_width)
        _thread_local.renderer.start()
    return _thread_local.renderer

def _render_svg_to_image(svg: str, image_path: str, image_width: int = 512) -> bool:
    _ensure_parent_dir(image_path)
    _safe_remove(image_path)
    html = svg2html(svg)
    try:
        get_renderer(image_width=image_width).render_to_file(html=html, output_path=image_path)
    except Exception as e:
        logger.warning(f"Render failed: {image_path}, err={e}")
        return False
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        return False
    return True

class OpenAIVLMClient:
    def __init__(self, model_name: str = "model_id", max_retries: int = MAX_RETRIES):
        # Make sure OPENAI_API_KEY is set in your environment variables
        self.client = OpenAI()
        self.model_name = model_name
        self.max_retries = max_retries

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def infer_one_example(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raw_messages = inputs.get("messages", [])
        image_paths = inputs.get("images", [])

        openai_messages = []
        for msg in raw_messages:
            role = msg.get("role", "user")
            content_text = msg.get("content", "")

            if role == "user" and image_paths:
                content_list = [{"type": "text", "text": content_text}]
                for img_path in image_paths:
                    try:
                        base64_image = self._encode_image(img_path)
                        content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Failed to encode image {img_path}: {e}")
                
                openai_messages.append({"role": role, "content": content_list})
            else:
                openai_messages.append({"role": role, "content": content_text})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    response_format={"type": "json_object"} # Require JSON output
                )
                return {"answers": [response.choices[0].message.content]}
            except Exception as e:
                logger.warning(f"OpenAI API attempt {attempt+1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)

_critic_client: Optional[OpenAIVLMClient] = None

def process_critic_example(user_prompt: str, svg: str, image_path: str) -> Dict[str, Any]:
    caption = user_prompt
    user_content = f"""
<caption>
{caption}
</caption>

Initial SVG Code：
<svg_code>
{svg}
</svg_code>
This image is the rendered result of the initial code. Please evaluate and score based on both the image and the code.
Strictly output in JSON format, including the following fields: attributes (list), problems (string), needs_repair (boolean), score (integer).
<image>""".lstrip()

    return {
        "messages": [{"role": "user", "content": user_content}], 
        "images": [image_path]
    }

def get_critic_client() -> OpenAIVLMClient:
    global _critic_client
    with _init_lock:
        if _critic_client is None:
            logger.info("Initializing OpenAIVLMClient...")
            _critic_client = OpenAIVLMClient(max_retries=MAX_RETRIES)
        return _critic_client

def score_svg(
    user_prompt: str,
    svg: str,
    output_dir: str,
    image_width: int = 1080,
    data_id: Optional[str] = None,
    render_fail_score: float = 0.0,
    critic_fail_score: float = 0.0,
    keep_image: bool = True,
) -> float:
    critic_client = get_critic_client()
    start_time = time.time()
    
    if not data_id:
        data_id = f"unknown_{uuid.uuid4().hex}"

    image_path = os.path.join(output_dir, "cache", f"{data_id}.jpg")

    # 1. render
    t0 = time.time()
    ok = _render_svg_to_image(svg=svg, image_path=image_path, image_width=image_width)
    t_render = time.time() - t0

    if not ok:
        elapsed = time.time() - start_time
        _retry_logger.log(data_id, "render_fail", render_fail_score, elapsed, "Render returned False")
        return render_fail_score

    critic_input = process_critic_example(user_prompt=user_prompt, svg=svg, image_path=image_path)

    # 2. infer
    t1 = time.time()
    try:
        critic_result = critic_client.infer_one_example(critic_input)
    except Exception as e:
        logger.warning(f"[score_svg] critic infer error: {e}")
        elapsed = time.time() - start_time
        _retry_logger.log(data_id, "critic_error", critic_fail_score, elapsed, str(e))
        if not keep_image: _safe_remove(image_path)
        return critic_fail_score
    
    t_infer = time.time() - t1
    if not keep_image: _safe_remove(image_path)

    # 3. parse
    if not critic_result or not critic_result.get("answers"):
        elapsed = time.time() - start_time
        _retry_logger.log(data_id, "empty_response", critic_fail_score, elapsed, "No answers in result")
        return critic_fail_score

    try:
        ans0 = critic_result["answers"][0]
        ans0_obj = json.loads(ans0) if isinstance(ans0, str) else ans0
        output = SVGVerifierOutput.model_validate(ans0_obj)

        raw_score = max(0.0, min(float(output.score), 6.0))
        score = raw_score / 6.0

        logger.info(f"ID: {data_id} | RawScore: {raw_score} | NormScore: {score:.4f} | Total: {t_render + t_infer:.2f}s")
        _retry_logger.log(data_id, "success", score, time.time() - start_time)
        return score
        
    except Exception as e:
        logger.warning(f"[score_svg] parse verifier output error: {e}")
        _retry_logger.log(data_id, "parse_fail", critic_fail_score, time.time() - start_time, str(e))
        return critic_fail_score


# ==========================================
# Part 5: Swift Plugin / ORM
# ==========================================

class SVGScoreORM(ORM):
    def __init__(self):
        get_critic_client()

    def _extract_svg(self, content: str) -> str:
        if not content: return ""
        svgs = re.findall(r"<svg [^>]*>.*?</svg>", content, re.DOTALL)
        return svgs[0] if len(svgs) == 1 else ""

    def _process_single_item(self, args: Tuple[str, str, Any]) -> float:
        svg_content, prompt, uid = args
        valid_svg = self._extract_svg(svg_content)
        if not valid_svg: return 0.0

        try:
            base_uid = str(uid) if uid is not None else uuid.uuid4().hex
            time_step = time.time_ns() 
            random_hex = uuid.uuid4().hex[:8]
            unique_run_id = f"{base_uid}_{time_step}_{random_hex}"

            s = score_svg(
                user_prompt=prompt,
                svg=valid_svg,
                output_dir=OUTPUT_DIR,
                image_width=IMAGE_WIDTH,
                data_id=unique_run_id, 
            )
            return float(max(0.0, min(float(s) if s else 0.0, 1.0)))
        except Exception as e:
            logger.error(f"[SVGScoreORM] ID {uid} score exception: {e}")
            return 0.0

    def __call__(self, completions: List[str], user_prompt: List[str], data_id: List[str], **kwargs) -> List[float]:
        tasks = list(zip(completions, user_prompt, data_id))
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            rewards = list(executor.map(self._process_single_item, tasks))
        return rewards

orms["svg_scorer"] = SVGScoreORM


# ==========================================================
# Part 6: Complexity ORM 
# ==========================================================

class SVGComplexityORM(ORM):  
    def __init__(self, scale_factor: float = 1.0, **kwargs): 
        self.scale_factor = float(scale_factor)
        logger.info(f"[Init] SVGComplexityORM: scale_factor={self.scale_factor}, threshold=0.8")

    def _extract_svg(self, content: str) -> str:
        if not content: return ""
        svgs = re.findall(r"<svg [^>]*>.*?</svg>", content, re.DOTALL)
        if not svgs: return ""
        return max(svgs, key=len)

    def _count_structural_tags(self, svg_code: str) -> int:
        if not svg_code: return 0
        pattern = r'<(path|circle|poly|ellipse|line)'
        return len(re.findall(pattern, svg_code, re.IGNORECASE))

    def _calculate_rmatch(self, gen_svg_code: str, ref_svg_code: str) -> float:
        valid_gen = self._extract_svg(gen_svg_code)
        valid_ref = ref_svg_code  
        
        if not valid_gen: return 0.0
            
        n_gen = self._count_structural_tags(valid_gen)
        n_ref = self._count_structural_tags(valid_ref)

        if n_ref == 0:
            reward = self.scale_factor if n_gen > 0 else 0.0
        else:
            threshold_val = n_ref * 0.8
            if n_gen >= threshold_val:
                reward = self.scale_factor
            else:
                reward = self.scale_factor * (n_gen / threshold_val)

        final_reward = max(0.0, min(reward, 1.0))
        logger.info(f"[Check] Gen: {n_gen} | Ref: {n_ref} | Ratio: {(n_gen/n_ref if n_ref>0 else 1.0):.2f} | Reward: {final_reward:.4f}")
        return float(final_reward)

    def __call__(self, completions: List[str], reference: List[str], **kwargs) -> List[float]:
        tasks = list(zip(completions, reference))
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            rewards = list(executor.map(lambda p: self._calculate_rmatch(p[0], p[1]), tasks))
        return rewards

orms["svg_complexity_scorer"] = SVGComplexityORM