import sys
import time
import json
from pathlib import Path
from config import local_params as lp
import os
import contextlib

# --- Output suppression context manager ---
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python is required. Install with: pip install llama-cpp-python")
    sys.exit(1)

class LlamaCppModel:
    def __init__(self, model_path, n_ctx=8192, max_tokens=2048, temperature=0.8, n_threads=None):
        with suppress_output():
            self.client = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,  # CPU only
                verbose=False,
            )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt, stop=None):
        stop = stop or ["}"]  # try to stop at end of JSON
        output = self.client(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop,
        )
        return output["choices"][0]["text"]

# --- Prompt for JSON gold template ---
gold_prompt = '''
You are a document designer. Output a JSON object describing a 1-2 page PDF document for a business or scientific report, using the following structure:

{
  "meta": {"title": "...", "page_size": "LETTER", "columns": 1, "margins_pt": [72,72,72,72]},
  "styles": {"base_font": "Helvetica", "base_size": 10, "heading_font": "Helvetica-Bold", "heading_sizes": {"h1": 16, "h2": 13}},
  "pages": [
    {"elements": [
      {"type": "heading", "level": "h1", "text": "..."},
      {"type": "paragraph", "text": "..."},
      {"type": "table", "rows": [["Year","A","B","C"],["2023","10","11","12"]], "col_widths": "auto"},
      {"type": "figure", "caption": "Example figure", "height_pt": 120},
      {"type": "paragraph", "text": "..."}
    ]}
  ]
}

Include at least one heading, two paragraphs, one table, and one figure. Output only valid JSON, no markdown or comments.
'''

model = LlamaCppModel(
    model_path=lp.llm_location,
    n_ctx=8192,
    max_tokens=2048,
    temperature=0.8,
)

print("Generating gold template JSON with LLM...")
gold_json = model.generate(gold_prompt, stop=None)  # No stop sequence

# --- Strip markdown code block markers ---
def strip_code_blocks(text: str) -> str:
    lines = text.splitlines()
    filtered = [line for line in lines if not line.strip().startswith('```')]
    return '\n'.join(filtered)

gold_json = strip_code_blocks(gold_json)

# --- Auto-fix truncated JSON ---
def auto_fix_json(text: str) -> str:
    # Add missing closing brackets/braces if needed
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')
    fixed = text
    if close_braces < open_braces:
        fixed += '}' * (open_braces - close_braces)
    if close_brackets < open_brackets:
        fixed += ']' * (open_brackets - close_brackets)
    return fixed

# --- Try to parse and pretty-print JSON ---
try:
    gold_obj = json.loads(gold_json)
    auto_fixed = False
except json.JSONDecodeError:
    # Try to auto-fix
    gold_json_fixed = auto_fix_json(gold_json)
    try:
        gold_obj = json.loads(gold_json_fixed)
        auto_fixed = True
    except json.JSONDecodeError as e:
        print("[ERROR] LLM did not return valid JSON.\n", gold_json)
        print("JSON error:", e)
        sys.exit(1)

if 'auto_fixed' in locals() and auto_fixed:
    print("[WARNING] JSON was auto-fixed for missing brackets/braces.")

# --- Save JSON to gold_out/llm_doc.gold.json ---
out_dir = Path("gold_out")
out_dir.mkdir(exist_ok=True)
timestamp = int(time.time())
json_path = out_dir / f"llm_doc_{timestamp}.gold.json"
json_path.write_text(json.dumps(gold_obj, indent=2), encoding="utf-8")
print(f"Success! Gold template saved to {json_path}")
