import sys
from pathlib import Path
from config import local_params as lp
import subprocess
import time
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

# --- LlamaCppModel (minimal, CPU only, quiet load) ---
try:
    from llama_cpp import Llama
except ImportError:
    print("llama-cpp-python is required. Install with: pip install llama-cpp-python")
    sys.exit(1)

class LlamaCppModel:
    def __init__(self, model_path, n_ctx=8192, max_tokens=1024, temperature=0.2, n_threads=None):
        with suppress_output():
            self.client = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,  # CPU only
                verbose=False,   # Suppress Python-side logs
            )
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt, stop=None):
        stop = stop or ["\\end{document}"]
        output = self.client(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop,
        )
        return output["choices"][0]["text"]

# --- Prompt for LaTeX ---
latex_prompt = (
    """
You are a LaTeX expert. Write a complete, short, but content-rich LaTeX document (article class) that is about 2 pages long when compiled. 
Include a title, author, date, at least two sections, and a table. 
Do not include any code explanations or comments. Output only valid LaTeX code, starting with \\documentclass and ending with \\end{document}.
"""
)

# --- Generate LaTeX ---
model = LlamaCppModel(
    model_path=lp.llm_location,
    n_ctx=8192,
    max_tokens=2048,
    temperature=2,
)

print("Generating LaTeX with LLM...")
latex_code = model.generate(latex_prompt)

# --- Strip markdown code block markers ---
def strip_code_blocks(text: str) -> str:
    lines = text.splitlines()
    # Remove lines that are exactly ```latex, ``` or similar
    filtered = [line for line in lines if not line.strip().startswith('```')]
    return '\n'.join(filtered)

latex_code = strip_code_blocks(latex_code)

# Ensure document starts/ends correctly
if not latex_code.strip().startswith("\\documentclass"):
    latex_code = "\\documentclass[11pt]{article}\n" + latex_code
if not latex_code.strip().endswith("\\end{document}"):
    latex_code += "\n\\end{document}\n"

print(latex_code)

# --- LaTeX validity check ---
def is_valid_latex(tex: str) -> bool:
    # Basic checks: starts/ends, balanced braces
    if not tex.strip().startswith("\\documentclass"):
        print("[ERROR] LaTeX does not start with \\documentclass.")
        return False
    if not tex.strip().endswith("\\end{document}"):
        print("[ERROR] LaTeX does not end with \\end{document}.")
        return False
    # Check for balanced curly braces
    stack = []
    for c in tex:
        if c == '{':
            stack.append(c)
        elif c == '}':
            if not stack:
                print("[ERROR] Unmatched closing brace '}' in LaTeX.")
                return False
            stack.pop()
    if stack:
        print("[ERROR] Unmatched opening brace '{' in LaTeX.")
        return False
    return True

if not is_valid_latex(latex_code):
    print("[ERROR] Invalid LaTeX detected. Aborting PDF generation.")
    sys.exit(1)

# --- Save LaTeX to file ---
timestamp = int(time.time())
tex_filename = f"llm_{timestamp}.tex"
pdf_filename = f"llm_{timestamp}.pdf"
tex_path = Path.cwd() / tex_filename
tex_path.write_text(latex_code, encoding="utf-8")

# --- Compile PDF using pdf_generator.py logic ---
print(f"Compiling {tex_path} to PDF...")
cmd = [sys.executable, str(Path(__file__).parent / "pdf_generator.py"), str(tex_path)]
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError:
    print("PDF compilation failed.")
    sys.exit(1)

# --- Move PDF to synth_pdfs/latex_pdf ---
synth_pdfs_dir = lp.latex_pdf_directory
synth_pdfs_dir.mkdir(parents=True, exist_ok=True)
generated_pdf = Path.cwd() / "output.pdf"
if generated_pdf.exists():
    final_pdf_path = synth_pdfs_dir / pdf_filename
    generated_pdf.rename(final_pdf_path)
    print(f"Success! PDF at: {final_pdf_path}")
else:
    print("PDF was not generated as expected.")
