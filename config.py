from dataclasses import dataclass
from pathlib import Path

@dataclass
class LocalParams:
    output_directory: Path = Path("output")
    pdf_directory: Path = Path("pdfs")
    local_llm_model: str = "gpt-4o-mini"
    local_llm_path: Path = Path("llm")

local_params = LocalParams()
