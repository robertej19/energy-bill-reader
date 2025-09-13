from dataclasses import dataclass
from pathlib import Path

@dataclass
class LocalParams:
    output_directory: Path = Path("output")
    pdf_directory: Path = Path("pdfs")
    local_llm_model: str = "gpt-4o-mini"
    local_llm_path: Path = Path("llm")
    latex_pdf_directory: Path = Path("/home/rober/synth-reader/input_data/synth_pdfs/latex_pdf")
    image_pdf_directory: Path = Path("/home/rober/synth-reader/input_data/synth_pdfs/image_pdf")
    real_pdf_directory: Path = Path("/home/rober/synth-reader/real_pdfs")

    llm_location: Path = Path("/home/rober/UNIVERSAL_UTILITIES/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")

    
local_params = LocalParams()
