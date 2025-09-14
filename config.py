from dataclasses import dataclass
from pathlib import Path

@dataclass
class LocalParams:
    gold_input_data_location: Path = Path("input_data/synthetic_data/gold_pdfs")
    llm_location: Path = Path("/home/rober/UNIVERSAL_UTILITIES/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")

    
local_params = LocalParams()
