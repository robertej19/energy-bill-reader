# make_pdf.py
import shutil
import subprocess
from pathlib import Path
import sys
import time
from config import local_params as lp

def main():
    # Output directory for PDFs
    synth_pdfs_dir = lp.latex_pdf_directory
    synth_pdfs_dir.mkdir(parents=True, exist_ok=True)
    
    # Use Unix timestamp for filename
    timestamp = int(time.time())
    pdf_filename = f"{timestamp}.pdf"
    pdf_path = synth_pdfs_dir / pdf_filename
    
    # Work in a temp directory to avoid clutter
    out_dir = Path.cwd()
    tex_path = out_dir / "output.tex"
    temp_pdf_path = out_dir / "output.pdf"

    # Check for .tex file argument
    input_tex = None
    if len(sys.argv) > 1:
        input_tex = Path(sys.argv[1])
        if not input_tex.exists():
            print(f"Provided .tex file does not exist: {input_tex}")
            sys.exit(1)
        shutil.copy(str(input_tex), str(tex_path))
    else:
        TEX = r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{booktabs}
\title{The Role of Quantum Computing in Cryptography}
\author{John Doe}
\date{\today}
\begin{document}
\maketitle
\section{Introduction}
Quantum computing is a rapidly evolving field that promises to revolutionize various domains, including cryptography. Unlike classical computers, which process information using bits (0s and 1s), quantum computers use quantum bits or qubits, allowing for the processing of complex computations at an unprecedented speed. This capability has significant implications for cryptographic systems, which rely on the difficulty of certain mathematical problems to ensure security.

\section{Quantum Computing and Cryptography}
\subsection{Shor's Algorithm}
One of the most significant threats to traditional cryptographic systems is Shor's algorithm, a quantum algorithm that can efficiently factor large integers. This is particularly concerning for public-key cryptography, where the security of many encryption schemes, such as RSA, relies on the difficulty of factoring large numbers. Shor's algorithm can break these systems in polynomial time, whereas classical algorithms require exponential time.

\subsection{Post-Quantum Cryptography}
To mitigate the risks posed by quantum computers, researchers are developing post-quantum cryptographic algorithms that are resistant to quantum attacks. These algorithms are based on problems that are believed to be hard for both classical and quantum computers. Examples include lattice-based cryptography, code-based cryptography, and multivariate polynomial cryptography. These new cryptographic systems are designed to provide security in a post-quantum world.

\section{Conclusion}
The advent of quantum computing presents both challenges and opportunities for the field of cryptography. While it poses a significant threat to current cryptographic systems, it also drives the development of new, more secure cryptographic algorithms. As quantum technology continues to advance, it is crucial for the cryptographic community to stay informed and adapt to these new challenges.

\begin{table}[h]
\centering
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Cryptography Type} & \textbf{Example} & \textbf{Security Assumption} \\ \midrule
Public-Key & RSA & Factoring large integers \\
Post-Quantum & Lattice-based & Hardness of lattice problems \\
 & Code-based & Hardness of decoding random linear codes \\
 & Multivariate & Solving systems of multivariate polynomial equations \\ \bottomrule
\end{tabular}
\caption{Comparison of Cryptographic Systems}
\end{table}

\section*{Acknowledgments}
I would like to thank my colleagues for their valuable insights and support.

\section*{References}
\href{https://www.nist.gov/programs-projects/post-quantum-cryptography-standardization}{NIST Post-Quantum Cryptography Standardization}


\end{document}
"""
        tex_path.write_text(TEX, encoding="utf-8")
    print(f"Wrote {tex_path}")

    # Prefer latexmk (handles multiple passes + aux files) else fallback to pdflatex (run twice).
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")

    try:
        if latexmk:
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", str(tex_path.name)]
            print("Compiling with latexmk...")
            subprocess.run(cmd, check=True, cwd=out_dir)
            # Clean aux files but keep PDF
            subprocess.run(["latexmk", "-c"], check=True, cwd=out_dir)
        elif pdflatex:
            print("latexmk not found. Falling back to pdflatex (two passes)...")
            cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", str(tex_path.name)]
            subprocess.run(cmd, check=True, cwd=out_dir)
            subprocess.run(cmd, check=True, cwd=out_dir)
        else:
            print("Error: Neither latexmk nor pdflatex was found on PATH.")
            print("Install TeX Live (Linux), MacTeX (macOS), or MiKTeX (Windows) and ensure the binaries are on PATH.")
            sys.exit(1)
    except subprocess.CalledProcessError:
        # Show last few lines if compilation failed
        log = out_dir / "output.log"
        if log.exists():
            print("\nCompilation failed. Last 40 lines of output.log:\n")
            print("\n".join(log.read_text(errors="ignore").splitlines()[-40:]))
        else:
            print("\nCompilation failed. Check console output for errors.")
        # Clean up before exit
        for ext in (".aux", ".log", ".out", ".tex", ".pdf"):
            aux_file = out_dir / f"output{ext}"
            if aux_file.exists():
                aux_file.unlink()
        sys.exit(1)

    if temp_pdf_path.exists():
        shutil.move(str(temp_pdf_path), str(pdf_path))
        print(f"Success! PDF at: {pdf_path}")
    else:
        print("Compilation finished but output.pdf not found—check logs.")

    # Clean up LaTeX auxiliary files and .tex file
    for ext in (".aux", ".log", ".out", ".tex"):
        aux_file = out_dir / f"output{ext}"
        if aux_file.exists():
            aux_file.unlink()
    # If we copied a .tex file, also remove it
    if input_tex is not None and tex_path.exists():
        tex_path.unlink()

if __name__ == "__main__":
    main()
