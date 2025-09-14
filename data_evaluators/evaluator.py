"""
Main evaluation logic for comparing extracted text against ground truth.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import argparse

from data_evaluators.metrics import compute_metrics, normalize_text, align_texts
from config import local_params as lp

# Optional import for visualization
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from visualizers.visualizer import create_error_visualization
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def load_gold_data(gold_json_path: Path) -> Dict[str, Any]:
    """Load ground truth data from JSON file."""
    with open(gold_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def linearize_gold_text(gold_data: Dict[str, Any]) -> Tuple[str, List[Dict], List[str]]:
    """
    Convert gold JSON structure to linear text for comparison.
    
    Returns:
        - Full document text
        - List of element metadata with positions
        - List of page texts
    """
    all_text_parts = []
    page_texts = []
    elements = []
    offset = 0
    
    for p_idx, page in enumerate(gold_data.get("pages", [])):
        page_parts = []
        
        for e_idx, el in enumerate(page.get("elements", [])):
            et = el.get("type", "")
            
            if et == "heading":
                text = str(el.get("text", "")).strip()
                if not text:
                    continue
                start = offset + sum(len(x) for x in all_text_parts) + sum(len(x) for x in page_parts)
                page_parts.append(text + "\n\n")
                end = offset + sum(len(x) for x in all_text_parts) + sum(len(x) for x in page_parts)
                elements.append({
                    "page": p_idx + 1,
                    "idx": e_idx,
                    "type": "heading",
                    "text": text,
                    "start": start,
                    "end": end
                })
            
            elif et == "paragraph":
                text = str(el.get("text", "")).strip()
                if not text:
                    continue
                start = offset + sum(len(x) for x in all_text_parts) + sum(len(x) for x in page_parts)
                page_parts.append(text + "\n\n")
                end = offset + sum(len(x) for x in all_text_parts) + sum(len(x) for x in page_parts)
                elements.append({
                    "page": p_idx + 1,
                    "idx": e_idx,
                    "type": "paragraph",
                    "text": text,
                    "start": start,
                    "end": end
                })
            
            elif et == "table":
                # Handle table data if available
                rows = el.get("rows_data") or el.get("rows")
                if isinstance(rows, list) and rows:
                    lines = ["\t".join(map(str, r)) for r in rows]
                    text = "\n".join(lines)
                else:
                    text = ""
                
                if text:
                    start = offset + sum(len(x) for x in all_text_parts) + sum(len(x) for x in page_parts)
                    page_parts.append(text + "\n\n")
                    end = offset + sum(len(x) for x in all_text_parts) + sum(len(x) for x in page_parts)
                    elements.append({
                        "page": p_idx + 1,
                        "idx": e_idx,
                        "type": "table",
                        "text": text,
                        "start": start,
                        "end": end
                    })
        
        page_text = "".join(page_parts)
        page_texts.append(page_text)
        all_text_parts.append(page_text)
    
    ref_text = "".join(all_text_parts).rstrip() + "\n"
    return ref_text, elements, page_texts


def split_ocr_by_pages(ocr_text: str) -> List[str]:
    """Split OCR text by page markers if present."""
    import re
    
    # Look for page markers like "--- Page N ---"
    page_pattern = r"---\s*Page\s*\d+\s*---"
    
    if re.search(page_pattern, ocr_text):
        # Split by page markers
        parts = re.split(page_pattern, ocr_text)
        # Remove empty parts and strip
        return [p.strip() for p in parts if p.strip()]
    else:
        # No page markers, treat as single page
        return [ocr_text.strip()]


def evaluate_extraction(gold_json_path: Union[str, Path],
                       ocr_text_path: Union[str, Path],
                       output_dir: Optional[Union[str, Path]] = None,
                       lower: bool = False,
                       strip_punct: bool = False,
                       assume_page_markers: bool = True,
                       create_visualization: bool = False,
                       pdf_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Evaluate OCR extraction against ground truth.

    Args:
        gold_json_path: Path to ground truth JSON
        ocr_text_path: Path to extracted text file
        output_dir: Optional output directory for detailed results
        lower: Whether to lowercase for comparison
        strip_punct: Whether to strip punctuation for WER
        assume_page_markers: Whether to look for page markers in OCR text
        create_visualization: Whether to create error visualization on PDF
        pdf_path: Path to original PDF (required if create_visualization=True)

    Returns:
        Dictionary with evaluation results
    """
    gold_json_path = Path(gold_json_path)
    ocr_text_path = Path(ocr_text_path)
    
    # Load data
    gold_data = load_gold_data(gold_json_path)
    ocr_text = ocr_text_path.read_text(encoding='utf-8', errors='ignore')
    
    # Get reference text and structure
    ref_text, elements, ref_pages = linearize_gold_text(gold_data)
    
    # Document-level evaluation
    doc_scores = compute_metrics(ref_text, ocr_text, lower=lower, strip_punct=strip_punct)
    
    # Page-level evaluation if possible
    page_scores = []
    if assume_page_markers and len(ref_pages) > 1:
        ocr_pages = split_ocr_by_pages(ocr_text)
        
        for i, ref_page in enumerate(ref_pages):
            if i < len(ocr_pages):
                page_score = compute_metrics(ref_page, ocr_pages[i], lower=lower, strip_punct=strip_punct)
                page_score['page'] = i + 1
                page_scores.append(page_score)
    
    # Element-level evaluation if requested
    element_scores = []
    if elements and output_dir:
        # Create alignment for element mapping
        ops = align_texts(ref_text, ocr_text)
        
        for elem in elements:
            # Project element position to hypothesis
            ref_start = elem['start']
            ref_end = elem['end']
            elem_ref_text = ref_text[ref_start:ref_end]
            
            # Find corresponding text in hypothesis (simplified)
            elem_score = {
                'type': elem['type'],
                'page': elem['page'],
                'ref_text': elem_ref_text[:50] + '...' if len(elem_ref_text) > 50 else elem_ref_text
            }
            element_scores.append(elem_score)
    
    # Compile results
    results = {
        'gold_file': str(gold_json_path),
        'ocr_file': str(ocr_text_path),
        'doc': doc_scores,
        'pages': page_scores,
        'summary': {
            'doc_cer': doc_scores['cer'],
            'doc_wer': doc_scores['wer'],
            'n_pages': len(ref_pages),
            'n_elements': len(elements)
        }
    }
    
    # Save detailed results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed comparison if needed
        if page_scores:
            pages_df = pd.DataFrame(page_scores)
            pages_df.to_csv(output_dir / 'page_scores.csv', index=False)

    # Create error visualization if requested
    if create_visualization and VISUALIZATION_AVAILABLE:
        if pdf_path is None:
            print("[WARNING] PDF path required for visualization, skipping...")
        else:
            try:
                viz_output = output_dir / 'error_visualization.pdf' if output_dir else ocr_text_path.with_suffix('.errors.pdf')
                viz_result = create_error_visualization(
                    gold_json_path=gold_json_path,
                    ocr_text_path=ocr_text_path,
                    pdf_path=pdf_path,
                    output_path=viz_output
                )
                results['visualization'] = viz_result
                print(f"[INFO] Error visualization saved to: {viz_result['annotated_pdf']}")
            except Exception as e:
                print(f"[WARNING] Failed to create visualization: {e}")

    return results


def get_ocr_text_path(gold_json_path: Path) -> Path:
    """Get the corresponding OCR text file path for a gold JSON file."""
    # Convert gold JSON path to OCR text path
    # e.g., input_data/synthetic_data/gold_pdfs/doc_1.gold.json
    # -> output_data/synthetic_data/gold_pdfs/doc_1.txt

    input_data_path = Path("input_data")
    output_data_path = Path("output_data")

    # Remove .gold.json extension and add .txt
    stem = gold_json_path.stem.replace('.gold', '')
    txt_filename = f"{stem}.txt"

    try:
        # Get relative path from input_data
        relative_path = gold_json_path.parent.relative_to(input_data_path)
        ocr_path = output_data_path / relative_path / txt_filename
    except ValueError:
        # If not under input_data, use output_data root
        ocr_path = output_data_path / txt_filename

    return ocr_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR extraction against ground truth")
    parser.add_argument("--output-dir", type=str, help="Output directory for detailed results")
    parser.add_argument("--lower", action="store_true", help="Lowercase text for comparison")
    parser.add_argument("--strip-punct", action="store_true", help="Strip punctuation for WER calculation")
    parser.add_argument("--no-page-markers", action="store_true", help="Don't look for page markers in OCR text")
    parser.add_argument("--create-viz", action="store_true", help="Create error visualization on PDF")
    parser.add_argument("--pdf-path", type=str, help="Path to original PDF (required for visualization)")
    args = parser.parse_args()

    # Find all gold JSON files in the gold input data location
    gold_files = list(lp.gold_input_data_location.glob("*.gold.json"))

    if not gold_files:
        print(f"No gold JSON files found in {lp.gold_input_data_location}")
        exit(1)

    print(f"Found {len(gold_files)} gold JSON files to evaluate")

    all_results = []
    successful_evaluations = 0

    for gold_file in gold_files:
        print(f"\nEvaluating {gold_file.name}...")

        # Get corresponding OCR text file
        ocr_file = get_ocr_text_path(gold_file)

        if not ocr_file.exists():
            print(f"  Warning: Corresponding OCR file not found: {ocr_file}")
            continue

        try:
            # Get PDF path if visualization is requested
            pdf_path = None
            if args.create_viz:
                # Convert gold JSON path back to PDF path
                pdf_path = gold_file.with_suffix('').with_suffix('.pdf')
                if not pdf_path.exists():
                    print(f"  Warning: Corresponding PDF file not found: {pdf_path}")
                    pdf_path = None

            # Run evaluation
            result = evaluate_extraction(
                gold_json_path=gold_file,
                ocr_text_path=ocr_file,
                output_dir=args.output_dir,
                lower=args.lower,
                strip_punct=args.strip_punct,
                assume_page_markers=not args.no_page_markers,
                create_visualization=args.create_viz,
                pdf_path=pdf_path
            )

            # Print summary
            doc_cer = result['summary']['doc_cer']
            doc_wer = result['summary']['doc_wer']
            n_pages = result['summary']['n_pages']

            print(f"  Document CER: {doc_cer:.4f}")
            print(f"  Document WER: {doc_wer:.4f}")
            print(f"  Pages: {n_pages}")

            all_results.append(result)
            successful_evaluations += 1

        except Exception as e:
            print(f"  Error evaluating {gold_file.name}: {e}")

    # Print overall summary
    if successful_evaluations > 0:
        print(f"\n{'='*50}")
        print(f"SUMMARY: Successfully evaluated {successful_evaluations}/{len(gold_files)} files")

        total_cer = sum(r['summary']['doc_cer'] for r in all_results) / len(all_results)
        total_wer = sum(r['summary']['doc_wer'] for r in all_results) / len(all_results)
        total_pages = sum(r['summary']['n_pages'] for r in all_results)

        print(f"Average CER: {total_cer:.4f}")
        print(f"Average WER: {total_wer:.4f}")
        print(f"Total pages evaluated: {total_pages}")

        if args.output_dir:
            # Save overall summary
            summary_data = {
                'overall': {
                    'total_files': len(all_results),
                    'avg_cer': total_cer,
                    'avg_wer': total_wer,
                    'total_pages': total_pages
                },
                'individual_results': all_results
            }

            output_path = Path(args.output_dir) / 'overall_summary.json'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            print(f"\nDetailed results saved to {output_path}")
    else:
        print("No successful evaluations to summarize")
