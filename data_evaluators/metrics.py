"""
Evaluation metrics for text extraction quality.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher


# Common ligatures to normalize
LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", 
    "\ufb03": "ffi", "\ufb04": "ffl",
}


def normalize_text(s: str, lower: bool = False) -> str:
    """
    Normalize text for comparison.
    
    Args:
        s: Input text
        lower: Whether to lowercase
        
    Returns:
        Normalized text
    """
    if not s: 
        return ""
    
    # Remove soft hyphens
    s = s.replace("\u00ad", "")
    
    # Replace ligatures
    for k, v in LIGATURES.items():
        s = s.replace(k, v)
    
    # Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    
    # Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    
    return s.lower() if lower else s


def strip_punctuation_from_words(words: List[str]) -> List[str]:
    """Remove punctuation from words for WER calculation."""
    return [re.sub(r"[^\w\-']", "", w) for w in words if re.sub(r"[^\w\-']", "", w) != ""]


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute character-level Levenshtein distance.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        Edit distance
    """
    if a == b: 
        return 0
    if not a: 
        return len(b)
    if not b: 
        return len(a)
    
    if len(a) > len(b): 
        a, b = b, a
    
    prev = list(range(len(a) + 1))
    for bj in b:
        curr = [prev[0] + 1]
        for i, ai in enumerate(a, start=1):
            ins = curr[i-1] + 1
            dele = prev[i] + 1
            sub = prev[i-1] + (ai != bj)
            curr.append(min(ins, dele, sub))
        prev = curr
    
    return prev[-1]


def levenshtein_words(a_words: List[str], b_words: List[str]) -> int:
    """
    Compute word-level Levenshtein distance.
    
    Args:
        a_words: First word list
        b_words: Second word list
        
    Returns:
        Edit distance
    """
    na, nb = len(a_words), len(b_words)
    if na == 0: 
        return nb
    if nb == 0: 
        return na
    
    if na > nb: 
        a_words, b_words = b_words, a_words
        na, nb = nb, na
    
    prev = list(range(na + 1))
    for bj in b_words:
        curr = [prev[0] + 1]
        for i, ai in enumerate(a_words, start=1):
            ins = curr[i-1] + 1
            dele = prev[i] + 1
            sub = prev[i-1] + (ai != bj)
            curr.append(min(ins, dele, sub))
        prev = curr
    
    return prev[-1]


def compute_cer(reference: str, hypothesis: str, normalize: bool = True, lower: bool = False) -> float:
    """
    Compute Character Error Rate (CER).
    
    Args:
        reference: Ground truth text
        hypothesis: OCR/extracted text
        normalize: Whether to normalize text
        lower: Whether to lowercase
        
    Returns:
        CER as float (0.0 = perfect, 1.0 = completely wrong)
    """
    if normalize:
        reference = normalize_text(reference, lower=lower)
        hypothesis = normalize_text(hypothesis, lower=lower)
    
    if not reference:
        return 0.0 if not hypothesis else 1.0
    
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def compute_wer(reference: str, hypothesis: str, 
                normalize: bool = True, lower: bool = False, 
                strip_punct: bool = False) -> float:
    """
    Compute Word Error Rate (WER).
    
    Args:
        reference: Ground truth text
        hypothesis: OCR/extracted text
        normalize: Whether to normalize text
        lower: Whether to lowercase
        strip_punct: Whether to strip punctuation
        
    Returns:
        WER as float (0.0 = perfect, 1.0 = completely wrong)
    """
    if normalize:
        reference = normalize_text(reference, lower=lower)
        hypothesis = normalize_text(hypothesis, lower=lower)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if strip_punct:
        ref_words = strip_punctuation_from_words(ref_words)
        hyp_words = strip_punctuation_from_words(hyp_words)
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    
    distance = levenshtein_words(ref_words, hyp_words)
    return distance / len(ref_words)


def compute_metrics(reference: str, hypothesis: str, 
                   lower: bool = False, strip_punct: bool = False) -> Dict[str, float]:
    """
    Compute both CER and WER metrics.
    
    Args:
        reference: Ground truth text
        hypothesis: OCR/extracted text
        lower: Whether to lowercase
        strip_punct: Whether to strip punctuation for WER
        
    Returns:
        Dictionary with 'cer' and 'wer' values
    """
    ref_norm = normalize_text(reference, lower=lower)
    hyp_norm = normalize_text(hypothesis, lower=lower)
    
    cer = compute_cer(ref_norm, hyp_norm, normalize=False)
    wer = compute_wer(ref_norm, hyp_norm, normalize=False, strip_punct=strip_punct)
    
    return {
        "cer": cer,
        "wer": wer,
        "ref_chars": len(ref_norm),
        "ref_words": len(ref_norm.split())
    }


def align_texts(reference: str, hypothesis: str) -> List[Tuple[int, int, int, int, str]]:
    """
    Create alignment between reference and hypothesis texts.
    
    Returns list of operations (ref_start, ref_end, hyp_start, hyp_end, operation)
    where operation is one of: 'equal', 'replace', 'delete', 'insert'
    """
    sm = SequenceMatcher(None, reference, hypothesis, autojunk=False)
    ops = sm.get_opcodes()
    return [(i1, i2, j1, j2, tag) for tag, i1, i2, j1, j2 in ops]
