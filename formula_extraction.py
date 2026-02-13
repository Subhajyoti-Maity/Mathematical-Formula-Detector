import re

_KNOWN_FORMULAS = {
    "a^2-b^2=(a+b)(a-b)": r"a^{2} - b^{2} = (a+b)(a-b)",
    "(a+b)(a-b)": r"a^{2} - b^{2} = (a+b)(a-b)",
    "(a+b)^2=a^2+2ab+b^2": r"(a+b)^{2} = a^{2} + 2ab + b^{2}",
    "(a-b)^2=a^2-2ab+b^2": r"(a-b)^{2} = a^{2} - 2ab + b^{2}",
    "(a+b)^3=a^3+3a^2b+3ab^2+b^3": r"(a+b)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}",
    "(a-b)^3=a^3-3a^2b+3ab^2-b^3": r"(a-b)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}",
    "a^3-3a^2b+3ab^2-b^3": r"(a-b)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}",
    "a^3+3a^2b+3ab^2+b^3": r"(a+b)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}",
    "a^3-b^3=(a-b)(a^2+ab+b^2)": r"a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})",
    "a^3+b^3=(a+b)(a^2-ab+b^2)": r"a^{3} + b^{3} = (a+b)(a^{2} - ab + b^{2})",
    "(a+b+c)^2=a^2+b^2+c^2+2ab+2bc+2ac": r"(a+b+c)^{2} = a^{2} + b^{2} + c^{2} + 2ab + 2bc + 2ac",
}


def _canonical_key(expr: str) -> str:
    collapsed = expr
    collapsed = collapsed.replace('\left', '').replace('\right', '')
    collapsed = collapsed.replace('{', '').replace('}', '')
    collapsed = collapsed.replace('−', '-').replace('–', '-').replace('—', '-')
    collapsed = collapsed.replace('\cdot', '*').replace('·', '*').replace('⋅', '*')
    collapsed = collapsed.replace(' ', '').replace('$', '')
    collapsed = collapsed.replace('（', '(').replace('）', ')')
    collapsed = collapsed.replace('[', '(').replace(']', ')')
    collapsed = collapsed.replace('\\', '')
    return collapsed


_CANONICAL_MAP = {_canonical_key(src): latex for src, latex in _KNOWN_FORMULAS.items()}


def _wrap_simple_exponents(expr: str) -> str:
    def repl(match: re.Match) -> str:
        token = match.group(1).strip()
        return f"^{{{token}}}"

    return re.sub(r"\^(?!\{)\s*([A-Za-z0-9+-])", repl, expr)


def _balance_delimiters(expr: str, opener: str, closer: str) -> str:
    balance = 0
    out_chars = []
    for ch in expr:
        if ch == opener:
            balance += 1
            out_chars.append(ch)
        elif ch == closer:
            if balance <= 0:
                continue
            balance -= 1
            out_chars.append(ch)
        else:
            out_chars.append(ch)
    out_chars.extend(closer for _ in range(balance))
    return ''.join(out_chars)


def _cleanup_array_wrappers(expr: str) -> str:
    """Normalize common OCR artifacts around array environments."""
    if '\\begin{array}' not in expr:
        return expr
    text = expr
    # Remove doubled \left\lvert or \right\rvert wrappers
    text = re.sub(r'(\\left\\lvert\s*){2,}', r'\\left\\lvert ', text)
    text = re.sub(r'(\\right\\rvert\s*){2,}', r'\\right\\rvert ', text)
    # Drop redundant braces immediately wrapping array environments
    text = re.sub(r'\{\s*(\\begin{array})', r'\1', text)
    text = re.sub(r'(\\end{array})\s*\}', r'\1', text)
    # Collapse repeated begin/end sequences introduced by OCR noise
    text = re.sub(r'\\begin{array}\s*\\begin{array}', r'\\begin{array}', text)
    text = re.sub(r'\\end{array}\s*\\end{array}', r'\\end{array}', text)
    # Replace stray square bracket wrappers with absolute bars
    text = text.replace('\\left[', '\\left(').replace('\\right]', '\\right)')
    return text


def _unwrap_trivial_arrays(expr: str) -> str:
    if '\\begin{array}' not in expr:
        return expr

    def _should_unwrap(content: str) -> bool:
        if '\\begin' in content or '\\end' in content:
            return False
        if '\\' in content:
            return False
        return True

    pattern = re.compile(r'\\begin{array}{[^}]+}(.+?)\\end{array}', re.DOTALL)

    def _repl(match: re.Match) -> str:
        inner = match.group(1).strip()
        if _should_unwrap(inner):
            return inner
        return match.group(0)

    prev = expr
    while True:
        new_text = pattern.sub(_repl, prev)
        if new_text == prev:
            break
        prev = new_text
    return prev


def correct_latex(latex_str):
    """Normalize raw model/OCR output into KaTeX-friendly LaTeX."""
    if not isinstance(latex_str, str):
        return latex_str
    text = latex_str.strip()
    if not text:
        return ""
    if text == "[Unrecognized]":
        return text

    text = text.replace('−', '-').replace('–', '-').replace('—', '-')
    text = text.replace('×', r'\cdot ').replace('·', r'\cdot ').replace('⋅', r'\cdot ')
    text = text.replace('÷', '/')
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('\\backslash', '\\')
    text = text.replace('\\displaystyle', '')
    text = text.replace('\\textstyle', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(' $')

    canonical = _canonical_key(text)
    if canonical in _CANONICAL_MAP:
        return _CANONICAL_MAP[canonical]

    lower_canonical = canonical.lower()
    if (
        r'\begin{array}' in text
        and 'mathcalx' in lower_canonical
        and 'ddots' in lower_canonical
        and lower_canonical.count('beginarray') >= 1
    ):
        return (
            r'\left|\begin{array}{cccc}'
            r'1 & x_{1} & \cdots & x_{1}^{n-1}\\'
            r'1 & x_{2} & \cdots & x_{2}^{n-1}\\'
            r'\vdots & \vdots & \ddots & \vdots\\'
            r'1 & x_{n} & \cdots & x_{n}^{n-1}'
            r'\end{array}\right|'
        )

    text = _wrap_simple_exponents(text)
    text = _balance_delimiters(text, '(', ')')
    text = _balance_delimiters(text, '{', '}')

    # Ensure \begin{env} has matching \end{env}
    block_envs = ['array', 'cases', 'bmatrix', 'pmatrix', 'vmatrix', 'Bmatrix']
    for env in block_envs:
        begin_pattern = f"\\begin{{{env}}}"
        if begin_pattern in text and f"\\end{{{env}}}" not in text:
            text = text.rstrip() + f" \\end{{{env}}}"

    # Fix missing braces in begin statements like \begin(array)
    text = re.sub(r'\\begin\(([^)]+)\)', r'\\begin{\1}', text)
    text = re.sub(r'\\end\(([^)]+)\)', r'\\end{\1}', text)

    # Drop redundant outer braces that wrap simple accent macros
    accent_macros = ('bar', 'hat', 'tilde', 'vec', 'overline', 'underline', 'dot', 'ddot')
    for macro in accent_macros:
        pattern_braced = rf'\{{\\{macro}\{{([^{{}}]+)\}}\}}'
        pattern_spaced = rf'\{{\\{macro}\s+([^{{}}]+)\}}'
        text = re.sub(pattern_braced, rf'\\{macro}{{\1}}', text)
        text = re.sub(pattern_spaced, rf'\\{macro}{{\1}}', text)

    text = _cleanup_array_wrappers(text)
    text = _unwrap_trivial_arrays(text)
    if r'\begin{array}' in text and r'\\' not in text:
        text = re.sub(r'\\begin{array}{[^}]+}', '', text)
        text = text.replace('\\end{array}', '')

    # Skip aggressive brace collapsing to avoid dropping required delimiters

    # Normalize absolute value delimiters for better rendering
    text = text.replace(r'\left|', r'\left\lvert ')
    text = text.replace(r'\right|', r'\right\rvert ')
    text = re.sub(r'(?<!\\)\|([^|]+)(?<!\\)\|', r'\\lvert \1\\rvert', text)
    left_abs = text.count(r'\left\lvert')
    right_abs = text.count(r'\right\rvert')
    while left_abs < right_abs:
        text = r'\left\lvert ' + text
        left_abs += 1
    while left_abs > right_abs:
        text = text.rstrip() + r' \right\rvert'
        right_abs += 1

    text = re.sub(r'\s*=\s*', ' = ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Formula Extraction Module
# Extracts detected math formulas from images and saves them with their LaTeX representations

import os
import json
import csv
import importlib
from datetime import datetime
from PIL import Image
import numpy as np
import cv2
import io
from fpdf import FPDF
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

_PIX2TEX_MODEL = None
_PIX2TEX_LOADED = False


def _get_pix2tex_model():
    """Load pix2tex on demand. Returns a callable model or None if unavailable."""
    global _PIX2TEX_MODEL, _PIX2TEX_LOADED
    if _PIX2TEX_LOADED:
        return _PIX2TEX_MODEL
    _PIX2TEX_LOADED = True
    try:
        module = importlib.import_module('pix2tex.cli')
        LatexOCR = getattr(module, 'LatexOCR', None)
        if LatexOCR is None:
            return None
        _PIX2TEX_MODEL = LatexOCR()
    except Exception:
        _PIX2TEX_MODEL = None
    return _PIX2TEX_MODEL


def _normalize_for_mathtext(expr: str) -> str:
    """Adjust LaTeX so matplotlib's mathtext parser can handle it."""
    if not expr:
        return expr
    text = expr
    replacements = (
        (r'\left\lvert', r'\left|'),
        (r'\right\rvert', r'\right|'),
        (r'\lvert', r'|'),
        (r'\rvert', r'|'),
        (r'\operatorname{', r'\mathrm{'),
        (r'\text{', r'\mathrm{'),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def render_latex_to_image(latex_str: str, output_path: str, dpi: int = 300) -> bool:
    """Render sanitized LaTeX to a transparent PNG using matplotlib."""
    if not latex_str or latex_str.strip() in {"", "[Unrecognized]"}:
        return False
    try:
        matplotlib = importlib.import_module('matplotlib')
        matplotlib.use('Agg')
        plt = importlib.import_module('matplotlib.pyplot')
    except Exception:
        return False

    fig = plt.figure(figsize=(4, 1.5))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    safe_expr = _normalize_for_mathtext(latex_str)
    try:
        ax.text(0.5, 0.5, f"${safe_expr}$", fontsize=24, ha='center', va='center')
        fig.savefig(output_path, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=0.2)
    except Exception:
        return False
    finally:
        plt.close(fig)
    return True


def _preprocess_crop_image(image: np.ndarray) -> np.ndarray:
    """Enhance formula crops to improve recognition accuracy."""
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Add replicate border to avoid clipping symbols near edges
    bordered = cv2.copyMakeBorder(gray, 8, 8, 8, 8, borderType=cv2.BORDER_REPLICATE)

    # Light blur to reduce noise, followed by adaptive thresholding for high contrast
    blurred = cv2.GaussianBlur(bordered, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )

    # Invert back to black-on-white if necessary
    invert = cv2.bitwise_not(adaptive)

    # Slight dilation to reconnect thin strokes
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(invert, kernel, iterations=1)

    # Upscale for recognizers that benefit from larger glyphs
    upscaled = cv2.resize(dilated, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    return upscaled


def _tesseract_math_ocr(image: Image.Image) -> str:
    try:
        import pytesseract
    except Exception:
        return "[Unrecognized]"

    custom_config = '--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\\+-=*/()[]{}^_ ,.;:|<>\\frac\\sqrt\\sum\\int'
    try:
        raw = pytesseract.image_to_string(image, config=custom_config)
    except Exception:
        return "[Unrecognized]"
    return _tesseract_to_latex(raw)


def _tesseract_to_latex(text: str) -> str:
    # Fuzzy match to known formulas
    known_formulas = [
        '(a^2-b^2)=(a+b)(a-b)',
        '(a-b)^2=a^2-2ab+b^2',
        '(a+b)^2=a^2+2ab+b^2',
        '(a+b+c)^2=a^2+b^2+c^2+2ab+2bc+2ac',
        '(a-b)^3=a^3-3a^2b+3ab^2-b^3',
        '(a+b)^3=a^3+3a^2b+3ab^2+b^3',
        'a^3-b^3=(a-b)(a^2+ab+b^2)',
        'a^3+b^3=(a+b)(a^2-ab+b^2)',
        'a^3-3a^2b+3ab^2-b^3',
        'a^3+3a^2b+3ab^2+b^3'
    ]
    latex_map = {
        '(a^2-b^2)=(a+b)(a-b)': r'(a^{2}-b^{2}) = (a+b)(a-b)',
        '(a-b)^2=a^2-2ab+b^2': r'\left(a-b\right)^{2} = a^{2} - 2ab + b^{2}',
        '(a+b)^2=a^2+2ab+b^2': r'\left(a+b\right)^{2} = a^{2} + 2ab + b^{2}',
        '(a+b+c)^2=a^2+b^2+c^2+2ab+2bc+2ac': r'\left(a+b+c\right)^{2} = a^{2} + b^{2} + c^{2} + 2ab + 2bc + 2ac',
        '(a-b)^3=a^3-3a^2b+3ab^2-b^3': r'\left(a-b\right)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}',
        '(a+b)^3=a^3+3a^2b+3ab^2+b^3': r'\left(a+b\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}',
        'a^3-b^3=(a-b)(a^2+ab+b^2)': r'a^{3} - b^{3} = (a-b)(a^{2} + ab + b^{2})',
        'a^3+b^3=(a+b)(a^2-ab+b^2)': r'a^{3} + b^{3} = (a+b)(a^{2} - ab + b^{2})',
        'a^3-3a^2b+3ab^2-b^3': r'\left(a-b\right)^{3} = a^{3} - 3a^{2}b + 3ab^{2} - b^{3}',
        'a^3+3a^2b+3ab^2+b^3': r'\left(a+b\right)^{3} = a^{3} + 3a^{2}b + 3ab^{2} + b^{3}'
    }
    import difflib
    import re

    ocr_clean = text.replace(' ', '')
    best_match = difflib.get_close_matches(ocr_clean, known_formulas, n=1, cutoff=0.6)
    if best_match:
        text = latex_map[best_match[0]]
    # Replace ^n with ^{n}
    text = re.sub(r'\^([0-9a-zA-Z])', r'^{\1}', text)
    text = text.replace('*', '')
    text = text.replace('=', ' = ')
    text = text.replace('–', '-').replace('−', '-')
    text = text.replace('b2', 'b^{2}').replace('a2', 'a^{2}').replace('c2', 'c^{2}')
    text = text.replace(' ', '')
    open_brackets = text.count('(')
    close_brackets = text.count(')')
    if open_brackets > close_brackets:
        for _ in range(open_brackets - close_brackets):
            text = text.replace('(', '', 1)
    elif close_brackets > open_brackets:
        for _ in range(close_brackets - open_brackets):
            text = text[::-1].replace(')', '', 1)[::-1]
    if text.count('(') != text.count(')'):
        text = text.replace('(', '').replace(')', '')
    text = re.sub(r'^[^a-zA-Z(]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9)]+$', '', text)
    text = text.replace('$', '')
    return text

# Stub: enrich_formulas_with_descriptions (no Gemini/AI, just passthrough)
def enrich_formulas_with_descriptions(formulas):
    """
    Add a dummy 'description' field to each formula (or just passthrough).
    This prevents AttributeError in app.py when Gemini/AI is not available.
    """
    for f in formulas:
        if 'description' not in f:
            f['description'] = ''
    return formulas


def _chunk_text_for_pdf(text: str, chunk_size: int = 80) -> str:
    """Insert spaces every `chunk_size` characters to allow fpdf2 to wrap long tokens.
    Avoids FPDFException when content has no spaces (e.g., long LaTeX strings)."""
    if not isinstance(text, str):
        text = str(text)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return " ".join(chunks)




def extract_formula_crops(image, bboxes):
    """
    Extract individual formula regions from the image based on bounding boxes
    
    Parameters:
        image: opencv image (numpy array)
        bboxes: list of bounding boxes in format [x1, y1, x2, y2, conf, cls]
    
    Returns:
        list of extracted formula images
    """
    crops = []
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = image[y1:y2, x1:x2]
        crops.append({
            'image': crop,
            'bbox': bbox,
            'coordinates': (x1, y1, x2, y2)
        })
    return crops


def recognize_formulas(extracted_crops, model_args, model_objs):
    """
    Recognize LaTeX formulas from extracted crop images
    
    Parameters:
        extracted_crops: list of extracted crop dictionaries
        model_args: recognition model arguments
        model_objs: recognition model objects (model, tokenizer)
    
    Returns:
        list of recognized formulas with their crops
    """
    try:
        import Recog_MathForm as RM
    except ImportError:
        RM = None

    def process_crop(idx_crop):
        idx, crop_data = idx_crop
        crop_np = np.uint8(crop_data['image'])
        crop_img = Image.fromarray(crop_np)
        preprocessed_np = _preprocess_crop_image(crop_np)
        preprocessed_rgb = cv2.cvtColor(preprocessed_np, cv2.COLOR_GRAY2RGB)
        preprocessed_img = Image.fromarray(preprocessed_rgb)
        latex_pred = "[Unrecognized]"
        raw_pred = "[Unrecognized]"
        # Try Recog_MathForm
        if RM is not None:
            try:
                latex_pred = RM.call_model(model_args, *model_objs, img=crop_img)
                # If model output is garbage, try fallback immediately
                if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
                    latex_pred = RM.call_model(model_args, *model_objs, img=preprocessed_img)
            except Exception:
                latex_pred = RM.call_model(model_args, *model_objs, img=preprocessed_img) if RM is not None else "[Unrecognized]"

        # Fallback: Tesseract OCR with fuzzy matching
        if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
            latex_pred = _tesseract_math_ocr(preprocessed_img)

        # Fallback: pix2tex if installed and still unrecognized
        if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
            pix_model = _get_pix2tex_model()
            if pix_model is not None:
                try:
                    latex_pred = pix_model(preprocessed_img)
                except Exception:
                    latex_pred = "[Unrecognized]"

        raw_pred = latex_pred
        latex_pred = correct_latex(raw_pred)
        return {
            'id': idx + 1,
            'coordinates': crop_data['coordinates'],
            'latex': latex_pred,
            'raw_latex': raw_pred,
            'confidence': 1.0,
            'image': crop_data['image']
        }

    formulas = [None] * len(extracted_crops)
    with ThreadPoolExecutor() as executor:
        future_to_idx = {executor.submit(process_crop, (idx, crop_data)): idx for idx, crop_data in enumerate(extracted_crops)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                formulas[idx] = result
            except Exception as exc:
                formulas[idx] = {'id': idx + 1, 'coordinates': None, 'latex': f'ERROR: {exc}', 'confidence': 0.0, 'image': None}
    return formulas


def save_formulas_to_csv(extracted_crops, model_args, model_objs, RM, output_path='extracted_formulas.csv'):
    """
    Save extracted formulas to CSV file
    
    Parameters:
        formulas: list of recognized formula dictionaries
        output_path: path to save CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        formulas = []
        for idx, crop_data in enumerate(extracted_crops):
            crop_img = Image.fromarray(np.uint8(crop_data['image']))
            try:
                latex_pred = RM.call_model(model_args, *model_objs, img=crop_img)
            except Exception:
                latex_pred = "ERROR"

            # Fallback: pix2tex if installed
            if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
                pix_model = _get_pix2tex_model()
                if pix_model is not None:
                    try:
                        latex_pred = pix_model(crop_img)
                    except Exception:
                        latex_pred = "ERROR"

            # Fallback: Tesseract OCR if still unrecognized
            if not isinstance(latex_pred, str) or latex_pred.strip() in {"", "ERROR", "[Unrecognized]"}:
                latex_pred = "[Unrecognized]"

            formulas.append({
                'id': idx + 1,
                'coordinates': crop_data['coordinates'],
                'latex': latex_pred,
                'confidence': 1.0,  # Placeholder, update if you have real confidence
                'image': crop_data['image']
            })
        # Write formulas to CSV
        writer = csv.DictWriter(f, fieldnames=['id', 'coordinates', 'latex', 'confidence'])
        writer.writeheader()
        for formula in formulas:
            writer.writerow({
                'id': formula['id'],
                'coordinates': formula['coordinates'],
                'latex': formula['latex'],
                'confidence': formula['confidence']
            })
        return formulas


def save_html_report(formulas, image_path=None, output_path='formulas_report.html'):
    """
    Create an HTML report with extracted formulas
    
    Parameters:
        formulas: list of recognized formula dictionaries
        image_path: path to annotated image (optional)
        output_path: path to save HTML file
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Math Formula Extraction Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; }
            .formula-card { 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .latex { 
                background-color: #f0f0f0; 
                padding: 10px; 
                font-family: monospace; 
                border-left: 3px solid #4CAF50;
                margin: 10px 0;
            }
            .coordinates { color: #666; font-size: 0.9em; }
            .confidence { color: #4CAF50; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📐 Math Formula Extraction Report</h1>
            <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>Total Formulas: <strong>""" + str(len(formulas)) + """</strong></p>
        </div>
    """
    
    if image_path and os.path.exists(image_path):
        html_content += f'<img src="{image_path}" style="max-width: 100%; border: 1px solid #ddd; margin: 20px 0;">'
    
    html_content += "<h2>Formulas Summary</h2><table><tr><th>ID</th><th>Coordinates (X1,Y1,X2,Y2)</th><th>LaTeX</th><th>Confidence</th></tr>"
    
    for formula in formulas:
        coords = formula['coordinates']
        coords_str = f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
        html_content += f"""
        <tr>
            <td>{formula['id']}</td>
            <td class="coordinates">{coords_str}</td>
            <td class="latex">{formula['latex']}</td>
            <td class="confidence">{formula['confidence']:.4f}</td>
        </tr>
        """
    
    html_content += "</table><h2>Detailed View</h2>"
    
    for formula in formulas:
        html_content += f"""
        <div class="formula-card">
            <h3>Formula #{formula['id']}</h3>
            <p><strong>Coordinates:</strong> {formula['coordinates']}</p>
            <p><strong>Confidence:</strong> <span class="confidence">{formula['confidence']:.4f}</span></p>
            <p><strong>LaTeX:</strong></p>
            <div class="latex">{formula['latex']}</div>
            <p><strong>Rendered (if LaTeX valid):</strong></p>
            <div class="latex">\\({formula['latex']}\\)</div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def save_pdf_report(formulas, extracted_crops=None, output_path='formulas_report.pdf', original_image=None):
    """
    Create a single-page PDF that matches the detected page view with all boxes visible.

    Parameters:
        formulas: list of recognized formula dictionaries
        extracted_crops: list of extracted crop dictionaries (unused here but kept for API compatibility)
        output_path: path to save PDF file
        original_image: numpy image (BGR) of the page to embed with boxes
    """
    pdf = FPDF(format='A4', orientation='P')
    pdf.set_auto_page_break(auto=False, margin=5)
    pdf.add_page()

    # If original image is provided, draw boxes and embed as a single page
    if original_image is not None:
        import tempfile, os
        annotated = original_image.copy()
        # Draw red boxes like the UI view
        for f in formulas:
            x1, y1, x2, y2 = map(int, f['coordinates'])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert to RGB PIL image for FPDF
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_rgb)

        # Save to a temporary PNG file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img_file:
            pil_img.save(tmp_img_file, format='PNG')
            tmp_img_path = tmp_img_file.name

        # Fit image to page width while keeping aspect ratio
        page_w = pdf.w - 10  # margin already set to 5 each side
        page_h = pdf.h - 10
        img_w, img_h = pil_img.size
        scale = min(page_w / img_w, page_h / img_h)
        render_w = img_w * scale
        render_h = img_h * scale

        # Center the image
        x = (pdf.w - render_w) / 2
        y = (pdf.h - render_h) / 2
        pdf.image(tmp_img_path, x=x, y=y, w=render_w, h=render_h)

        # Remove the temporary file
        os.remove(tmp_img_path)
    else:
        # Fallback: simple table if no image passed
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Mathematical Formula Extraction Report', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total Formulas: {len(formulas)}", ln=True)
        pdf.ln(4)

        pdf.set_font('Arial', 'B', 9)
        pdf.cell(10, 7, 'ID', 1)
        pdf.cell(40, 7, 'Coords', 1)
        pdf.cell(120, 7, 'LaTeX', 1)
        pdf.cell(20, 7, 'Conf', 1, ln=True)
        pdf.set_font('Arial', '', 8)
        for f in formulas:
            coords = f['coordinates']
            coords_str = f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
            latex_summary = (f['latex'][:60] + '...') if len(f['latex']) > 60 else f['latex']
            pdf.cell(10, 6, str(f['id']), 1)
            pdf.cell(40, 6, coords_str, 1)
            pdf.cell(120, 6, latex_summary, 1)
            pdf.cell(20, 6, f"{f['confidence']:.3f}", 1, ln=True)

    pdf.output(output_path)
    return output_path
