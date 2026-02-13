
import streamlit as st
import cv2 
import numpy as np
import Inference_Math_Detection as MD
import Recog_MathForm as RM
import formula_extraction as FE
from PIL import Image
import pdf2image
import os
import zipfile
import io
import re
import json
import csv
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_text_boxes(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    text_boxes = []
    for i in range(len(data['level'])):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        text = data['text'][i]
        text_boxes.append(((x, y, x+w, y+h), text))
    return text_boxes

def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    # Calculate intersection
    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    # Calculate union
    box1Area = (x1_max - x1_min) * (y1_max - y1_min)
    box2Area = (x2_max - x2_min) * (y2_max - y2_min)
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def filter_formula_boxes(formula_boxes, text_boxes):
    # Remove only boxes that confidently match textual headings; keep inline formulas intact.
    filtered = []
    min_area = 800  # Minimum area to consider as formula (tune as needed)
    img = st.session_state.get('opencv_image')
    img_height = int(img.shape[0]) if isinstance(img, np.ndarray) else 0
    top_margin = int(img_height * 0.20) if img_height else 0
    heading_boxes = []

    def looks_like_heading(text: str, y_top: int) -> bool:
        if not text:
            return False
        stripped = text.strip()
        if len(stripped) < 4:
            return False
        # Reject anything that already contains obvious math symbols.
        if any(ch in "=+−-*/^_|()[]{}\\" for ch in stripped):
            return False
        alpha = sum(c.isalpha() for c in stripped)
        if alpha / max(len(stripped), 1) < 0.8:
            return False
        words = stripped.split()
        # Long headings or known keywords are safe to treat as non-formula text.
        keywords = {"chapter", "section", "exercise", "example", "formula", "formulas"}
        keyword_hit = any(word.lower() in keywords for word in words)
        if y_top <= top_margin or keyword_hit:
            return True
        # Single long uppercase word is very likely a title.
        if len(words) == 1 and words[0].isupper() and len(words[0]) >= 4:
            return True
        return False

    for fbox in formula_boxes:
        x1, y1, x2, y2 = fbox[:4]
        area = (x2 - x1) * (y2 - y1)
        is_heading = False
        for tbox, text in text_boxes:
            if not text:
                continue
            if is_overlapping(fbox[:4], tbox) > 0.4 and looks_like_heading(text, tbox[1]):
                is_heading = True
                break
        if is_heading and area > min_area:
            heading_boxes.append(fbox)
        elif not is_heading and area > min_area:
            filtered.append(fbox)
    # Store headings for underline drawing, but do NOT count or box them as formulas
    st.session_state['heading_boxes'] = heading_boxes
    return filtered

def download_models():
    mathdetector = 'Models/MathDetector.ts'
    mathrecog = 'Models/MathRecog.pth'
    
    if not os.path.exists(mathdetector):
        detector_url = 'gdown -O '+mathdetector+' https://drive.google.com/uc?id=1AGZTIRbx-KmLQ7bSEAcxUWWtdSrYucFz'
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(detector_url)
    else:
        print("Detector Model is here")

    if not os.path.exists(mathrecog):
        detector_url = 'gdown -O '+mathrecog+' https://drive.google.com/uc?id=1oR7eNBOC_3TBhFQ1KTzuWSl7-fet4cYh'
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(detector_url)
    else:
        print("Reconizer Model is here")

def draw_rectangles (image, preds):
    # Draw only formula boxes (not headings)
    for each_pred in preds:
        cv2.rectangle(image, (int(each_pred[0]),int(each_pred[1])), (int(each_pred[2]),int(each_pred[3])),(255,0,0),2)
    # Optionally, draw underline for headings (not counted as formulas)
    headings = st.session_state.get('heading_boxes', [])
    for hbox in headings:
        x1, y1, x2, y2 = map(int, hbox[:4])
        underline_y = y2 + 3
        cv2.line(image, (x1, underline_y), (x2, underline_y), (0,0,255), 3)


def _normalize_latex_for_katex(s: str) -> str:
    r"""Normalize common non-standard macros to KaTeX-safe equivalents.

    Examples handled:
    - \cal X -> \mathcal{X}
    - \bf X  -> \mathbf{X}; \bf\nabla -> \boldsymbol{\nabla}
    - \it X  -> \mathit{X}; \rm X -> \mathrm{X}
    - \simLambda -> \tilde{\Lambda} (heuristic)
    - \stackrel{a}{b} -> \overset{a}{b} (more robust in KaTeX)
    """
    if not s:
        return s
    t = s
    # \cal -> \mathcal{}
    t = re.sub(r"\\cal\s*([A-Za-z])", r"\\mathcal{\1}", t)
    # \bf token forms
    t = re.sub(r"\\bf\s*([A-Za-z])", r"\\mathbf{\1}", t)
    t = re.sub(r"\\bf\s*\{([^}]*)\}", r"\\mathbf{\1}", t)
    t = t.replace("\\bf\\nabla", "\\boldsymbol{\\nabla}")
    # \it, \rm
    t = re.sub(r"\\it\s*([A-Za-z])", r"\\mathit{\1}", t)
    t = re.sub(r"\\rm\s*([A-Za-z])", r"\\mathrm{\1}", t)
    # \simX -> \tilde{X} (heuristic for recognized tokens like \simLambda)
    t = re.sub(r"\\sim([A-Za-z])", r"\\tilde{\\\1}", t)
    # stackrel -> overset (KaTeX supports both; overset is often safer)
    t = t.replace("\\stackrel", "\\overset")
    # Absolute value variants not supported uniformly across renderers
    t = t.replace("\\left\\lvert", "\\left|")
    t = t.replace("\\right\\rvert", "\\right|")
    t = t.replace("\\lvert", "|")
    t = t.replace("\\rvert", "|")
    # Minor whitespace cleanup
    t = re.sub(r"\s+", " ", t).strip()
    return t

def render_latex_block(latex_text):
    """Render LaTeX with safe fallback to keep layout aligned."""
    if latex_text is None or str(latex_text).strip() == "":
        st.info("No LaTeX available for this formula.")
        return
    sanitized = FE.correct_latex(str(latex_text))
    if not sanitized or sanitized == "[Unrecognized]":
        st.info("Formula could not be recognized from the image.")
        raw_display = str(latex_text).strip() or "[Unrecognized]"
        st.code(raw_display, language='text')
        return
    normalized = _normalize_latex_for_katex(sanitized)
    try:
        # st.latex centers the formula and avoids overflowing raw text blocks
        st.latex(normalized)
    except Exception:
        st.warning("Could not render LaTeX; showing raw text instead.")
        st.code(sanitized, language='latex')

if __name__ == '__main__':
    st.set_page_config(page_title="Math Formula Detection", page_icon="➗", layout="wide")
    download_models()

    st.markdown(
        """
        <style>
        /* Modern look: bold type, soft card edges, subtle glow */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
        html, body, [class*="css"]  { font-family: 'Space Grotesk', sans-serif; }
        .main { background: radial-gradient(circle at 20% 20%, rgba(0,181,173,0.12), transparent 30%),
                          radial-gradient(circle at 80% 0%, rgba(255,140,66,0.14), transparent 32%),
                          #0f1116;
                 color: #e6e8ef; }
        .stSidebar { background: #0b0d12; }
        .stSidebar, .st-bb, .st-at, .st-bc { color: #e6e8ef; }
        .css-1d391kg, .css-1lcbmhc { color: #e6e8ef; }
        .stButton>button { border-radius: 12px; border: 1px solid #1dd3b0; color: #0b0d12;
                           background: linear-gradient(135deg, #1dd3b0 0%, #17a2f3 100%);
                           box-shadow: 0 10px 30px rgba(23,162,243,0.25); font-weight: 600; }
        .stButton>button:hover { box-shadow: 0 12px 34px rgba(29,211,176,0.35); transform: translateY(-1px); }
        .stDownloadButton>button { border-radius: 12px; background: #151a23; border: 1px solid #283344;
                                   color: #e6e8ef; box-shadow: 0 6px 20px rgba(0,0,0,0.25); }
        .stDownloadButton>button:hover { border-color: #1dd3b0; color: #1dd3b0; }
        .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
        .metric-card { background: #151a23; border: 1px solid #1f2a39; border-radius: 14px;
                       padding: 1rem 1.2rem; box-shadow: 0 12px 45px rgba(0,0,0,0.35); }
        .stExpander { background: #151a23; border: 1px solid #1f2a39; border-radius: 12px; }
        .stExpander > div > div { padding: 0.75rem 1rem; }
        .stAlert { border-radius: 12px; border: 1px solid #1dd3b0; background: rgba(29,211,176,0.08); }
        .stMarkdown h1, h2, h3 { color: #e6e8ef; }
        .latex { color: #e6e8ef; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize session state
    if 'extraction_done' not in st.session_state:
        st.session_state.extraction_done = False
    if 'extracted_formulas' not in st.session_state:
        st.session_state.extracted_formulas = None
    if 'extracted_crops' not in st.session_state:
        st.session_state.extracted_crops = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'detection_done' not in st.session_state:
        st.session_state.detection_done = False
    if 'results_boxes' not in st.session_state:
        st.session_state.results_boxes = None
    if 'opencv_image' not in st.session_state:
        st.session_state.opencv_image = None
    if 'pdf_pages' not in st.session_state:
        st.session_state.pdf_pages = None
    if 'pdf_file_name' not in st.session_state:
        st.session_state.pdf_file_name = None
    if 'pdf_active_page' not in st.session_state:
        st.session_state.pdf_active_page = None
    
    math_model = MD.initialize_model("Models/MathDetector.ts")
    mathargs, *mathobjs = RM.initialize()

    st.markdown("""
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:linear-gradient(135deg,#1dd3b0,#17a2f3); width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#0b0d12;">∑</div>
            <div>
                <div style="font-size:28px; font-weight:700; color:#e6e8ef;">Mathematical Formula Detector</div>
                <div style="color:#94a3b8;">Detect • Extract • Render LaTeX from images and PDFs</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    inf_style = st.sidebar.selectbox("Inference Type",('Image', 'PDF'))
    if inf_style == 'Image':

        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png','jpeg', 'jpg'])

    #     res = st.sidebar.radio("Final Result",("Detection","Detection And Recogntion"))
        if uploaded_file is not None:
            if st.sidebar.button('Clear uploaded file or image!'):
                st.warning("attempt to clear uploaded_file")
                uploaded_file.seek(0)
            with st.spinner(text='In progress'):
                # Read once and render from decoded array to avoid stale file refs
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                preview_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                st.sidebar.image(preview_image)
                
                # Store image in session for later use
                st.session_state.opencv_image = opencv_image

                if st.button('Launch the Detection!'):
                    results_boxes = MD.predict_formulas(opencv_image, math_model)
                    # Get text boxes using Tesseract OCR
                    text_boxes = get_text_boxes(opencv_image)
                    # Filter out formula boxes that overlap with text
                    results_boxes = filter_formula_boxes(results_boxes, text_boxes)
                    st.session_state.results_boxes = results_boxes
                    st.session_state.detection_done = True
                
                # Show detection result if detection was done
                if st.session_state.detection_done and st.session_state.results_boxes is not None:
                    results_boxes = st.session_state.results_boxes
                    images_rectangles = st.session_state.opencv_image.copy()
                    draw_rectangles(images_rectangles, results_boxes)
                    st.image(images_rectangles)
                    
                    # Add extraction option
                    if len(results_boxes) > 0:
                        st.success(f"✓ Found {len(results_boxes)} formulas!")
                        
                        # Create two columns for better layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("🚀 Extract Formulas to File"):
                                with st.spinner("Extracting and recognizing formulas..."):
                                    # Create output directory with timestamp
                                    from datetime import datetime
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_dir = f"extracted_output_{timestamp}"
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    # Only extract and recognize if not already done for current detection
                                    if not hasattr(st.session_state, 'extracted_formulas') or st.session_state.extracted_formulas is None:
                                        # Extract formula crops
                                        st.session_state.extracted_crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)
                                        
                                        # Recognize formulas
                                        st.session_state.extracted_formulas = FE.recognize_formulas(st.session_state.extracted_crops, mathargs, mathobjs)
                                        # Skip slow Gemini refinement for faster response
                                        # Enrich with AI descriptions if available
                                        st.session_state.extracted_formulas = FE.enrich_formulas_with_descriptions(st.session_state.extracted_formulas)
                                    
                                    # Save all files to output directory
                                    formulas = st.session_state.extracted_formulas
                                    extracted_crops = st.session_state.extracted_crops
                                    
                                    # Draw rectangles on a copy of the image (as in UI)
                                    annotated_img = st.session_state.opencv_image.copy()
                                    draw_rectangles(annotated_img, st.session_state.results_boxes)
                                    # Save Annotated Image (PNG)
                                    annotated_path = os.path.join(output_dir, 'annotated_image.png')
                                    cv2.imwrite(annotated_path, annotated_img)

                                    # Save consolidated PDF Report using the annotated image
                                    pdf_path = os.path.join(output_dir, 'formulas_report.pdf')
                                    FE.save_pdf_report(formulas, extracted_crops=extracted_crops, output_path=pdf_path, original_image=annotated_img)
                                    
                                    # Save individual formula images
                                    formula_dir = os.path.join(output_dir, 'formula_images')
                                    rendered_dir = os.path.join(output_dir, 'rendered_formulas')
                                    os.makedirs(formula_dir, exist_ok=True)
                                    os.makedirs(rendered_dir, exist_ok=True)
                                    metadata_rows = []
                                    render_failures = 0
                                    for crop_data, formula in zip(extracted_crops, formulas):
                                        filename = f'formula_{formula["id"]:04d}.png'
                                        img_path = os.path.join(formula_dir, filename)
                                        cv2.imwrite(img_path, crop_data['image'])
                                        rendered_name = f'formula_{formula["id"]:04d}_rendered.png'
                                        rendered_path = os.path.join(rendered_dir, rendered_name)
                                        rendered_rel = os.path.join('rendered_formulas', rendered_name)
                                        rendered_ok = FE.render_latex_to_image(formula['latex'], rendered_path)
                                        if not rendered_ok:
                                            rendered_rel = ''
                                            render_failures += 1
                                        image_rel = os.path.join('formula_images', filename).replace('\\', '/')
                                        rendered_rel = rendered_rel.replace('\\', '/') if rendered_rel else ''
                                        metadata_rows.append({
                                            'id': formula['id'],
                                            'image': image_rel,
                                            'latex': formula['latex'],
                                            'raw_latex': formula.get('raw_latex', ''),
                                            'rendered_image': rendered_rel,
                                            'render_success': rendered_ok,
                                            'coordinates': formula['coordinates'],
                                            'confidence': formula['confidence']
                                        })

                                    # Save metadata in JSON and CSV for easy pairing with crops
                                    metadata_json_path = os.path.join(output_dir, 'formulas_metadata.json')
                                    with open(metadata_json_path, 'w', encoding='utf-8') as meta_json:
                                        json.dump(metadata_rows, meta_json, ensure_ascii=False, indent=2)

                                    metadata_csv_path = os.path.join(output_dir, 'formulas_metadata.csv')
                                    fieldnames = ['id', 'image', 'rendered_image', 'render_success', 'latex', 'raw_latex', 'coordinates', 'confidence']
                                    with open(metadata_csv_path, 'w', newline='', encoding='utf-8') as meta_csv:
                                        writer = csv.DictWriter(meta_csv, fieldnames=fieldnames)
                                        writer.writeheader()
                                        for row in metadata_rows:
                                            serializable = row.copy()
                                            serializable['coordinates'] = ','.join(str(coord) for coord in row['coordinates'])
                                            writer.writerow(serializable)

                                    if render_failures and render_failures == len(metadata_rows):
                                        st.warning("Could not render LaTeX to images; ensure matplotlib is installed for PNG exports.")
                                    
                                    # Create ZIP package
                                    zip_path = os.path.join(output_dir, 'extracted_formulas.zip')
                                    with zipfile.ZipFile(zip_path, 'w') as zip_file:
                                        for root, dirs, files in os.walk(output_dir):
                                            for file in files:
                                                if not file.endswith('.zip'):
                                                    file_path = os.path.join(root, file)
                                                    arcname = os.path.relpath(file_path, output_dir)
                                                    zip_file.write(file_path, arcname)
                                    
                                    st.session_state.extraction_done = True
                                    st.session_state.output_dir = output_dir
                                    
                                    st.info(f"📁 All files saved to: **{output_dir}**")
                        
                        with col2:
                            if st.button("👁️ View Extracted Formulas"):
                                # Use cached formulas if already extracted, otherwise extract now
                                formulas = None
                                if (
                                    hasattr(st.session_state, 'extracted_formulas') and
                                    st.session_state.extracted_formulas is not None and
                                    len(st.session_state.extracted_formulas) > 0
                                ):
                                    # Use cached formulas
                                    formulas = st.session_state.extracted_formulas
                                    st.session_state.extraction_done = 'view'
                                elif (
                                    hasattr(st.session_state, 'results_boxes') and
                                    st.session_state.results_boxes is not None and
                                    len(st.session_state.results_boxes) > 0 and
                                    hasattr(st.session_state, 'opencv_image') and
                                    st.session_state.opencv_image is not None
                                ):
                                    with st.spinner("Extracting and recognizing formulas..."):
                                        st.session_state.extracted_crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)
                                        st.session_state.extracted_formulas = FE.recognize_formulas(st.session_state.extracted_crops, mathargs, mathobjs)
                                        # No Gemini refinement, just enrich if available
                                        if hasattr(FE, 'enrich_formulas_with_descriptions'):
                                            st.session_state.extracted_formulas = FE.enrich_formulas_with_descriptions(st.session_state.extracted_formulas)
                                        st.session_state.extraction_done = 'view'
                                    formulas = st.session_state.extracted_formulas
                                elif (
                                    hasattr(st.session_state, 'extracted_formulas') and
                                    st.session_state.extracted_formulas is not None and
                                    isinstance(st.session_state.extracted_formulas, list) and
                                    len(st.session_state.extracted_formulas) > 0
                                ):
                                    formulas = st.session_state.extracted_formulas
                                # (Removed debug/info message)
                                # Render all formulas in a vertical list of expanders, each with left (image) and right (details)
                                st.subheader("🔍 Extracted Formulas Details")
                                if formulas and len(formulas) > 0:
                                    for formula in formulas:
                                        with st.expander(f"📐 Formula #{formula['id']} (Confidence: {formula['confidence']:.4f})", expanded=False):
                                            col_img, col_latex = st.columns([1,2])
                                            with col_img:
                                                st.write("**Formula Image:**")
                                                coords = formula['coordinates']
                                                crop_img = st.session_state.opencv_image[coords[1]:coords[3], coords[0]:coords[2]]
                                                st.image(crop_img)
                                            with col_latex:
                                                st.write("**LaTeX Formula:**")
                                                st.code(formula['latex'], language='latex')
                                                raw_latex = formula.get('raw_latex')
                                                if raw_latex and raw_latex != formula['latex']:
                                                    st.write("**Model Output (raw):**")
                                                    st.code(raw_latex, language='latex')
                                                st.write("**Rendered:**")
                                                render_latex_block(formula['latex'])
                                                st.write(f"**Bounding Box:** {formula['coordinates']}")
                                                if 'description' in formula and formula['description']:
                                                    st.write("**About this formula:**")
                                                    st.write(formula['description'])
                                else:
                                    st.info("No formulas found to display.")
                        
                        # Show download buttons ONLY if formulas were extracted to file (not just viewed)
                        if st.session_state.output_dir is not None and st.session_state.extraction_done == True:
                            output_dir = st.session_state.output_dir
                            st.subheader("📥 Download Extracted Formulas")
                            dl_col1, dl_col2 = st.columns(2)
                            # PDF Report Download
                            with dl_col1:
                                pdf_file = os.path.join(output_dir, 'formulas_report.pdf')
                                if os.path.exists(pdf_file):
                                    with open(pdf_file, 'rb') as f:
                                        st.download_button(
                                            label="📄 PDF",
                                            data=f.read(),
                                            file_name="formulas_report.pdf",
                                            mime="application/pdf"
                                        )
                            # ZIP Download
                            with dl_col2:
                                zip_file = os.path.join(output_dir, 'extracted_formulas.zip')
                                if os.path.exists(zip_file):
                                    with open(zip_file, 'rb') as f:
                                        st.download_button(
                                            label="📦 ZIP",
                                            data=f.read(),
                                            file_name="extracted_formulas.zip",
                                            mime="application/zip"
                                        )
                                
                        
                        # (Removed duplicate Extracted Formulas Details section)
                        # Do not show yellow warning if formulas are found; only show details or nothing



    #                 col1, col2, col3 = st.columns(3)
    #                 col1.header("Image")
    #                 col2.header("Latext")
    #                 col3.header("Formula")
    #                 if res == "Detection And Recogntion":
    #                     for each_box in results_boxes:
    #                         each_box = list(map(int,each_box))
    #                         crop_box = opencv_image[each_box[1]:each_box[3],each_box[0]:each_box[2],:]
    #                         crop_img = Image.fromarray(np.uint8(crop_box))
    #                         pred = RM.call_model(mathargs, *mathobjs, img=crop_img)
    #                         col1, col2, col3 = st.columns(3)
    #                         with col1:
    #                             st.image(crop_box)
    #                         with col2:
    #                             st.write(pred, width=5)
    #                         with col3:
    #                             st.markdown("$$"+pred+"$$")
    elif inf_style == 'PDF':
        imagem_referencia = st.sidebar.file_uploader("Choose an image", type=["pdf"])
        if st.sidebar.button('Clear uploaded file or image!'):
            st.write("attempt to clear uploaded_file")
            imagem_referencia.seek(0)
    #     res = st.sidebar.radio("Final Result",("Detection","Detection And Recogntion"))

        if imagem_referencia is not None:
            if imagem_referencia.type == "application/pdf":
                # Cache PDF pages to avoid repeated conversions
                if st.session_state.pdf_file_name != imagem_referencia.name:
                    pdf_bytes = imagem_referencia.read()
                    st.session_state.pdf_pages = pdf2image.convert_from_bytes(pdf_bytes)
                    st.session_state.pdf_file_name = imagem_referencia.name
                    # Reset state for new PDF
                    st.session_state.detection_done = False
                    st.session_state.extraction_done = False
                    st.session_state.results_boxes = None
                    st.session_state.extracted_formulas = None
                    st.session_state.extracted_crops = None
                    st.session_state.output_dir = None

                if st.session_state.pdf_pages:
                    page_idx = st.sidebar.number_input("Page Number", min_value=1, max_value=len(st.session_state.pdf_pages), value=1, step=1)
                    page_image = st.session_state.pdf_pages[int(page_idx) - 1]
                    # Convert PIL RGB page to OpenCV BGR for detector/recognizer
                    opencv_image = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
                    st.session_state.opencv_image = opencv_image
                    st.sidebar.image(page_image, caption=f"PDF Page {page_idx}")

                    # Reset detection/extraction state when page changes
                    if st.session_state.pdf_active_page != page_idx:
                        st.session_state.detection_done = False
                        st.session_state.extraction_done = False
                        st.session_state.results_boxes = None
                        st.session_state.extracted_formulas = None
                        st.session_state.extracted_crops = None
                        st.session_state.output_dir = None
                        st.session_state.pdf_active_page = page_idx

                    if st.button('Launch the Detection!', key='pdf_detect'):
                        results_boxes = MD.predict_formulas(opencv_image, math_model)
                        st.session_state.results_boxes = results_boxes
                        st.session_state.detection_done = True

                    if st.session_state.detection_done and st.session_state.results_boxes is not None:
                        results_boxes = st.session_state.results_boxes
                        images_rectangles = opencv_image.copy()
                        draw_rectangles(images_rectangles, results_boxes)
                        st.image(images_rectangles, caption=f"Detections on Page {page_idx}")

                        if len(results_boxes) > 0:
                            st.success(f"✓ Found {len(results_boxes)} formulas on page {page_idx}!")

                            col1, col2 = st.columns(2)

                            with col1:
                                if st.button("🚀 Extract Formulas to File (PDF)", key='pdf_extract_files'):
                                    with st.spinner("Extracting and recognizing formulas from PDF page..."):
                                        from datetime import datetime
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        output_dir = f"extracted_output_pdf_p{page_idx}_{timestamp}"
                                        os.makedirs(output_dir, exist_ok=True)

                                        # Only extract and recognize if not already done
                                        if not hasattr(st.session_state, 'extracted_formulas') or st.session_state.extracted_formulas is None:
                                            st.session_state.extracted_crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)
                                            st.session_state.extracted_formulas = FE.recognize_formulas(st.session_state.extracted_crops, mathargs, mathobjs)
                                            # Skip slow Gemini refinement for faster response
                                            # Enrich with AI descriptions if available
                                            st.session_state.extracted_formulas = FE.enrich_formulas_with_descriptions(st.session_state.extracted_formulas)

                                        formulas = st.session_state.extracted_formulas
                                        extracted_crops = st.session_state.extracted_crops

                                        pdf_path = os.path.join(output_dir, 'formulas_report.pdf')
                                        FE.save_pdf_report(formulas, extracted_crops=extracted_crops, output_path=pdf_path, original_image=st.session_state.opencv_image)

                                        annotated_path = os.path.join(output_dir, 'annotated_image.png')
                                        FE.save_annotated_image(st.session_state.opencv_image, formulas, annotated_path)

                                        formula_dir = os.path.join(output_dir, 'formula_images')
                                        os.makedirs(formula_dir, exist_ok=True)
                                        for idx, crop_data in enumerate(extracted_crops):
                                            img_path = os.path.join(formula_dir, f'formula_{idx+1:04d}.png')
                                            cv2.imwrite(img_path, crop_data['image'])

                                        zip_path = os.path.join(output_dir, 'extracted_formulas.zip')
                                        with zipfile.ZipFile(zip_path, 'w') as zip_file:
                                            for root, dirs, files in os.walk(output_dir):
                                                for file in files:
                                                    if not file.endswith('.zip'):
                                                        file_path = os.path.join(root, file)
                                                        arcname = os.path.relpath(file_path, output_dir)
                                                        zip_file.write(file_path, arcname)

                                        st.session_state.extraction_done = True
                                        st.session_state.output_dir = output_dir

                                        st.success(f"✓ Successfully extracted {len(formulas)} formulas!")
                                        st.info(f"📁 All files saved to: **{output_dir}**")

                            with col2:
                                if st.button("👁️ View Extracted Formulas (PDF)", key='pdf_view_extract'):
                                    # Use cached formulas if already extracted
                                    if not hasattr(st.session_state, 'extracted_formulas') or st.session_state.extracted_formulas is None:
                                        with st.spinner("Extracting and recognizing formulas from PDF page..."):
                                            st.session_state.extracted_crops = FE.extract_formula_crops(st.session_state.opencv_image, st.session_state.results_boxes)
                                            st.session_state.extracted_formulas = FE.recognize_formulas(st.session_state.extracted_crops, mathargs, mathobjs)
                                            # Skip slow Gemini refinement for faster response
                                            st.session_state.extracted_formulas = FE.enrich_formulas_with_descriptions(st.session_state.extracted_formulas)
                                    st.session_state.extraction_done = 'view'

                            if st.session_state.extraction_done == True:
                                if st.session_state.extracted_formulas is not None and st.session_state.output_dir is not None:
                                    formulas = st.session_state.extracted_formulas
                                    output_dir = st.session_state.output_dir

                                    st.success(f"✓ Successfully extracted {len(formulas)} formulas!")
                                    st.info(f"📁 All files saved to: **{output_dir}**")

                                    st.subheader("📥 Download Extracted Formulas")

                                    dl_col1, dl_col2 = st.columns(2)

                                    with dl_col1:
                                        pdf_file = os.path.join(output_dir, 'formulas_report.pdf')
                                        if os.path.exists(pdf_file):
                                            with open(pdf_file, 'rb') as f:
                                                st.download_button(
                                                    label="📄 PDF",
                                                    data=f.read(),
                                                    file_name="formulas_report.pdf",
                                                    mime="application/pdf"
                                                )

                                    with dl_col2:
                                        zip_file = os.path.join(output_dir, 'extracted_formulas.zip')
                                        if os.path.exists(zip_file):
                                            with open(zip_file, 'rb') as f:
                                                st.download_button(
                                                    label="📦 ZIP",
                                                    data=f.read(),
                                                    file_name="extracted_formulas.zip",
                                                    mime="application/zip"
                                                )

                                    st.success("✓ All files are ready for download!")

                            if st.session_state.extraction_done == 'view':
                                if st.session_state.extracted_formulas is not None:
                                    formulas = st.session_state.extracted_formulas

                                    st.subheader("🔍 Extracted Formulas Details")

                                    for formula in formulas:
                                        with st.expander(f"📐 Formula #{formula['id']} (Confidence: {formula['confidence']:.4f})", expanded=False):
                                            exp_col1, exp_col2 = st.columns(2)

                                            with exp_col1:
                                                st.write("**Formula Image:**")
                                                coords = formula['coordinates']
                                                crop_img = st.session_state.opencv_image[coords[1]:coords[3], coords[0]:coords[2]]
                                                st.image(crop_img)
                                                if 'description' in formula and formula['description']:
                                                    st.write("**About this formula:**")
                                                    st.write(formula['description'])

                                            with exp_col2:
                                                st.write("**LaTeX Formula:**")
                                                st.code(formula['latex'], language='latex')
                                                st.write("**Rendered:**")
                                                render_latex_block(formula['latex'])
                                                st.write(f"**Bounding Box:** {formula['coordinates']}")
                        else:
                            st.warning(f"No formulas detected on page {page_idx}.")


