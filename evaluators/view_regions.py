# app_regions_viewer.py
from __future__ import annotations
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import os
from config import local_params as lp

# ---------------------------
# Helpers for PDF rendering
# ---------------------------
def render_page_image(pdf_path: Path, page_index: int, dpi: int = 200) -> Tuple[np.ndarray, float, float, float]:
    """
    Render a page to a numpy RGB image and return:
    (img_np, width_px, height_px, scale_px_per_point)
    """
    print(f"[DEBUG] render_page_image: {pdf_path}, page_index={page_index}, dpi={dpi}")
    doc = fitz.open(pdf_path)
    print(f"[DEBUG] PDF opened, num pages: {len(doc)}")
    if page_index < 0 or page_index >= len(doc):
        print(f"[DEBUG] page_index {page_index} out of range, using 0")
        page_index = 0
    page = doc[page_index]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)
    width_px, height_px = img_np.shape[1], img_np.shape[0]
    scale = zoom  # px per PDF point
    doc.close()
    print(f"[DEBUG] Rendered page: {width_px}x{height_px} px, scale={scale}")
    return img_np, float(width_px), float(height_px), float(scale)


def get_pdf_page_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n


def image_to_fig(img_np: np.ndarray) -> go.Figure:
    """
    Create a base figure with the page image.
    Coordinate system will be pixels; we flip y-axis so (0,0) is top-left.
    """
    h, w = img_np.shape[0], img_np.shape[1]
    fig = go.Figure(go.Image(z=img_np))
    fig.update_xaxes(visible=False, range=[0, w])                  # left->right
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor='x') # top->bottom
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        hovermode="closest",
    )
    return fig


# ---------------------------
# Regions overlay (polygons)
# ---------------------------
LABEL_COLORS = {
    "text":   "rgba(0, 120, 255, 0.18)",
    "table":  "rgba(0, 180, 0, 0.22)",
    "figure": "rgba(255, 165, 0, 0.22)",
    "other":  "rgba(130, 130, 130, 0.18)",
}

LINE_COLORS = {
    "text":   "rgb(0, 120, 255)",
    "table":  "rgb(0, 160, 0)",
    "figure": "rgb(200, 120, 0)",
    "other":  "rgb(100, 100, 100)",
}

def df_regions_for_page(regions_df: pd.DataFrame, page_index: int) -> pd.DataFrame:
    if regions_df is None or regions_df.empty:
        return pd.DataFrame(columns=["page_index","region_id","label","score","source","x0","y0","x1","y1"])
    return regions_df.loc[regions_df["page_index"] == page_index].copy()

def polygons_by_label(df_page: pd.DataFrame, scale: float, label_filter: List[str], min_score: float):
    """
    Return dict(label -> scatter trace data) for polygons on this page.
    Coordinates are converted from PDF points to pixels with 'scale'.
    """
    data_by_label: Dict[str, Dict[str, list]] = {}
    if df_page.empty:
        return data_by_label

    # Filter
    if label_filter:
        df_page = df_page[df_page["label"].isin(label_filter)]
    if "score" in df_page.columns:
        df_page = df_page[(df_page["score"].isna()) | (df_page["score"] >= min_score)]

    for _, r in df_page.iterrows():
        label = r.get("label", "other")
        x0, y0, x1, y1 = float(r["x0"])*scale, float(r["y0"])*scale, float(r["x1"])*scale, float(r["y1"])*scale
        # Build a closed rectangle polygon
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        text = f"{label.upper()}<br>id: {r['region_id']}<br>score: {r.get('score', None)}<br>source: {r.get('source','')}"
        if label not in data_by_label:
            data_by_label[label] = {"xs": [], "ys": [], "texts": [], "ids": []}
        # Use 'None' separators to draw multiple polygons in one trace
        data_by_label[label]["xs"].extend(xs + [None])
        data_by_label[label]["ys"].extend(ys + [None])
        data_by_label[label]["texts"].extend([text]*len(xs) + [None])
        data_by_label[label]["ids"].extend([r["region_id"]]*len(xs) + [None])

    return data_by_label


def add_region_traces(fig: go.Figure, data_by_label: Dict[str, Dict[str, list]]):
    # One trace per label so user can toggle visibility from legend
    for label, payload in data_by_label.items():
        fig.add_trace(go.Scatter(
            x=payload["xs"],
            y=payload["ys"],
            mode="lines",
            name=label,
            line=dict(width=2, color=LINE_COLORS.get(label, "black")),
            fill="toself",
            fillcolor=LABEL_COLORS.get(label, "rgba(0,0,0,0.1)"),
            hoverinfo="text",
            hovertext=payload["texts"],
            customdata=payload["ids"],
        ))


def add_selected_outline(fig: go.Figure, df_page: pd.DataFrame, selected_region_id: str | None, scale: float):
    if not selected_region_id:
        return
    row = df_page.loc[df_page["region_id"] == selected_region_id]
    if row.empty:
        return
    r = row.iloc[0]
    x0, y0, x1, y1 = float(r["x0"])*scale, float(r["y0"])*scale, float(r["x1"])*scale, float(r["y1"])*scale
    fig.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color="magenta", width=4),
        fillcolor="rgba(0,0,0,0)"
    )


# ---------------------------
# Dash app
# ---------------------------
app: Dash = Dash(__name__)
app.title = "PDF Region Visualizer"

def find_all_pdfs_and_regions():
    input_root = Path("/home/rober/synth-reader/input_data")
    out_root = Path("/home/rober/synth-reader/out_regions")
    options = []
    for pdf_path in input_root.rglob("*.pdf"):
        stem = pdf_path.stem
        out_dir = out_root / stem
        region_file = out_dir / f"{stem}_regions.parquet"
        if region_file.exists():
            label = f"{pdf_path.relative_to(input_root)}"
            options.append({
                "label": label,
                "value": str(pdf_path),
                "region_file": str(region_file)
            })
    return sorted(options, key=lambda x: x["label"])

PDF_OPTIONS = find_all_pdfs_and_regions()

app.layout = html.Div([
    html.Div([
        html.Label("Select Document"),
        dcc.Dropdown(
            id="pdf-dropdown",
            options=[{"label": o["label"], "value": o["value"]} for o in PDF_OPTIONS],
            value=PDF_OPTIONS[0]["value"] if PDF_OPTIONS else None,
            style={"width": "100%"}
        ),
    ], style={"margin": "10px 0"}),
    html.Div([
        html.Div([
            html.Label("DPI"),
            dcc.Input(id="dpi", type="number", value=200, min=100, step=50, style={"width":"100%"}),
        ], style={"flex":"1", "padding":"0 8px"}),
    ], style={"display":"flex", "margin":"10px 0"}),

    html.Div([
        html.Div([
            html.Label("Page"),
            dcc.Slider(id="page-slider", min=1, max=1, step=1, value=1,
                       tooltip={"placement":"bottom", "always_visible":True}),
        ], style={"flex":"4", "padding":"0 8px"}),

        html.Div([
            html.Label("Labels"),
            dcc.Dropdown(
                id="label-filter",
                options=[{"label": l, "value": l} for l in ["text","table","figure","other"]],
                value=["text","table","figure","other"],
                multi=True
            ),
        ], style={"flex":"3", "padding":"0 8px"}),

        html.Div([
            html.Label("Min score"),
            dcc.Slider(id="score-thresh", min=0, max=1, step=0.05, value=0.0),
        ], style={"flex":"3", "padding":"0 8px"}),
    ], style={"display":"flex", "margin":"10px 0"}),

    dcc.Graph(id="page-graph", style={"height":"88vh", "border":"1px solid #ddd"}),

    # Stores
    dcc.Store(id="pdf-meta"),         # {"n_pages": int}
    dcc.Store(id="regions-data"),     # JSON: list of dict rows
    dcc.Store(id="selected-region"),  # region_id string
])

# Remove pdf-path and regions-path from the UI and callbacks
# Instead, use the dropdown value to determine both paths

@app.callback(
    Output("pdf-meta", "data"),
    Output("regions-data", "data"),
    Output("page-slider", "min"),
    Output("page-slider", "max"),
    Output("page-slider", "value"),
    Input("pdf-dropdown", "value"),
)
def load_inputs_auto(selected_pdf):
    print(f"[DEBUG] load_inputs_auto: selected_pdf={selected_pdf}")
    if not selected_pdf:
        print("[DEBUG] No PDF selected")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    pdf_path = Path(selected_pdf)
    stem = pdf_path.stem
    region_file = Path("/home/rober/synth-reader/out_regions") / stem / f"{stem}_regions.parquet"
    if not region_file.exists():
        print(f"[DEBUG] Region file not found: {region_file}")
        raise dash.exceptions.PreventUpdate
    if not pdf_path.exists():
        print(f"[DEBUG] PDF not found: {pdf_path}")
        raise dash.exceptions.PreventUpdate
    print(f"[DEBUG] Using PDF: {pdf_path}, regions: {region_file}")
    n_pages = get_pdf_page_count(pdf_path)
    regions_df = pd.read_parquet(region_file)
    needed = {"pdf_name","page_index","region_id","label","score","source","x0","y0","x1","y1","is_scanned_page"}
    if not needed.issubset(set(regions_df.columns)):
        print(f"[DEBUG] regions parquet missing required columns: {needed - set(regions_df.columns)}")
        raise ValueError(f"regions parquet missing required columns: {needed - set(regions_df.columns)}")
    print(f"[DEBUG] Loaded regions_df: {len(regions_df)} rows")
    return (
        {"n_pages": n_pages, "pdf_path": str(pdf_path)},
        regions_df.to_dict("records"),
        1,
        n_pages,
        1,
    )

# In update_figure, get the PDF path from pdf-meta
@app.callback(
    Output("page-graph", "figure"),
    Output("selected-region", "data"),
    Input("page-slider", "value"),
    Input("label-filter", "value"),
    Input("score-thresh", "value"),
    Input("dpi", "value"),
    Input("page-graph", "clickData"),
    State("pdf-meta", "data"),
    State("regions-data", "data"),
    State("selected-region", "data"),
)
def update_figure(page_value, labels_filter, score_thresh, dpi, clickData, pdf_meta, regions_data, selected_region):
    pdf_path = pdf_meta["pdf_path"] if pdf_meta and "pdf_path" in pdf_meta else None
    print(f"[DEBUG] update_figure: page_value={page_value}, pdf_path={pdf_path}, dpi={dpi}")
    if not pdf_path or not regions_data:
        print("[DEBUG] No pdf_path or regions_data")
        fig = go.Figure()
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        return fig, selected_region
    page_index = int(page_value) - 1  # slider is 1-based for humans
    pdfp = Path(pdf_path)
    try:
        img_np, w_px, h_px, scale = render_page_image(pdfp, page_index, dpi=dpi or 200)
        print(f"[DEBUG] Got image for page {page_index}")
    except Exception as e:
        print(f"[DEBUG] Exception in render_page_image: {e}")
        fig = go.Figure()
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        return fig, selected_region
    fig = image_to_fig(img_np)
    regions_df = pd.DataFrame(regions_data)
    df_page = df_regions_for_page(regions_df, page_index)
    data_by_label = polygons_by_label(df_page, scale=scale, label_filter=labels_filter or [], min_score=score_thresh or 0.0)
    add_region_traces(fig, data_by_label)
    add_selected_outline(fig, df_page, selected_region, scale=scale)
    return fig, selected_region


if __name__ == "__main__":
    app.run(debug=True)
