import json
import os
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

import analyze_capture as ac

st.set_page_config(
    page_title="SiteSee Drone Capture QA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .reportview-container, .sidebar .sidebar-content {
        background: #111218;
        color: #e3e8ef;
    }
    .stButton>button {
        background-color: #0f72ff;
        color: white;
    }
    .st-badge {
        background: #1f2937;
        color: #fafafa;
        padding: 8px 12px;
        border-radius: 12px;
    }
    .stDownloadButton>button {
        background: #272e3a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SiteSee Drone Capture QA")
st.write("Upload a ZIP capture bundle or choose a local capture folder for fast drone-quality analysis.")

# Warning about large captures
st.warning(
    "⚠️ **Note on Large Captures**: This tool is optimized for sample analysis (50-200 images). "
    "For production processing of 1000+ image captures, use local batch mode: `python3 analyze_capture.py --folder /path/to/capture --dataset`. "
    "See README.md for deployment recommendations."
)

base_datasets = Path("./datasets")
dataset_options = [""]
if base_datasets.exists() and base_datasets.is_dir():
    dataset_options += [str(path.relative_to('.')) for path in sorted(base_datasets.iterdir()) if path.is_dir()]

with st.sidebar:
    st.header("Capture selection")
    selected_dataset = st.selectbox("Choose a dataset folder", dataset_options)
    folder_input = st.text_input("Or enter a local folder path", value=selected_dataset)
    uploaded_zip = st.file_uploader("Upload a ZIP file", type=["zip"])
    export_dataset = st.checkbox("Export training_dataset.csv", value=True)

    st.markdown("### Capture review metadata")
    gcp_evidence_status = st.selectbox("GCP evidence present?", ["unknown", "yes", "no"], index=0)
    gcp_image_count = st.number_input("Number of GCP images", min_value=0, max_value=100, value=0)
    scale_reference_status = st.selectbox("Scale reference present?", ["unknown", "yes", "no"], index=0)
    capture_position = st.selectbox("Capture position", ["unknown", "ground", "roof"], index=0)
    roof_access_available = st.selectbox("Roof access available?", ["unknown", "yes", "no"], index=0)
    reviewer_notes = st.text_area("Reviewer notes", value="", height=120)

    st.markdown("### Optional metadata files")
    apriltag_metadata_json = st.file_uploader("Upload SiteSee job metadata JSON", type=["json"])
    gcp_csv = st.file_uploader("Upload GCP CSV (optional)", type=["csv"])
    ground_truth_csv = st.file_uploader("Upload ground-truth CSV (optional)", type=["csv"])

    analyze_clicked = st.button("Analyze capture")


def find_image_root(folder_path: Path):
    if any(file.suffix.lower() in {".jpg", ".jpeg"} for file in folder_path.iterdir() if file.is_file()):
        return folder_path

    for child in sorted(folder_path.iterdir()):
        if child.is_dir() and any(
            file.suffix.lower() in {".jpg", ".jpeg"} for file in child.rglob("*")
        ):
            return child

    return None


def analyze_folder(folder_path: str, review_metadata: dict, gcp_csv_path=None, ground_truth_csv_path=None):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Capture folder not found: {folder}")

    metrics, summary_data, continuity, gps_stability_score = ac.run_capture_analysis(
        str(folder),
        export_csv=True,
        export_html=True,
        export_dataset=export_dataset,
        report_path=ac.OUTPUT_HTML,
        review_metadata=review_metadata,
        gcp_csv_path=gcp_csv_path,
        ground_truth_csv_path=ground_truth_csv_path,
    )
    return metrics, summary_data, continuity, gps_stability_score


output_status = None
analysis_results = None

if analyze_clicked:
    review_metadata = {
        "gcp_evidence_status": gcp_evidence_status,
        "gcp_image_count": gcp_image_count,
        "scale_reference_status": scale_reference_status,
        "capture_position": capture_position,
        "roof_access_available": roof_access_available,
        "reviewer_notes": reviewer_notes,
    }
    
    # Handle optional metadata JSON
    temp_gcp_csv_path = None
    temp_ground_truth_csv_path = None
    
    if apriltag_metadata_json is not None:
        try:
            review_metadata["job_metadata"] = json.loads(apriltag_metadata_json.getvalue().decode("utf-8"))
        except Exception as exc:
            st.error(f"Unable to parse uploaded metadata JSON: {exc}")
    
    # Handle optional GCP CSV
    if gcp_csv is not None:
        temp_gcp_csv_path = f"/tmp/gcp_{id(gcp_csv)}.csv"
        with open(temp_gcp_csv_path, "wb") as f:
            f.write(gcp_csv.getvalue())
    
    # Handle optional ground-truth CSV
    if ground_truth_csv is not None:
        temp_ground_truth_csv_path = f"/tmp/ground_truth_{id(ground_truth_csv)}.csv"
        with open(temp_ground_truth_csv_path, "wb") as f:
            f.write(ground_truth_csv.getvalue())
    
    if uploaded_zip is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "capture.zip"
            zip_path.write_bytes(uploaded_zip.getvalue())
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(temp_dir)
            image_root = find_image_root(Path(temp_dir))
            if image_root is None:
                st.error("Uploaded ZIP does not contain JPG/JPEG images in a discoverable folder.")
            else:
                output_status = f"Analyzing extracted ZIP folder: {image_root}"
                with st.spinner(output_status):
                    analysis_results = analyze_folder(str(image_root), review_metadata, temp_gcp_csv_path, temp_ground_truth_csv_path)
    else:
        candidate = folder_input.strip() or selected_dataset
        if not candidate:
            st.error("Please select a dataset folder or enter a local folder path.")
        else:
            if not Path(candidate).is_dir():
                st.error(f"Folder not found: {candidate}")
            else:
                output_status = f"Analyzing folder: {candidate}"
                with st.spinner(output_status):
                    analysis_results = analyze_folder(candidate, review_metadata, temp_gcp_csv_path, temp_ground_truth_csv_path)

if analysis_results is not None:
    metrics, summary_data, continuity, gps_stability_score = analysis_results

    badge_color = "#21ba45" if summary_data["risk_label"] == "LOW RISK" else "#f2c037" if summary_data["risk_label"] == "REVIEW RECOMMENDED" else "#db2828"
    badge_html = f"<div style='display:inline-flex;align-items:center;gap:10px;padding:12px 18px;border-radius:999px;background:{badge_color};color:#111;font-weight:700;'>{summary_data['risk_label']} &nbsp; {summary_data['final_risk_score']}/100</div>"
    st.markdown(badge_html, unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Summary")
        st.write(
            f"**Risk reason:** {summary_data['failure_reason']}  \
             **Risk severity:** {summary_data['failure_severity']}  \
             **Confidence:** {summary_data['confidence_score']}/100"
        )
        metrics_grid = {
            "GPS stability": gps_stability_score,
            "Blurry images": summary_data["blurry_images"],
            "Exposure warnings": summary_data["exposure_warnings"],
            "True altitude drift events": summary_data["suspected_battery_swaps"],
            "GCP evidence status": summary_data.get("gcp_evidence_status", "N/A"),
            "Scale evidence status": summary_data.get("scale_reference_status", "N/A"),
            "GCP evidence quality": summary_data.get("gcp_evidence_quality", "N/A"),
            "Scale evidence quality": summary_data.get("scalepoint_quality", "N/A"),
            "Apriltag detected": summary_data.get("apriltag_detected", False),
            "Unique AprilTags": summary_data.get("apriltag_count", 0),
            "AprilTag images": summary_data.get("apriltag_image_count", 0),
            "Detections per image": summary_data.get("apriltag_detection_rate_per_image", 0.0),
            "Mission interruption": "YES" if continuity["mission_interruption"] else "NO",
        }
        for name, value in metrics_grid.items():
            st.metric(label=name, value=value)

    with right:
        st.subheader("Capture status")
        st.write(f"**Images analyzed:** {summary_data['images_analyzed']}")
        st.write(f"**Max altitude jump:** {summary_data['max_z_jump_m']} m")
        st.write(f"**True altitude drift:** {summary_data.get('true_altitude_drift_m', summary_data.get('true_battery_swap_drift_m', 0))} m")
        st.write(f"**Drift severity:** {summary_data.get('true_altitude_drift_severity', 'UNKNOWN')}")
        st.write(f"**Timestamp gaps:** {summary_data['timestamp_gap_events']}")
        st.write(f"**Evidence guidance:** {summary_data.get('evidence_advice', summary_data.get('MARKER_ADVICE', 'No guidance available.'))}")

    flagged_images = [item for item in metrics if item.get("thumbnail_refs")]
    if flagged_images:
        st.subheader("Flagged image thumbnails")
        thumbs = []
        for item in flagged_images:
            for thumb in item.get("thumbnail_refs", []):
                if Path(thumb).exists():
                    thumbs.append((item["file"], thumb))
        cols = st.columns(4)
        for idx, (filename, thumb_path) in enumerate(thumbs):
            with cols[idx % 4]:
                st.image(str(thumb_path), caption=filename, use_column_width=True)
    else:
        st.info("No flagged images detected for thumbnail preview.")

    st.subheader("Download reports")
    files = [
        ("HTML report", ac.OUTPUT_HTML),
        ("Summary CSV", ac.OUTPUT_SUMMARY_CSV),
        ("Details CSV", ac.OUTPUT_DETAILS_CSV),
        ("Training dataset CSV", ac.OUTPUT_DATASET_CSV),
    ]
    for label, file_name in files:
        file_path = Path(file_name)
        if file_path.exists():
            with open(file_path, "rb") as fp:
                st.download_button(label=f"Download {label}", data=fp.read(), file_name=file_path.name, mime="application/octet-stream")
        else:
            st.write(f"{label} not available yet.")

    st.write("---")
    st.write("Analysis outputs are generated in the current app folder.")
