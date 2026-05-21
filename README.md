# SiteSee Drone Capture QA Analyzer

**Telecom Drone Capture Risk Intelligence System**

Analyzes drone image captures for reconstruction risk assessment using image quality metrics, GPS/altitude stability, mission continuity, and optional AprilTag metadata validation.

## Features

- 🖼️ **Image Quality Analysis**: Blur variance, exposure scoring, brightness analysis
- 📍 **GPS/Altitude Stability**: Haversine distance calculations, altitude drift detection, battery swap identification
- ⏱️ **Mission Continuity**: Timestamp gap detection, capture cadence consistency, interruption flagging
- 🏷️ **AprilTag Evidence**: SiteSee apriltagStats metadata parsing, GCP/scale detection, tag quality scoring
- 📊 **Flexible Metadata**: Optional job metadata JSON, GCP control point CSV, ground-truth validation CSV
- 📈 **Risk Scoring**: Weighted combination (0-100 scale) with tri-state assessment (LOW RISK / REVIEW RECOMMENDED / HIGH RISK)
- 📋 **Export Options**: HTML reports, CSV summaries, training datasets
- 🌐 **Web UI**: Streamlit app for interactive analysis with ZIP upload and local folder selection

## System Requirements

### Required
- Python 3.8+
- ExifTool (system binary) - [install instructions](https://exiftool.org/install.html)
  - macOS: `brew install exiftool`
  - Linux: `sudo apt-get install exiftool`
  - Windows: Download from exiftool.org

### Python Dependencies
See `requirements.txt` for full list:
- Streamlit ≥1.28.0
- OpenCV (cv2) ≥4.8.0
- NumPy ≥1.24.0
- Pillow ≥10.0.0

## Installation

### Local Development

```bash
# Clone or download the repository
cd dronecapture-q-a

# Create a Python virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Streamlit Cloud Deployment

1. Push repository to GitHub (must include `requirements.txt`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repository, branch, and entry point: `app.py`
4. Streamlit Cloud will auto-detect and install dependencies from `requirements.txt`

**Note**: ExifTool is pre-installed on Streamlit Cloud servers.

## Usage

### Web Application (Streamlit)

```bash
streamlit run app.py
```

Opens interactive UI at `http://localhost:8501` (or deployed cloud URL)

**Sidebar Controls:**
- Dataset folder selection (from local `./datasets/` subdirectory)
- Local folder path text input
- ZIP file uploader
- Training dataset export checkbox
- Capture review metadata (optional):
  - GCP evidence status
  - GCP image count
  - Scale reference status
  - Capture position
  - Roof access availability
  - Reviewer notes textarea
- Optional metadata file uploaders:
  - SiteSee job metadata JSON
  - GCP control points CSV
  - Ground-truth CSV for validation

**Outputs:**
- Risk badge with color-coded assessment and 0-100 score
- Risk reason, severity, and confidence
- Metrics grid: GPS stability, blurry %, exposure warnings, battery swaps, AprilTag counts
- Flagged image thumbnails in 4-column grid
- Download buttons:
  - HTML report (`capture_report.html`)
  - CSV summary (`capture_summary.csv`)
  - CSV details (`capture_details.csv`)
  - Training dataset (`training_dataset.csv`)

### CLI (Command-line)

```bash
python3 analyze_capture.py --folder ./images --raw
```

**Arguments:**
- `--folder PATH`: Path to image folder (default: `./images`)
- `--raw`: Print per-image metrics table to console
- `--dataset`: Export training_dataset.csv

## Input Format & Optional Metadata

### Images (Required)
- Format: JPG or JPEG files
- Source: Local folder or uploaded ZIP
- EXIF data extracted automatically (GPS, altitude, timestamp)

### Job Metadata JSON (Optional)
Enables AprilTag evidence scoring. Expected structure:
```json
{
  "job_metadata": {
    "apriltagStats": {
      "apriltag_detected": true,
      "apriltag_count": 42,
      "apriltag_detection_count": 128,
      "apriltag_image_count": 30,
      "gcp_detected": true,
      "scalepoint_detected": false,
      "gcp_quality": "GOOD",
      "scalepoint_quality": "UNKNOWN",
      "weak_tag_ids": [],
      "strong_tag_ids": [1, 2, 3, 4, 5]
    }
  }
}
```

### GCP CSV (Optional)
Control points from photogrammetry. Expected columns: `point_id`, `x`, `y`, `z` (or any schema; row count used).

### Ground-Truth CSV (Optional)
For validation comparison. Expected columns:
- `outcome`: "success" or "failed"
- `reconstruction_success`: "true" or "false"
- `quality_score`: numeric score or text
- `notes`: optional field

## Output Formats

### HTML Report (`capture_risk_assessment.html`)
9-section interactive report:
1. **Risk Summary**: Stat grid with all metrics
2. **GPS/Altitude Stability**: Score, jump count, drift
3. **Mission Continuity**: Interruption status, gaps, cadence
4. **Image Quality**: Blur %, exposure %, battery bars
5. **GCP/Scale/AprilTag Evidence**: Marker counts, detection rate, quality
6. **Altitude Drift Diagnostics**: Battery swap boundaries
7. **Risk Factors**: Label breakdown
8. **Visual Evidence**: Flagged image thumbnails
9. **Pilot Actions & Reviewer Notes**: Recommendations

### CSV Summary (`capture_summary.csv`)
Single row with all aggregated metrics.

### CSV Details (`capture_details.csv`)
Per-image metrics: file, timestamp, GPS, altitude, blur, exposure, jumps, flags.

### Training Dataset (`training_dataset.csv`)
Per-image data for model training/validation.

## Important: Large Capture Deployments

⚠️ **This tool is optimized for interactive analysis of sample captures (50-200 images).**

**For production processing of 1000+ image captures:**
1. **Use local batch mode** on deployment server with adequate CPU/RAM:
   ```bash
   python3 analyze_capture.py --folder /path/to/capture --dataset
   ```
2. **Use cloud storage** (Google Cloud Storage, AWS S3) to:
   - Store large capture ZIPs
   - Stream images without uploading to web server
   - Archive results
3. **Streamlit Cloud limitations**:
   - File upload max: ~200 MB
   - Session memory: ~1 GB
   - Processing timeout: ~45 minutes
   - No persistent disk storage

**Recommended workflow for large captures:**
- Web UI: Use for sample validation (50-200 images)
- Production: Use CLI on server with local SSD, process in parallel batches

## Risk Scoring Model

**Weighted components (0-100 scale):**
- Altitude jumps (Z-axis): 25%
- GPS drift (XY): 25%
- Blur variance: 40%
- Exposure issues: 30%
- Timestamp gaps: 20%

**Risk Assessment:**
- **LOW RISK** (score < 35): Safe for reconstruction
- **REVIEW RECOMMENDED** (score 35-65): Manual inspection suggested
- **HIGH RISK** (score ≥ 65): Significant reconstruction challenges predicted

## Deployment Checklist

- [ ] ExifTool installed on server
- [ ] Python 3.8+ available
- [ ] `requirements.txt` in repository root
- [ ] No hardcoded local paths (uses `./datasets/` for local testing, temp folders for uploads)
- [ ] `.streamlit/config.toml` configured (if custom settings needed)
- [ ] `.gitignore` excludes build artifacts, Python cache, generated reports
- [ ] README.md present with installation/usage instructions
- [ ] Test: `streamlit run app.py` runs without errors
- [ ] Test: File upload and analysis completes within timeout
- [ ] Document: Large capture limitations in UI or README

## Troubleshooting

### "exiftool not found"
Install ExifTool system binary:
- macOS: `brew install exiftool`
- Linux: `sudo apt-get install exiftool`
- Windows: Download from [exiftool.org](https://exiftool.org/install.html), add to PATH

### "ModuleNotFoundError: No module named 'cv2'"
Run: `pip install -r requirements.txt`

### "ZIP upload timeout on Streamlit Cloud"
- Reduce file size (upload sample of 50-100 images instead of 1000+)
- Use local folder mode if available
- Process large captures locally with CLI mode

### "No JPGs found in uploaded ZIP"
Ensure JPG/JPEG files are directly in ZIP or in a single subdirectory. Nested folder structures should work.

## Development

### Project Structure
```
dronecapture-q-a/
├── app.py                    # Streamlit web UI
├── analyze_capture.py        # Core analysis engine
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git exclusions
├── .streamlit/
│   └── config.toml          # Streamlit settings (optional)
├── datasets/                # Local test datasets (optional)
│   ├── sample_capture_1/
│   ├── sample_capture_2/
│   └── ...
└── images/                  # Default images folder for CLI
```

### Adding New Metrics
Edit `analyze_capture.py`:
1. Add field to `get_image_metrics()` return dict
2. Update `compute_capture_summary()` to aggregate
3. Add to `summary_data` dict
4. Include in `write_html_report()` template

## License & Attribution

**SiteSee Drone Capture Risk Intelligence System**

Part of SiteSee infrastructure for mobile network site assessment.

## Support

For issues, feature requests, or questions:
1. Check troubleshooting section above
2. Verify ExifTool is installed and working: `exiftool -ver`
3. Test with sample captures in `./datasets/`
4. Review generated HTML reports for diagnostic details

---

**Last Updated**: May 2026
