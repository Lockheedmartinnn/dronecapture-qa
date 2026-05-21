import argparse
import csv
import datetime
import json
import math
import os
import statistics
import subprocess
import sys

IMAGE_FOLDER = "./images"

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

# Thresholds and scoring constants used by the analyzer.
BLUR_VARIANCE_THRESHOLD = 75.0
SEVERE_UNDEREXPOSED = 0.15
SEVERE_OVEREXPOSED = 0.98
SEVERE_DARK_PERCENT = 20.0
SEVERE_BRIGHT_PERCENT = 15.0
GPS_WARNING_DISTANCE = 30.0
TIMESTAMP_GAP_SECONDS = 300
OUTPUT_HTML = "capture_report.html"
CAPTURE_RISK_ASSESSMENT_HTML = "capture_risk_assessment.html"
OUTPUT_SUMMARY_CSV = "capture_summary.csv"
OUTPUT_DETAILS_CSV = "capture_details.csv"
OUTPUT_DATASET_CSV = "training_dataset.csv"


# Read Exif metadata from an image file using ExifTool.
def run_exiftool(image_path):
    result = subprocess.run(
        [
            "exiftool",
            "-json",
            "-n",
            "-CreateDate",
            "-GPSLatitude",
            "-GPSLongitude",
            "-GPSAltitude",
            "-RelativeAltitude",
            "-AbsoluteAltitude",
            image_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"exiftool failed for {image_path}: {result.stderr.strip()}")

    data = json.loads(result.stdout)[0]
    return data


# Convert Exif timestamp strings into datetime objects.
def parse_timestamp(value):
    if not value:
        return None

    for fmt in ["%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"]:
        try:
            return datetime.datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None


# Extract both metadata and image quality metrics for a single image.
def get_image_metrics(image_path):
    exif = run_exiftool(image_path)
    metadata = {
        "file": os.path.basename(image_path),
        "create_date": parse_timestamp(exif.get("CreateDate")),
        "gps_latitude": exif.get("GPSLatitude"),
        "gps_longitude": exif.get("GPSLongitude"),
        "gps_altitude": exif.get("GPSAltitude"),
        "relative_altitude": exif.get("RelativeAltitude"),
        "absolute_altitude": exif.get("AbsoluteAltitude"),
        "blur_variance": None,
        "brightness": None,
        "percent_dark": None,
        "percent_bright": None,
        "exposure_score": None,
        "exposure_warning": False,
        "blurry": False,
    }

    if cv2 is None or np is None:
        raise ImportError(
            "OpenCV and NumPy are required for blur and exposure scoring. Install with 'pip install opencv-python numpy'."
        )

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to read image {image_path}")

    lap = cv2.Laplacian(image, cv2.CV_64F)
    variance = float(np.var(lap))
    metadata["blur_variance"] = variance

    brightness = float(np.mean(image) / 255.0)
    metadata["brightness"] = brightness
    metadata["blurry"] = variance < BLUR_VARIANCE_THRESHOLD and brightness > 0.08

    percent_dark = float(np.mean(image < 30) * 100.0)
    percent_bright = float(np.mean(image > 225) * 100.0)
    metadata["percent_dark"] = percent_dark
    metadata["percent_bright"] = percent_bright

    exposure_penalty = 0.0
    if brightness < SEVERE_UNDEREXPOSED and percent_dark > SEVERE_DARK_PERCENT:
        exposure_penalty += 70.0
    if brightness > SEVERE_OVEREXPOSED and percent_bright > SEVERE_BRIGHT_PERCENT:
        exposure_penalty += 70.0
    exposure_penalty += max(0.0, percent_dark - SEVERE_DARK_PERCENT) * 0.4
    exposure_penalty += max(0.0, percent_bright - SEVERE_BRIGHT_PERCENT) * 0.4

    score = max(0.0, 100.0 - exposure_penalty)
    metadata["exposure_score"] = score
    metadata["exposure_warning"] = (
        brightness < SEVERE_UNDEREXPOSED and percent_dark > SEVERE_DARK_PERCENT
        or brightness > SEVERE_OVEREXPOSED and percent_bright > SEVERE_BRIGHT_PERCENT
    )

    return metadata


# Calculate horizontal distance between two GPS coordinates in meters.
def haversine(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return 6371000.0 * c


# Normalize altitude metadata values into floating point meters.
def parse_altitude(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(str(value).replace(" m", ""))
        except Exception:
            return None


def load_job_metadata(folder_path):
    if not os.path.isdir(folder_path):
        return None
    candidates = [
        os.path.join(folder_path, "job_metadata.json"),
        os.path.join(folder_path, "metadata.json"),
        os.path.join(folder_path, "site_metadata.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                continue
    return None


def parse_apriltag_stats(review_metadata, folder_path=None):
    if not isinstance(review_metadata, dict):
        review_metadata = {}

    apriltag_payload = None
    if "apriltagStats" in review_metadata:
        apriltag_payload = review_metadata["apriltagStats"]
    elif "job_metadata" in review_metadata and isinstance(review_metadata["job_metadata"], dict):
        apriltag_payload = review_metadata["job_metadata"].get("apriltagStats")

    if apriltag_payload is None and folder_path:
        job_metadata = load_job_metadata(folder_path)
        if isinstance(job_metadata, dict):
            apriltag_payload = job_metadata.get("apriltagStats")

    if isinstance(apriltag_payload, str):
        try:
            apriltag_payload = json.loads(apriltag_payload)
        except Exception:
            apriltag_payload = None

    if not isinstance(apriltag_payload, dict):
        return {
            "apriltag_payload_used": False,
            "apriltag_detected": False,
            "apriltag_count": 0,
            "apriltag_detection_count": 0,
            "apriltag_image_count": 0,
            "apriltag_detection_rate_per_image": 0.0,
            "gcp_detected": "unknown",
            "scalepoint_detected": "unknown",
            "gcp_quality": "UNKNOWN",
            "scalepoint_quality": "UNKNOWN",
            "weak_tag_ids": [],
            "strong_tag_ids": [],
            "weak_tag_count": 0,
            "strong_tag_count": 0,
        }

    tags = apriltag_payload.get("tags") or []
    if isinstance(tags, dict):
        tags = [tags]

    def tag_flag(tag, keys):
        for key in keys:
            value = tag.get(key)
            if isinstance(value, str):
                value = value.lower()
                if value in {"true", "yes", "1"}:
                    return True
            if value in (True, 1, "1", "yes", "true"):
                return True
        return False

    tag_count = int(apriltag_payload.get("tagCount") or len(tags) or 0)
    detection_count = int(apriltag_payload.get("detectionCount") or 0)
    image_count = int(apriltag_payload.get("imageCount") or apriltag_payload.get("numberOfFiles") or 0)
    apriltag_detected = tag_count > 0 and detection_count > 0
    detection_rate_per_image = round(detection_count / image_count, 2) if image_count > 0 else 0.0

    weak_tag_ids = [int(tag.get("tagId", -1)) for tag in tags if int(tag.get("detectionCount") or 0) < 5]
    strong_tag_ids = [int(tag.get("tagId", -1)) for tag in tags if int(tag.get("detectionCount") or 0) >= 10]
    gcp_detected = "yes" if any(tag_flag(tag, ["gcp", "isGCP", "is_gcp", "gcpPoint", "hasGcp"]) for tag in tags) else "no"
    scalepoint_detected = "yes" if any(tag_flag(tag, ["scale", "isScale", "is_scale", "scalePoint", "scale_point", "hasScale"]) for tag in tags) else "no"

    def quality_label(detected, count, rate):
        if detected != "yes":
            return "MISSING" if apriltag_payload is not None else "UNKNOWN"
        if count >= 3 and rate >= 1.0:
            return "GOOD"
        if count >= 2 and rate >= 0.5:
            return "CAUTION"
        return "WEAK"

    gcp_quality = quality_label(gcp_detected, tag_count, detection_rate_per_image)
    scalepoint_quality = quality_label(scalepoint_detected, tag_count, detection_rate_per_image)

    return {
        "apriltag_payload_used": True,
        "apriltag_detected": apriltag_detected,
        "apriltag_count": tag_count,
        "apriltag_detection_count": detection_count,
        "apriltag_image_count": image_count,
        "apriltag_detection_rate_per_image": detection_rate_per_image,
        "gcp_detected": gcp_detected,
        "scalepoint_detected": scalepoint_detected,
        "gcp_quality": gcp_quality,
        "scalepoint_quality": scalepoint_quality,
        "weak_tag_ids": [tid for tid in weak_tag_ids if tid >= 0],
        "strong_tag_ids": [tid for tid in strong_tag_ids if tid >= 0],
        "weak_tag_count": len([tid for tid in weak_tag_ids if tid >= 0]),
        "strong_tag_count": len([tid for tid in strong_tag_ids if tid >= 0]),
    }


def parse_gcp_csv(file_path):
    """
    Parse GCP CSV file and return count of GCP points.
    Expected format: rows with GCP data (headers optional).
    Returns: dict with gcp_point_count and gcp_csv_used flag.
    """
    if not file_path or not os.path.isfile(file_path):
        return {"gcp_point_count": 0, "gcp_csv_used": False}
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return {"gcp_point_count": 0, "gcp_csv_used": False}
            rows = list(reader)
            return {
                "gcp_point_count": len(rows),
                "gcp_csv_used": True,
                "gcp_csv_path": file_path,
            }
    except Exception:
        return {"gcp_point_count": 0, "gcp_csv_used": False}


def parse_ground_truth_csv(file_path):
    """
    Parse ground-truth CSV file for validation comparison.
    Expected columns: job_id, outcome, quality_score, reconstruction_success, notes.
    Returns: dict with ground truth data and used flag.
    """
    if not file_path or not os.path.isfile(file_path):
        return {"ground_truth_used": False, "ground_truth_outcome": None}
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return {"ground_truth_used": False, "ground_truth_outcome": None}
            rows = list(reader)
            if not rows:
                return {"ground_truth_used": False, "ground_truth_outcome": None}
            first_row = rows[0]
            return {
                "ground_truth_used": True,
                "ground_truth_outcome": first_row.get("outcome", first_row.get("reconstruction_success")),
                "ground_truth_quality_score": first_row.get("quality_score"),
                "ground_truth_notes": first_row.get("notes"),
                "ground_truth_path": file_path,
            }
    except Exception:
        return {"ground_truth_used": False, "ground_truth_outcome": None}


def generate_evidence_advice(evidence_summary, continuity, blurry_count):
    advice = []
    if not evidence_summary.get("apriltag_payload_used"):
        advice.append("No SiteSee apriltagStats metadata was available; GCP/scale evidence status is unknown.")
    else:
        if evidence_summary.get("gcp_detected") == "no":
            advice.append("No GCP tags were detected in apriltagStats metadata.")
        elif evidence_summary.get("gcp_quality") == "WEAK":
            advice.append("GCP evidence is weak; review tag visibility and coverage.")
        if evidence_summary.get("scalepoint_detected") == "no":
            advice.append("No scale point tags were detected in apriltagStats metadata.")
        elif evidence_summary.get("scalepoint_quality") == "WEAK":
            advice.append("Scale evidence appears weak; review scale point usage.")
        if evidence_summary.get("apriltag_detected") and evidence_summary.get("apriltag_count", 0) < 3:
            advice.append("Too few AprilTags were detected for strong evidence.")

    if continuity and continuity.get("mission_interruption"):
        advice.append("Mission continuity appears interrupted; review timestamp gaps.")
    if blurry_count > 0:
        advice.append("Blurry images were detected; review image sharpness and capture speed.")
    if not advice:
        return "Capture evidence appears consistent for review."
    return " ".join(advice)


def compute_final_decision(risk_label, failure_severity, confidence_score, gcp_evidence_quality, apriltag_detected, continuity):
    if failure_severity == "CRITICAL" or risk_label == "HIGH RISK":
        return "REVIEW"
    if failure_severity in ("HIGH", "MEDIUM"):
        return "REVIEW"
    if gcp_evidence_quality in ("WEAK", "MISSING"):
        return "REVIEW"
    if continuity.get("mission_interruption") and confidence_score < 90:
        return "REVIEW"
    if not apriltag_detected and continuity.get("long_gaps"):
        return "REVIEW"
    return "PASS"


def decision_badge_state(final_decision):
    if final_decision == "PASS":
        return "OK"
    if final_decision == "REVIEW":
        return "NEEDS REVIEW"
    return "CRITICAL"


def format_altitude_drift_severity(drift_m):
    if drift_m is None:
        return "UNKNOWN"
    if drift_m > 10.0:
        return "CRITICAL"
    if drift_m > 5.0:
        return "HIGH"
    if drift_m > 2.0:
        return "MODERATE"
    return "LOW"


def pilot_recommendation(failure_reason):
    mapping = {
        "BATTERY_ALTITUDE_DRIFT": "Stabilize GPS/altitude after the battery swap before continuing the mission.",
        "MOTION_BLUR": "Recapture the affected orbit/section with slower speed and stable focus.",
        "EXPOSURE_ISSUE": "Recapture with correct manual exposure settings.",
        "MISSING_GCP_OR_SCALE": "Provide visible GCP/scale reference images before approving the capture.",
        "GROUND_CAPTURED_ROOFTOP": "Recapture from the rooftop if safe access is available.",
        "INTERRUPTED_CAPTURE": "Review mission continuity and confirm the flight path before re-flying.",
        "NO_FAILURE": "No immediate corrective action required unless review metadata changes.",
        "MIXED_FAILURE": "Address all identified issues before returning to site for recapture.",
    }
    return mapping.get(
        failure_reason,
        "Review captured evidence and make a site-specific recommendation before returning."
    )


# Compute an overall risk score using calibrated thresholds and issue counts.
def summary_risk_score(metrics, continuity=None, gps_score=None, true_altitude_drift_m=0.0):
    count = len(metrics)
    if count == 0:
        return 0.0

    max_z = max((item.get("z_jump", 0.0) or 0.0 for item in metrics), default=0.0)
    max_xy = max((item.get("xy_jump", 0.0) or 0.0 for item in metrics), default=0.0)
    blurry_count = sum(1 for item in metrics if item.get("blurry"))
    exposure_warnings = sum(1 for item in metrics if item.get("exposure_warning"))
    battery_swap_count = sum(1 for item in metrics if item.get("battery_swap"))

    blur_ratio = blurry_count / max(1, count)

    score = 0.0
    if max_z > 4.0:
        score += 35.0
    elif max_z > 2.0:
        score += 20.0

    if max_xy > GPS_WARNING_DISTANCE:
        score += min((max_xy - GPS_WARNING_DISTANCE) / 70.0, 1.0) * 20.0

    if blur_ratio >= 0.50:
        score += 60.0
    elif blur_ratio >= 0.25:
        score += 35.0
    elif blur_ratio > 0.0:
        score += 15.0

    score += min(exposure_warnings / max(1, count), 0.4) * 25.0
    score += min(battery_swap_count / max(1, count), 0.4) * 20.0
    if true_altitude_drift_m and true_altitude_drift_m > 2.0:
        score += min(true_altitude_drift_m / 4.0, 1.0) * 15.0

    if continuity and continuity.get("mission_interruption"):
        score += 20.0
    if gps_score is not None:
        score += min(max(0.0, 100.0 - gps_score) * 0.15, 15.0)

    return min(100.0, score)


# Convert a numeric risk score into a PASS / CAUTION / HIGH RISK label.
def format_risk_label(score, blur_ratio=0.0):
    if blur_ratio >= 0.50:
        return "HIGH RISK"
    if blur_ratio >= 0.25:
        return "REVIEW RECOMMENDED"
    if score < 35.0:
        return "LOW RISK"
    if score < 65.0:
        return "REVIEW RECOMMENDED"
    return "HIGH RISK"


def classify_failure(blurry_images, exposure_warnings, true_altitude_drift_m, continuity, timestamp_gap_count, risk_score):
    motion_blur = blurry_images > 2
    exposure_issue = exposure_warnings > 5
    altitude_drift = true_altitude_drift_m > 5.0
    interrupted_capture = continuity.get("mission_interruption", False) and timestamp_gap_count > 3
    issue_count = sum((motion_blur, exposure_issue, altitude_drift, interrupted_capture))

    if issue_count > 1:
        failure_reason = "MIXED_FAILURE"
    elif motion_blur:
        failure_reason = "MOTION_BLUR"
    elif exposure_issue:
        failure_reason = "EXPOSURE_ISSUE"
    elif altitude_drift:
        failure_reason = "BATTERY_ALTITUDE_DRIFT"
    elif interrupted_capture:
        failure_reason = "INTERRUPTED_CAPTURE"
    else:
        failure_reason = "NO_FAILURE"

    if failure_reason == "NO_FAILURE":
        severity = "LOW"
    elif failure_reason == "MIXED_FAILURE":
        severity = "CRITICAL"
    elif failure_reason == "BATTERY_ALTITUDE_DRIFT":
        severity = "CRITICAL" if true_altitude_drift_m > 10.0 else "HIGH"
    elif failure_reason == "EXPOSURE_ISSUE":
        severity = "HIGH" if exposure_warnings > 10 else "MEDIUM"
    elif failure_reason == "MOTION_BLUR":
        severity = "HIGH" if blurry_images > 5 else "MEDIUM"
    elif failure_reason == "INTERRUPTED_CAPTURE":
        severity = "HIGH" if timestamp_gap_count > 6 else "MEDIUM"
    else:
        severity = "LOW"

    if failure_reason == "NO_FAILURE":
        confidence = 100
    else:
        confidence = min(100, max(50, int(40 + risk_score * 0.35 + issue_count * 10)))

    return failure_reason, confidence, severity


def write_csv_reports(metrics, summary):
    """Write summary and per-image detail reports to CSV files."""
    detail_fields = [
        "file",
        "create_date",
        "gps_latitude",
        "gps_longitude",
        "gps_altitude",
        "relative_altitude",
        "absolute_altitude",
        "altitude_delta",
        "mission_segment_id",
        "z_jump",
        "xy_jump",
        "true_battery_drift_m",
        "true_altitude_drift_m",
        "true_altitude_drift_severity",
        "true_altitude_drift_delta_before",
        "true_altitude_drift_delta_after",
        "battery_drift_boundary_before",
        "battery_drift_boundary_after",
        "battery_drift_boundary_timestamp_before",
        "battery_drift_boundary_timestamp_after",
        "battery_drift_boundary_gap_minutes",
        "battery_drift_boundary_xy_distance_m",
        "battery_drift_boundary_z_difference_m",
        "mission_segment_change",
        "blur_variance",
        "brightness",
        "percent_dark",
        "percent_bright",
        "exposure_score",
        "exposure_warning",
        "blurry",
        "timestamp_gap",
        "battery_swap",
        "battery_drift_risk",
        "marker_detected_count",
        "marker_detected_ids",
        "marker_avg_marker_size_px",
        "marker_placement_score",
        "marker_advice",
        "failure_reason",
        "confidence_score",
        "failure_severity",
        "final_decision",
        "decision_badge",
        "reviewer_notes",
    ]

    with open("capture_details.csv", "w", newline="", encoding="utf-8") as detail_csv:
        writer = csv.DictWriter(detail_csv, fieldnames=detail_fields)
        writer.writeheader()
        for item in metrics:
            row = {field: item.get(field, "") for field in detail_fields}
            for dt_field in [
                "create_date",
                "battery_drift_boundary_timestamp_before",
                "battery_drift_boundary_timestamp_after",
            ]:
                if row.get(dt_field):
                    row[dt_field] = row[dt_field].isoformat(sep=" ")
            writer.writerow(row)

    with open("capture_summary.csv", "w", newline="", encoding="utf-8") as summary_csv:
        writer = csv.DictWriter(summary_csv, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def choose_training_label(item):
    """Choose a per-image label for ML dataset export."""
    issues = []
    if item.get("blurry"):
        issues.append("BLURRY")
    if item.get("exposure_warning"):
        issues.append("EXPOSURE")
    if item.get("battery_drift_risk"):
        issues.append("BATTERY_DRIFT")

    if len(issues) == 0:
        return "GOOD"
    if len(issues) == 1:
        return issues[0]
    return "MIXED_FAILURE"


def write_training_dataset(metrics):
    """Write a training dataset CSV for machine learning."""
    fields = [
        "filename",
        "blur_score",
        "exposure_score",
        "z_jump",
        "xy_jump",
        "mission_segment_id",
        "altitude_delta",
        "true_altitude_drift_m",
        "true_altitude_drift_severity",
        "battery_swap_flag",
        "overall_capture_label",
    ]
    with open(OUTPUT_DATASET_CSV, "w", newline="", encoding="utf-8") as dataset_csv:
        writer = csv.DictWriter(dataset_csv, fieldnames=fields)
        writer.writeheader()
        for item in metrics:
            writer.writerow({
                "filename": item.get("file", ""),
                "blur_score": item.get("blur_variance", ""),
                "exposure_score": item.get("exposure_score", ""),
                "z_jump": item.get("z_jump", ""),
                "xy_jump": item.get("xy_jump", ""),
                "mission_segment_id": item.get("mission_segment_id", ""),
                "altitude_delta": item.get("altitude_delta", ""),
                "true_altitude_drift_m": item.get("true_altitude_drift_m", ""),
                "true_altitude_drift_severity": item.get("true_altitude_drift_severity", ""),
                "battery_swap_flag": bool(item.get("battery_swap", False)),
                "overall_capture_label": choose_training_label(item),
            })


def collect_folder_images(folder_path):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")

    images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith((".jpg", ".jpeg")):
            path = os.path.join(folder_path, file_name)
            try:
                metrics = get_image_metrics(path)
            except Exception as exc:
                print(f"Skipping {file_name}: {exc}")
                continue
            images.append(metrics)

    images.sort(key=lambda x: (x["create_date"] or datetime.datetime.min, x["file"]))
    return images


def run_capture_analysis(folder_path, export_csv=True, export_html=True, export_dataset=False, report_path=None, review_metadata=None, gcp_csv_path=None, ground_truth_csv_path=None):
    images = collect_folder_images(folder_path)
    metrics, summary_data, continuity, gps_stability_score = compute_capture_summary(images, folder_path, review_metadata=review_metadata, gcp_csv_path=gcp_csv_path, ground_truth_csv_path=ground_truth_csv_path)

    if export_csv:
        write_csv_reports(metrics, summary_data)
    if export_dataset:
        write_training_dataset(metrics)
    if export_html:
        main_report_path = report_path or OUTPUT_HTML
        write_html_report(metrics, summary_data, continuity, gps_stability_score, folder_path, main_report_path)
        if summary_data["risk_label"] in ("REVIEW RECOMMENDED", "HIGH RISK"):
            failed_report_path = CAPTURE_RISK_ASSESSMENT_HTML if report_path is None else f"{os.path.splitext(report_path)[0]}_capture_risk_assessment.html"
            write_html_report(
                metrics,
                summary_data,
                continuity,
                gps_stability_score,
                folder_path,
                failed_report_path,
                html_title="Capture Risk Assessment",
                heading="Capture Risk Assessment",
            )

    return metrics, summary_data, continuity, gps_stability_score


def compute_capture_summary(images, folder_path, review_metadata=None, gcp_csv_path=None, ground_truth_csv_path=None):
    review_metadata = review_metadata or {}
    
    # Parse optional CSV data
    gcp_data = parse_gcp_csv(gcp_csv_path) if gcp_csv_path else {"gcp_point_count": 0, "gcp_csv_used": False}
    ground_truth_data = parse_ground_truth_csv(ground_truth_csv_path) if ground_truth_csv_path else {"ground_truth_used": False, "ground_truth_outcome": None}
    
    # Merge GCP and ground-truth data into review metadata
    if gcp_data.get("gcp_csv_used"):
        review_metadata["gcp_point_count"] = gcp_data.get("gcp_point_count", 0)
    if ground_truth_data.get("ground_truth_used"):
        review_metadata["ground_truth_outcome"] = ground_truth_data.get("ground_truth_outcome")
        review_metadata["ground_truth_quality_score"] = ground_truth_data.get("ground_truth_quality_score")
        review_metadata["ground_truth_notes"] = ground_truth_data.get("ground_truth_notes")
    
    previous = None
    segment_id = 1
    metrics = []
    for image in images:
        image["relative_altitude"] = parse_altitude(image["relative_altitude"])
        image["absolute_altitude"] = parse_altitude(image["absolute_altitude"])
        image["gps_altitude"] = parse_altitude(image["gps_altitude"])

        image["altitude_delta"] = None
        if image["relative_altitude"] is not None and image["absolute_altitude"] is not None:
            image["altitude_delta"] = image["absolute_altitude"] - image["relative_altitude"]

        current_alt = image["relative_altitude"]
        if current_alt is None:
            current_alt = image["absolute_altitude"]
        if current_alt is None:
            current_alt = image["gps_altitude"]

        image["altitude_used"] = current_alt
        image["z_jump"] = None
        image["xy_jump"] = None
        image["timestamp_gap"] = False
        image["battery_swap"] = False
        image["battery_drift_risk"] = False
        image["mission_segment_change"] = False
        image["mission_segment_id"] = segment_id
        image["true_battery_drift_m"] = None
        image["true_altitude_drift_m"] = None
        image["true_altitude_drift_severity"] = "UNKNOWN"
        image["battery_drift_boundary_before"] = None
        image["battery_drift_boundary_after"] = None
        image["battery_drift_boundary_timestamp_before"] = None
        image["battery_drift_boundary_timestamp_after"] = None
        image["battery_drift_boundary_gap_minutes"] = None
        image["battery_drift_boundary_xy_distance_m"] = None
        image["battery_drift_boundary_z_difference_m"] = None
        image["true_altitude_drift_delta_before"] = None
        image["true_altitude_drift_delta_after"] = None

        if previous is not None:
            if current_alt is not None and previous["altitude_used"] is not None:
                image["z_jump"] = abs(current_alt - previous["altitude_used"])

            if (
                image["gps_latitude"] is not None
                and image["gps_longitude"] is not None
                and previous["gps_latitude"] is not None
                and previous["gps_longitude"] is not None
            ):
                image["xy_jump"] = haversine(
                    previous["gps_latitude"],
                    previous["gps_longitude"],
                    image["gps_latitude"],
                    image["gps_longitude"],
                )

            if previous["create_date"] is not None and image["create_date"] is not None:
                gap = (image["create_date"] - previous["create_date"]).total_seconds()
                if gap > TIMESTAMP_GAP_SECONDS:
                    image["timestamp_gap"] = True
                    same_area = image["xy_jump"] is not None and image["xy_jump"] <= 10.0
                    altitude_swap_candidate = same_area and image["xy_jump"] <= 5.0
                    if altitude_swap_candidate:
                        segment_id += 1
                        image["mission_segment_id"] = segment_id
                        image["battery_swap"] = True
                        image["battery_drift_boundary_before"] = previous["file"]
                        image["battery_drift_boundary_after"] = image["file"]
                        image["battery_drift_boundary_timestamp_before"] = previous["create_date"]
                        image["battery_drift_boundary_timestamp_after"] = image["create_date"]
                        image["battery_drift_boundary_gap_minutes"] = round(gap / 60.0, 2)
                        image["battery_drift_boundary_xy_distance_m"] = round(image["xy_jump"], 2) if image["xy_jump"] is not None else None
                        image["battery_drift_boundary_z_difference_m"] = round(image["z_jump"], 2) if image["z_jump"] is not None else None
                        image["true_altitude_drift_delta_before"] = previous.get("altitude_delta")
                        image["true_altitude_drift_delta_after"] = image.get("altitude_delta")
                        if image["true_altitude_drift_delta_before"] is not None and image["true_altitude_drift_delta_after"] is not None:
                            drift_value = abs(image["true_altitude_drift_delta_after"] - image["true_altitude_drift_delta_before"])
                            image["true_battery_drift_m"] = drift_value
                            image["true_altitude_drift_m"] = drift_value
                            image["true_altitude_drift_severity"] = format_altitude_drift_severity(drift_value)
                            if drift_value > 2.0:
                                image["battery_drift_risk"] = True
                        else:
                            image["true_altitude_drift_severity"] = "UNKNOWN"
                    else:
                        image["mission_segment_change"] = True
                        segment_id += 1
                        image["mission_segment_id"] = segment_id

        metrics.append(image)
        previous = image

    apriltag_summary = parse_apriltag_stats(review_metadata, folder_path=folder_path)
    apriltag_summary = apriltag_summary or {}

    evidence_advice = ""
    for item in metrics:
        item["marker_detected_count"] = 0
        item["marker_detected_ids"] = ""
        item["marker_avg_marker_size_px"] = None
        item["marker_placement_score"] = 0
        item["marker_advice"] = ""

    analyzed = len(metrics)
    max_z_jump = max((item["z_jump"] or 0.0 for item in metrics), default=0.0)
    max_xy_jump = max((item["xy_jump"] or 0.0 for item in metrics), default=0.0)
    altitude_values = [item["altitude_used"] for item in metrics if item.get("altitude_used") is not None]
    max_global_altitude_variation = max(altitude_values) - min(altitude_values) if altitude_values else 0.0
    true_altitude_drift = max((item.get("true_altitude_drift_m") or 0.0 for item in metrics), default=0.0)
    blurry_count = sum(1 for item in metrics if item.get("blurry"))
    exposure_warnings = sum(1 for item in metrics if item.get("exposure_warning"))
    gap_count = sum(1 for item in metrics if item.get("timestamp_gap"))
    battery_swap_count = sum(1 for item in metrics if item.get("battery_swap"))
    battery_drift_count = sum(1 for item in metrics if item.get("battery_drift_risk"))

    continuity = calculate_mission_continuity(metrics)
    gps_stability_score = compute_gps_stability_score(metrics, continuity)
    blur_ratio = blurry_count / max(1, analyzed)
    risk = summary_risk_score(
        metrics,
        continuity=continuity,
        gps_score=gps_stability_score,
        true_altitude_drift_m=true_altitude_drift,
    )
    risk_label = format_risk_label(risk, blur_ratio)

    evidence_advice = generate_evidence_advice(apriltag_summary, continuity, blurry_count)
    for item in metrics:
        item["marker_advice"] = evidence_advice

    mission_segment_changes = sum(1 for item in metrics if item.get("mission_segment_change"))
    failure_reason, confidence_score, failure_severity = classify_failure(
        blurry_count,
        exposure_warnings,
        true_altitude_drift,
        continuity,
        gap_count,
        risk,
    )

    gcp_evidence_status = review_metadata.get("gcp_evidence_status", "unknown")
    if apriltag_summary.get("apriltag_payload_used"):
        gcp_evidence_status = "yes" if apriltag_summary.get("gcp_detected") == "yes" else "no"
    gcp_image_count = review_metadata.get("gcp_image_count", 0)
    scale_reference_status = review_metadata.get("scale_reference_status", "unknown")
    if scale_reference_status == "unknown" and apriltag_summary.get("scalepoint_detected") in ("yes", "no"):
        scale_reference_status = apriltag_summary.get("scalepoint_detected")
    capture_position = review_metadata.get("capture_position", "unknown")
    roof_access_available = review_metadata.get("roof_access_available", "unknown")
    reviewer_notes = review_metadata.get("reviewer_notes", "")

    if gcp_evidence_status == "no" or scale_reference_status == "no":
        failure_reason = "MISSING_GCP_OR_SCALE"
        failure_severity = "HIGH" if gcp_evidence_status == "no" and scale_reference_status == "no" else "MEDIUM"
        confidence_score = min(100, max(60, int(50 + risk * 0.35)))
    elif capture_position == "ground" and roof_access_available == "yes":
        failure_reason = "GROUND_CAPTURED_ROOFTOP"
        failure_severity = "HIGH"
        confidence_score = min(100, max(60, int(50 + risk * 0.25)))

    final_decision = compute_final_decision(
        risk_label,
        failure_severity,
        confidence_score,
        apriltag_summary.get("gcp_quality", "UNKNOWN"),
        apriltag_summary.get("apriltag_detected", False),
        continuity,
    )
    decision_badge = decision_badge_state(final_decision)

    drift_entries = [item for item in metrics if item.get("battery_swap") and item.get("battery_drift_boundary_before")]
    first_drift = drift_entries[0] if drift_entries else {}

    for item in metrics:
        item["failure_reason"] = failure_reason
        item["confidence_score"] = confidence_score
        item["failure_severity"] = failure_severity
        item["final_decision"] = final_decision
        item["decision_badge"] = decision_badge

    quality_scores = {
        "GOOD": 80,
        "CAUTION": 55,
        "WEAK": 25,
        "MISSING": 0,
        "UNKNOWN": 50,
    }
    gcp_evidence_score = quality_scores.get(apriltag_summary.get("gcp_quality", "UNKNOWN"), 50)
    scalepoint_evidence_score = quality_scores.get(apriltag_summary.get("scalepoint_quality", "UNKNOWN"), 50)

    summary_data = {
        "images_analyzed": analyzed,
        "max_z_jump_m": f"{max_z_jump:.2f}",
        "max_xy_jump_m": f"{max_xy_jump:.2f}",
        "max_global_altitude_variation_m": f"{max_global_altitude_variation:.2f}",
        "true_altitude_drift_m": f"{true_altitude_drift:.2f}",
        "true_battery_swap_drift_m": f"{true_altitude_drift:.2f}",
        "true_altitude_drift_severity": format_altitude_drift_severity(true_altitude_drift) if true_altitude_drift > 0 else "LOW",
        "true_altitude_drift_delta_before": first_drift.get("true_altitude_drift_delta_before"),
        "true_altitude_drift_delta_after": first_drift.get("true_altitude_drift_delta_after"),
        "true_altitude_drift_gap_minutes": first_drift.get("battery_drift_boundary_gap_minutes"),
        "true_altitude_drift_xy_distance_m": first_drift.get("battery_drift_boundary_xy_distance_m"),
        "mission_segment_changes": mission_segment_changes,
        "blurry_images": blurry_count,
        "exposure_warnings": exposure_warnings,
        "suspected_battery_swaps": battery_swap_count,
        "high_risk_battery_drift": battery_drift_count,
        "timestamp_gap_events": gap_count,
        "long_gap_count": len(continuity.get("long_gaps", [])),
        "max_timestamp_gap_s": max((gap["seconds"] for gap in continuity.get("long_gaps", [])), default=0),
        "mean_interval_s": round(continuity.get("mean_interval_s", 0.0), 1) if continuity.get("mean_interval_s") is not None else None,
        "std_interval_s": round(continuity.get("std_interval_s", 0.0), 1) if continuity.get("std_interval_s") is not None else None,
        "gps_stability_score": gps_stability_score,
        "mission_interruption": continuity["mission_interruption"],
        "final_risk_score": f"{risk:.0f}",
        "risk_label": risk_label,
        "failure_reason": failure_reason,
        "confidence_score": confidence_score,
        "failure_severity": failure_severity,
        "final_decision": final_decision,
        "decision_badge": decision_badge,
        "pilot_recommendation": pilot_recommendation(failure_reason),
        "gcp_evidence_status": gcp_evidence_status,
        "gcp_detected": apriltag_summary.get("gcp_detected", "unknown"),
        "gcp_image_count": gcp_image_count,
        "scale_reference_status": scale_reference_status,
        "scale_detected": apriltag_summary.get("scalepoint_detected", "unknown"),
        "capture_position": capture_position,
        "roof_access_available": roof_access_available,
        "reviewer_notes": reviewer_notes,
        "gcp_evidence_score": gcp_evidence_score,
        "scalepoint_evidence_score": scalepoint_evidence_score,
        "gcp_quality": apriltag_summary.get("gcp_quality", "UNKNOWN"),
        "gcp_evidence_quality": apriltag_summary.get("gcp_quality", "UNKNOWN"),
        "scalepoint_quality": apriltag_summary.get("scalepoint_quality", "UNKNOWN"),
        "apriltag_count": apriltag_summary.get("apriltag_count", 0),
        "unique_apriltag_count": apriltag_summary.get("apriltag_count", 0),
        "apriltag_detection_count": apriltag_summary.get("apriltag_detection_count", 0),
        "total_apriltag_detections": apriltag_summary.get("apriltag_detection_count", 0),
        "apriltag_image_count": apriltag_summary.get("apriltag_image_count", 0),
        "apriltag_detection_rate_per_image": apriltag_summary.get("apriltag_detection_rate_per_image", 0.0),
        "weak_tag_ids": apriltag_summary.get("weak_tag_ids", []),
        "strong_tag_ids": apriltag_summary.get("strong_tag_ids", []),
        "weak_tag_count": apriltag_summary.get("weak_tag_count", 0),
        "strong_tag_count": apriltag_summary.get("strong_tag_count", 0),
        "apriltag_detected": apriltag_summary.get("apriltag_detected", False),
        "evidence_advice": evidence_advice,
        "gcp_point_count": review_metadata.get("gcp_point_count", 0),
        "gcp_csv_used": gcp_data.get("gcp_csv_used", False),
        "ground_truth_used": ground_truth_data.get("ground_truth_used", False),
        "ground_truth_outcome": ground_truth_data.get("ground_truth_outcome"),
        "ground_truth_quality_score": ground_truth_data.get("ground_truth_quality_score"),
        "ground_truth_notes": ground_truth_data.get("ground_truth_notes"),
    }

    return metrics, summary_data, continuity, gps_stability_score


def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def generate_thumbnail(image_path, output_path, width=300):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Unable to read image for thumbnail: {image_path}")
    height, orig_width = image.shape[:2]
    if orig_width <= 0:
        raise RuntimeError(f"Invalid image width for thumbnail: {image_path}")
    scale = width / float(orig_width)
    new_size = (width, max(1, int(height * scale)))
    thumb = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, thumb)


def calculate_mission_continuity(metrics):
    continuity = {
        "timestamp_intervals": [],
        "long_gaps": [],
        "inconsistent_cadence": False,
        "mission_interruption": False,
        "mean_interval_s": None,
        "std_interval_s": None,
    }

    previous = None
    for item in metrics:
        if previous is not None and item.get("create_date") is not None and previous.get("create_date") is not None:
            interval = (item["create_date"] - previous["create_date"]).total_seconds()
            continuity["timestamp_intervals"].append(interval)
            if interval > TIMESTAMP_GAP_SECONDS:
                continuity["long_gaps"].append({
                    "from": previous["file"],
                    "to": item["file"],
                    "seconds": interval,
                })
        previous = item

    intervals = continuity["timestamp_intervals"]
    if intervals:
        continuity["mean_interval_s"] = sum(intervals) / len(intervals)
        continuity["std_interval_s"] = statistics.pstdev(intervals)
        if continuity["std_interval_s"] > max(1.0, continuity["mean_interval_s"] * 0.6):
            continuity["inconsistent_cadence"] = True

    continuity["mission_interruption"] = bool(continuity["long_gaps"] or continuity["inconsistent_cadence"])
    return continuity


def compute_gps_stability_score(metrics, continuity):
    if not metrics:
        return 0

    max_xy = max((item.get("xy_jump") or 0.0 for item in metrics), default=0.0)
    max_z = max((item.get("z_jump") or 0.0 for item in metrics), default=0.0)
    score = 100.0

    if max_xy > 30.0:
        score -= min((max_xy - 30.0) * 0.35, 35.0)
    if max_z > 2.0:
        score -= min((max_z - 2.0) * 4.0, 30.0)
    if continuity["long_gaps"]:
        score -= min(len(continuity["long_gaps"]) * 12.5, 30.0)
    if continuity["inconsistent_cadence"]:
        score -= 20.0

    return int(max(0.0, min(score, 100.0)))


def write_html_report(metrics, summary, continuity, gps_stability_score, folder_path, report_path="capture_report.html", thumbnails_dir="thumbnails", html_title="Drone Capture Report", heading="Drone Capture Report"):
    ensure_dir(thumbnails_dir)
    flagged_items = []
    all_photos = []
    label_counts = {}
    for item in metrics:
        label = choose_training_label(item)
        label_counts[label] = label_counts.get(label, 0) + 1
        issues = []
        if item.get("blurry"):
            issues.append("BLURRY")
        if item.get("exposure_warning"):
            issues.append("EXPOSURE")
        if item.get("battery_drift_risk"):
            issues.append("DRIFT")

        if issues:
            flagged_items.append({
                "file": item["file"],
                "issues": issues,
                "thumbnail_refs": [],
            })
            for issue in issues:
                thumb_name = f"{issue.lower()}_{sanitize_filename(item['file'])}.jpg"
                thumb_path = os.path.join(thumbnails_dir, thumb_name)
                source_path = os.path.join(folder_path, item["file"])
                try:
                    generate_thumbnail(source_path, thumb_path)
                    flagged_items[-1]["thumbnail_refs"].append(thumb_path)
                except Exception:
                    continue

        photo_thumb = os.path.join(thumbnails_dir, f"photo_{sanitize_filename(item['file'])}.jpg")
        source_path = os.path.join(folder_path, item["file"])
        try:
            if not os.path.exists(photo_thumb):
                generate_thumbnail(source_path, photo_thumb)
            all_photos.append({"file": item["file"], "thumb": photo_thumb})
        except Exception:
            continue

    badge_color = "#21ba45" if summary["risk_label"] == "LOW RISK" else "#f2c037" if summary["risk_label"] == "REVIEW RECOMMENDED" else "#db2828"
    interruption_text = "POSSIBLE MISSION INTERRUPTION" if continuity["mission_interruption"] else "Mission continuity stable"
    gap_text = f"{len(continuity['long_gaps'])} long gaps" if continuity["long_gaps"] else "No long timestamp gaps"
    cadence_text = (
        f"Inconsistent capture cadence ({continuity['std_interval_s']:.1f}s std)" if continuity["inconsistent_cadence"] else "Capture cadence consistent"
    )

    def escape_html(value):
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def build_issue_grid(items):
        html = ""
        for item in items:
            thumb_html = "".join(
                f"<div class='thumb-card'><img src='{escape_html(thumb)}' alt='{escape_html(item['file'])}' /><div class='thumb-label'>{escape_html(issue)}</div></div>" for issue, thumb in zip(item['issues'], item.get('thumbnail_refs', []))
            )
            html += f"<div class='issue-row'><div class='issue-name'>{escape_html(item['file'])}</div>{thumb_html}</div>"
        return html

    def build_related_photo_grid(items):
        if not items:
            return "<p>No related photos available.</p>"
        html = ""
        for item in items[:32]:
            html += (
                f"<div class='thumb-card'><img src='{escape_html(item['thumb'])}' alt='{escape_html(item['file'])}' /><div class='thumb-label'>{escape_html(item['file'])}</div></div>"
            )
        return html

    total_images = summary.get("images_analyzed", 0)
    blurry_pct = int(round((summary.get("blurry_images", 0) / max(1, total_images)) * 100))
    exposure_pct = int(round((summary.get("exposure_warnings", 0) / max(1, total_images)) * 100))
    battery_pct = int(round((summary.get("suspected_battery_swaps", 0) / max(1, total_images)) * 100))
    drift_entries = [item for item in metrics if item.get("battery_swap") and item.get("battery_drift_boundary_before")]

    def build_drift_details(items):
        if not items:
            return "<p>No true altitude drift boundaries detected.</p>"
        html_rows = []
        for item in items:
            drift_value = item.get("true_altitude_drift_m")
            severity = item.get("true_altitude_drift_severity", "UNKNOWN")
            html_rows.append(
                f"<div class='drift-row'>"
                f"<div><strong>Before:</strong> {item['battery_drift_boundary_before']}<br/><strong>After:</strong> {item['battery_drift_boundary_after']}</div>"
                f"<div><strong>Gap:</strong> {item['battery_drift_boundary_gap_minutes']} min</div>"
                f"<div><strong>XY:</strong> {item['battery_drift_boundary_xy_distance_m']} m</div>"
                f"<div><strong>True altitude drift:</strong> {drift_value if drift_value is not None else 'N/A'} m</div>"
                f"<div><strong>Severity:</strong> {severity}</div>"
                f"<div><strong>Before ts:</strong> {item['battery_drift_boundary_timestamp_before'].isoformat(sep=' ')}<br/><strong>After ts:</strong> {item['battery_drift_boundary_timestamp_after'].isoformat(sep=' ')}</div>"
                f"</div>"
            )
        return "".join(html_rows)
    drift_html = build_drift_details(drift_entries)

    def build_evidence_summary_grid(summary_data):
        html = ""
        html += f"<div class='stat-block'><strong>{escape_html(str(summary_data.get('gcp_evidence_status', 'unknown')))}</strong>GCP evidence status</div>"
        html += f"<div class='stat-block'><strong>{escape_html(str(summary_data.get('scale_reference_status', 'unknown')))}</strong>Scale evidence status</div>"
        html += f"<div class='stat-block'><strong>{escape_html(str(summary_data.get('gcp_quality', 'UNKNOWN')))}</strong>GCP evidence quality</div>"
        html += f"<div class='stat-block'><strong>{escape_html(str(summary_data.get('scalepoint_quality', 'UNKNOWN')))}</strong>ScalePoint evidence quality</div>"
        html += f"<div class='stat-block'><strong>{escape_html(str(summary_data.get('apriltag_count', 0)))}</strong>AprilTags detected</div>"
        html += f"<div class='stat-block'><strong>{escape_html(str(summary_data.get('apriltag_detection_count', 0)))}</strong>AprilTag detections</div>"
        html += f"<div class='stat-block'><strong>{escape_html(str(summary_data.get('apriltag_image_count', 0)))}</strong>Images with Apriltag data</div>"
        html += f"<div class='stat-block'><strong>{escape_html(str(summary_data.get('apriltag_detection_rate_per_image', 0.0)))}</strong>Detections per image</div>"
        return html

    marker_evidence_html = build_evidence_summary_grid(summary)

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8' />
<meta name='viewport' content='width=device-width, initial-scale=1.0' />
<title>{escape_html(html_title)}</title>
<style>
body {{ background:#111; color:#eee; font-family:Inter,system-ui,sans-serif; margin:0; padding:0; }}
.container {{ max-width:1200px; margin:0 auto; padding:24px; }}
.header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }}
.card {{ background:#1c1c24; border:1px solid #282832; border-radius:18px; padding:20px; margin-top:20px; }}
.card h2 {{ margin:0 0 12px; font-size:1.3rem; }}
.badge {{ display:inline-flex; align-items:center; gap:8px; padding:10px 14px; border-radius:999px; font-weight:700; color:#111; background:{badge_color}; }}
.stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
.stat-block {{ background:#14141d; border:1px solid #282832; border-radius:16px; padding:16px; }}
.stat-block strong {{ display:block; font-size:1.6rem; margin-bottom:8px; }}
.bar-row {{ display:flex; align-items:center; gap:12px; margin:10px 0; }}
.bar-label {{ width:160px; font-size:0.94rem; color:#c8c8d0; }}
.bar-bg {{ flex:1; background:#272732; border-radius:999px; height:14px; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:999px; background:linear-gradient(90deg,#57b2ff,#8e54ff); }}
.issue-row {{ display:grid; grid-template-columns:1fr; gap:12px; margin-bottom:18px; }}
.thumb-card {{ background:#0f0f16; border:1px solid #282832; border-radius:16px; padding:8px; text-align:center; width:220px; }}
.thumb-card img {{ width:100%; height:auto; border-radius:12px; display:block; }}
.thumb-label {{ margin-top:8px; color:#ddd; font-size:0.85rem; }}
.issue-name {{ color:#fff; font-size:0.98rem; margin-bottom:8px; }}
.issue-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:16px; }}
.section-title {{ color:#8ab4f8; margin-top:0; }}
</style>
</head>
<body>
<div class='container'>
  <div class='header'>
    <div>
      <h1>{escape_html(heading)}</h1>
      <p style='color:#aaa;margin:8px 0 0;'>Analyzed folder: {os.path.abspath(folder_path)}</p>
      <p class='disclaimer' style='color:#bbb;margin:12px 0 0;'>This assessment highlights capture-quality risk signals. It is not a final reconstruction pass/fail decision.</p>
    </div>
    <div class='badge'>{summary['risk_label']} SCORE {summary['final_risk_score']}/100</div>
  </div>

  <div class='card'>
    <h2>Risk Summary</h2>
    <div class='stat-grid'>
      <div class='stat-block'><strong>{summary['images_analyzed']}</strong>Images analyzed</div>
      <div class='stat-block'><strong>{summary['max_z_jump_m']} m</strong>Max altitude jump</div>
      <div class='stat-block'><strong>{summary['max_global_altitude_variation_m']} m</strong>Max global altitude variation</div>
      <div class='stat-block'><strong>{summary['true_altitude_drift_m']} m</strong>True altitude drift</div>
      <div class='stat-block'><strong>{summary['true_altitude_drift_severity']}</strong>Drift severity</div>
      <div class='stat-block'><strong>{summary['max_xy_jump_m']} m</strong>Max horizontal drift</div>
      <div class='stat-block'><strong>{summary['blurry_images']} ({blurry_pct}%)</strong>Blurry images</div>
      <div class='stat-block'><strong>{summary['exposure_warnings']} ({exposure_pct}%)</strong>Exposure alerts</div>
      <div class='stat-block'><strong>{summary['suspected_battery_swaps']}</strong>Battery swap events</div>
      <div class='stat-block'><strong>{summary['mission_segment_changes']}</strong>Mission segment changes</div>
      <div class='stat-block'><strong>{summary['failure_reason']}</strong>Primary risk factor</div>
      <div class='stat-block'><strong>{summary['failure_severity']}</strong>Risk severity</div>
      <div class='stat-block'><strong>{summary['confidence_score']}/100</strong>Confidence score</div>
      <div class='stat-block'><strong>{gps_stability_score}</strong>GPS stability score</div>
      <div class='stat-block'><strong>{summary['pilot_recommendation']}</strong>Pilot recommendation</div>
      <div class='stat-block'><strong>{summary['gcp_evidence_status']}</strong>GCP evidence</div>
      <div class='stat-block'><strong>{summary['gcp_image_count']}</strong>GCP image count</div>
      <div class='stat-block'><strong>{summary['scale_reference_status']}</strong>Scale reference</div>
      <div class='stat-block'><strong>{summary['capture_position']}</strong>Capture position</div>
      <div class='stat-block'><strong>{summary['roof_access_available']}</strong>Roof access</div>
      <div class='stat-block'><strong>{summary['gcp_evidence_quality']}</strong>GCP evidence quality</div>
      <div class='stat-block'><strong>{summary['scalepoint_quality']}</strong>ScalePoint evidence quality</div>
      <div class='stat-block'><strong>{summary['gcp_evidence_score']}</strong>GCP evidence score</div>
      <div class='stat-block'><strong>{summary['scalepoint_evidence_score']}</strong>ScalePoint evidence score</div>
      <div class='stat-block'><strong>{summary['apriltag_count']}</strong>AprilTags detected</div>
      <div class='stat-block'><strong>{summary['apriltag_detection_count']}</strong>AprilTag detections</div>
      <div class='stat-block'><strong>{summary['apriltag_image_count']}</strong>Images with Apriltag data</div>
      <div class='stat-block'><strong>{summary['apriltag_detection_rate_per_image']}</strong>Detections per image</div>
      <div class='stat-block'><strong>{"YES" if continuity['mission_interruption'] else "NO"}</strong>Mission interruption</div>
    </div>
  </div>

  <div class='card'>
    <h2>GPS / Altitude Stability</h2>
    <p><strong>GPS stability score:</strong> {gps_stability_score}/100</p>
    <p><strong>Max altitude jump:</strong> {summary['max_z_jump_m']} m</p>
    <p><strong>Max horizontal drift:</strong> {summary['max_xy_jump_m']} m</p>
    <div class='bar-row'><span class='bar-label'>Timestamp consistency</span><div class='bar-bg'><div class='bar-fill' style='width:{max(0,min(100,100 - (continuity['std_interval_s'] or 0) * 2))}%;'></div></div></div>
  </div>

  <div class='card'>
    <h2>Mission Continuity</h2>
    <p>{interruption_text}</p>
    <p>{gap_text} · {cadence_text}</p>
  </div>

  <div class='card'>
    <h2>Image Quality</h2>
    <div class='bar-row'><span class='bar-label'>Blurry</span><div class='bar-bg'><div class='bar-fill' style='width:{blurry_pct}%;'></div></div></div>
    <div class='bar-row'><span class='bar-label'>Exposure</span><div class='bar-bg'><div class='bar-fill' style='width:{exposure_pct}%;'></div></div></div>
    <div class='bar-row'><span class='bar-label'>Battery swap</span><div class='bar-bg'><div class='bar-fill' style='width:{battery_pct}%;'></div></div></div>
  </div>

  <div class='card'>
    <h2>GCP / Scale / AprilTag Evidence</h2>
    <p><strong>Evidence guidance:</strong> {escape_html(summary.get('evidence_advice', summary.get('MARKER_ADVICE', 'No guidance available.')))}</p>
    <div class='stat-grid'>
      {marker_evidence_html}
    </div>
    <div class='bar-row'><span class='bar-label'>Evidence status</span><span>{escape_html(summary.get('gcp_evidence_status', 'unknown'))} / {escape_html(summary.get('scale_reference_status', 'unknown'))}</span></div>
  </div>

  <div class='card'>
    <h2>True Altitude Drift Diagnostics</h2>
    {drift_html}
  </div>

  <div class='card'>
    <h2>Risk Factors</h2>
    <div class='stat-grid'>
      {''.join(f"<div class='stat-block'><strong>{label_counts[label]}</strong>{label}</div>" for label in sorted(label_counts))}
    </div>
  </div>

  <div class='card'>
    <h2>Visual Evidence</h2>
    <div class='issue-grid'>
      {build_issue_grid(flagged_items)}
    </div>
  </div>

  <div class='card'>
    <h2>Recommended Pilot Actions</h2>
    <p>{escape_html(summary['pilot_recommendation'])}</p>
    <p><strong>Evidence guidance:</strong> {escape_html(summary.get('evidence_advice', summary.get('MARKER_ADVICE', 'No guidance available.')))}</p>
  </div>

  <div class='card'>
    <h2>Reviewer Notes</h2>
    <p>{escape_html(summary['reviewer_notes'] or 'No reviewer notes provided.')}</p>
  </div>
</div>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as html_file:
        html_file.write(html)


# Print optional raw per-image metrics to the terminal.
def print_raw_output(metrics):
    header = [
        "File",
        "Timestamp",
        "GPS Lat",
        "GPS Lon",
        "RelAlt",
        "AbsAlt",
        "AltDelta",
        "SegID",
        "GPSAlt",
        "ZJump",
        "XYJump",
        "TrueDrift",
        "DriftSev",
        "BlurVar",
        "Brightness",
        "Dark%",
        "Bright%",
        "ExposureWarn",
    ]
    print("\nRaw image metrics:")
    print("	".join(header))
    for item in metrics:
        values = [
            item.get("file", ""),
            item.get("create_date").isoformat(sep=" ") if item.get("create_date") else "",
            f"{item.get('gps_latitude', '')}" if item.get("gps_latitude") is not None else "",
            f"{item.get('gps_longitude', '')}" if item.get("gps_longitude") is not None else "",
            f"{item.get('relative_altitude', '')}" if item.get("relative_altitude") is not None else "",
            f"{item.get('absolute_altitude', '')}" if item.get("absolute_altitude") is not None else "",
            f"{item.get('altitude_delta', '')}" if item.get("altitude_delta") is not None else "",
            f"{item.get('mission_segment_id', '')}",
            f"{item.get('gps_altitude', '')}" if item.get("gps_altitude") is not None else "",
            f"{item.get('z_jump', 0.0):.2f}" if item.get("z_jump") is not None else "",
            f"{item.get('xy_jump', 0.0):.2f}" if item.get("xy_jump") is not None else "",
            f"{item.get('true_altitude_drift_m', '')}" if item.get("true_altitude_drift_m") is not None else "",
            item.get("true_altitude_drift_severity", ""),
            f"{item.get('blur_variance', 0.0):.1f}" if item.get("blur_variance") is not None else "",
            f"{item.get('brightness', 0.0):.3f}" if item.get("brightness") is not None else "",
            f"{item.get('percent_dark', 0.0):.1f}" if item.get("percent_dark") is not None else "",
            f"{item.get('percent_bright', 0.0):.1f}" if item.get("percent_bright") is not None else "",
            str(item.get("exposure_warning", "")),
        ]
        print("	".join(values))


# Main analyzer entry point: process images, score issues, print summary, and write CSV reports.
def main():
    parser = argparse.ArgumentParser(description="Analyze drone capture images and metadata.")
    parser.add_argument("folder", nargs="?", default=IMAGE_FOLDER, help="Path to the images folder")
    parser.add_argument("--raw", action="store_true", help="Print raw per-image output")
    parser.add_argument("--dataset", action="store_true", help="Export training_dataset.csv for ML")
    args = parser.parse_args()

    folder_path = args.folder
    if not os.path.isdir(folder_path):
        print(f"Image folder not found: {folder_path}")
        sys.exit(1)

    print(f"\nAnalyzing folder: {os.path.abspath(folder_path)}")
    metrics, summary_data, continuity, gps_stability_score = run_capture_analysis(
        folder_path,
        export_csv=True,
        export_html=True,
        export_dataset=args.dataset,
        report_path=OUTPUT_HTML,
    )

    blurry_images = [item["file"] for item in metrics if item.get("blurry")]
    exposure_warning_images = [item["file"] for item in metrics if item.get("exposure_warning")]
    altitude_jump_between = [
        f"{metrics[i-1]['file']} -> {item['file']}"
        for i, item in enumerate(metrics)
        if i > 0 and item.get("z_jump") is not None and item["z_jump"] > 2.0
    ]
    battery_swap_images = [item["file"] for item in metrics if item.get("battery_swap")]
    battery_drift_images = [item["file"] for item in metrics if item.get("battery_drift_risk")]

    print("\nCapture analysis summary")
    print("------------------------")
    print(f"Images analysed:        {summary_data['images_analyzed']}")
    print(f"Max Z altitude jump:    {summary_data['max_z_jump_m']} m")
    print(f"Max XY distance jump:   {summary_data['max_xy_jump_m']} m")
    print(f"Blurry images:          {summary_data['blurry_images']}")
    if blurry_images:
        print(f"Blurry image files:     {', '.join(blurry_images)}")
    print(f"Exposure warnings:      {summary_data['exposure_warnings']}")
    if exposure_warning_images:
        print(f"Exposure warning files: {', '.join(exposure_warning_images)}")
    print(f"GCP evidence quality:   {summary_data.get('gcp_evidence_quality', 'N/A')}")
    print(f"Unique AprilTags:       {summary_data.get('unique_apriltag_count', 0)}")
    print(f"Total AprilTag detections: {summary_data.get('total_apriltag_detections', 0)}")
    print(f"AprilTag image count:   {summary_data.get('apriltag_image_count', 0)}")
    print(f"AprilTag detections/image: {summary_data.get('apriltag_detection_rate_per_image', 0.0)}")
    print(f"Strong tag IDs:         {', '.join(str(x) for x in summary_data.get('strong_tag_ids', [])) or 'None'}")
    print(f"Weak tag IDs:           {', '.join(str(x) for x in summary_data.get('weak_tag_ids', [])) or 'None'}")
    print(f"Suspected battery swaps: {summary_data['suspected_battery_swaps']}")
    if battery_swap_images:
        print(f"Battery swap images:     {', '.join(battery_swap_images)}")
    print(f"Mission segment changes: {summary_data['mission_segment_changes']}")
    print(f"High-risk battery drift: {summary_data['high_risk_battery_drift']}")
    if battery_drift_images:
        print(f"Battery drift images:    {', '.join(battery_drift_images)}")
    print(f"Max global altitude variation: {summary_data['max_global_altitude_variation_m']} m")
    print(f"True altitude drift: {summary_data['true_altitude_drift_m']} m")
    print(f"Drift severity:       {summary_data['true_altitude_drift_severity']}")
    print(f"Risk reason:           {summary_data['failure_reason']}")
    print(f"Risk severity:         {summary_data['failure_severity']}")
    print(f"Confidence score:      {summary_data['confidence_score']}/100")
    if summary_data['suspected_battery_swaps']:
        print("True altitude drift diagnostics:")
        for item in metrics:
            if item.get("battery_swap") and item.get("battery_drift_boundary_before"):
                print(f"  - before_image: {item['battery_drift_boundary_before']}")
                print(f"    after_image: {item['battery_drift_boundary_after']}")
                print(f"    timestamp_before: {item['battery_drift_boundary_timestamp_before'].isoformat(sep=' ')}")
                print(f"    timestamp_after: {item['battery_drift_boundary_timestamp_after'].isoformat(sep=' ')}")
                print(f"    timestamp_gap_minutes: {item['battery_drift_boundary_gap_minutes']:.2f}")
                print(f"    xy_distance_m: {item['battery_drift_boundary_xy_distance_m']}")
                print(f"    z_difference_m: {item['battery_drift_boundary_z_difference_m']}")
                print(f"    true_altitude_drift_m: {item.get('true_altitude_drift_m')}")
                print(f"    drift_severity: {item.get('true_altitude_drift_severity')}")
    print(f"Altitude jumps:         {len(altitude_jump_between)}")
    if altitude_jump_between:
        print(f"Jump pairs:             {', '.join(altitude_jump_between)}")
    print(f"Final risk score:       {summary_data['final_risk_score']}/100")
    print(f"Overall assessment:     {summary_data['risk_label']}")
    print(f"GPS stability score:    {summary_data['gps_stability_score']}/100")
    print(f"Mission interruption:   {'YES' if continuity['mission_interruption'] else 'NO'}")
    if continuity['long_gaps']:
        print(f"Long timestamp gaps:    {len(continuity['long_gaps'])}")
    if continuity['inconsistent_cadence']:
        print(f"Capture cadence issue:  inconsistent timing (std {continuity['std_interval_s']:.1f}s)")

    print(f"HTML dashboard report:  {OUTPUT_HTML}")

    if args.raw:
        print_raw_output(metrics)


if __name__ == "__main__":
    main()
