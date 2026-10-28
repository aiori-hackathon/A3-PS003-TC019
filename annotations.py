
import csv
import cv2
import torch
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_DIR / "BMD45"

OUTPUT_DIR = PROJECT_DIR / "outputs"

LABEL_DIR = OUTPUT_DIR / "labels"

ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"

CSV_FILE = OUTPUT_DIR / "vehicle_counts.csv"

SUMMARY_FILE = OUTPUT_DIR / "annotation_summary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. CONFIGURATION
# =====================================================

FOLDERS = [
    "images_000",
    "images_001",
    "images_002",
    "images_003",
    "images_004"
]

MODEL_NAME = str(PROJECT_DIR / "yolo26n.pt")

IMAGE_SIZE = 640

CONFIDENCE = 0.25

IOU_THRESHOLD = 0.7

MAX_DETECTIONS = 300

DEVICE = "cpu"

SAVE_ANNOTATED_IMAGES = True

# None = all images
# 20 = test only 20 images
MAX_IMAGES = None

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".webp", ".tif", ".tiff"
}

CSV_COLUMNS = [
    "image_path",
    "total_vehicles",
    "car",
    "motorcycle",
    "bus",
    "truck"
]


# =====================================================
# 3. FIND ALL IMAGES
# =====================================================

print("\n==========================================")
print("       BMD-45 VEHICLE ANNOTATION")
print("==========================================")

all_images = []

for folder_name in FOLDERS:

    folder_path = DATASET_ROOT / folder_name

    if not folder_path.exists():

        raise FileNotFoundError(
            f"Missing dataset folder: {folder_path}"
        )

    folder_images = sorted([
        p for p in folder_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print(
        f"{folder_name}: {len(folder_images)} images"
    )

    all_images.extend([
        (folder_name, image)
        for image in folder_images
    ])

print("------------------------------------------")
print("TOTAL IMAGES:", len(all_images))

if not all_images:
    raise RuntimeError("No images found!")

if MAX_IMAGES is not None:
    all_images = all_images[:MAX_IMAGES]

print("Images selected:", len(all_images))


# =====================================================
# 4. LOAD COMPLETED IMAGE RECORDS
# =====================================================

processed_images = set()

if CSV_FILE.exists() and CSV_FILE.stat().st_size > 0:

    try:

        previous_df = pd.read_csv(CSV_FILE)

        if "image_path" in previous_df.columns:

            for image_key in previous_df["image_path"].dropna():

                key = str(image_key).replace("\\", "/")

                label_file = (
                    LABEL_DIR / Path(key)
                ).with_suffix(".txt")

                if label_file.exists():
                    processed_images.add(key)

        print(
            "\nPreviously completed:",
            len(processed_images)
        )

    except Exception as error:

        print("Previous progress could not be loaded:", error)


# =====================================================
# 5. REMOVE ALREADY COMPLETED IMAGES
# =====================================================

remaining_images = []

for folder_name, image_path in all_images:

    relative_path = image_path.relative_to(
        DATASET_ROOT / folder_name
    )

    image_key = (
        Path(folder_name) / relative_path
    ).as_posix()

    if image_key not in processed_images:

        remaining_images.append(
            (image_key, image_path)
        )

print("Images remaining:", len(remaining_images))


# =====================================================
# 6. LOAD YOLO MODEL
# =====================================================

print("\nLoading YOLO model...")

if not Path(MODEL_NAME).exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_NAME}"
    )

model = YOLO(MODEL_NAME)

print("YOLO model loaded successfully!")

print("Running on CPU:", not torch.cuda.is_available())


# =====================================================
# 7. AUTOMATIC ANNOTATION
# =====================================================

write_header = (
    not CSV_FILE.exists()
    or CSV_FILE.stat().st_size == 0
)

print("\nStarting annotation...\n")

failed_images = []

with open(
    CSV_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=CSV_COLUMNS
    )

    if write_header:
        writer.writeheader()

    for image_key, image_path in tqdm(
        remaining_images,
        desc="Annotating BMD-45",
        unit="image"
    ):

        try:

            # -----------------------------------------
            # YOLO DETECTION
            # -----------------------------------------

            result = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                classes=list(VEHICLE_CLASSES.keys()),
                max_det=MAX_DETECTIONS,
                device=DEVICE,
                verbose=False
            )[0]

            counts = {
                name: 0
                for name in VEHICLE_CLASSES.values()
            }

            annotation_lines = []

            # -----------------------------------------
            # EXTRACT DETECTIONS
            # -----------------------------------------

            if result.boxes is not None:

                boxes = result.boxes

                coordinates = boxes.xywhn.cpu().numpy()

                class_ids = boxes.cls.cpu().numpy().astype(int)

                confidences = boxes.conf.cpu().numpy()

                for box, class_id, confidence in zip(
                    coordinates,
                    class_ids,
                    confidences
                ):

                    class_name = VEHICLE_CLASSES.get(
                        class_id
                    )

                    if class_name is None:
                        continue

                    counts[class_name] += 1

                    x, y, w, h = box

                    annotation_lines.append(
                        f"{class_id} "
                        f"{x:.6f} "
                        f"{y:.6f} "
                        f"{w:.6f} "
                        f"{h:.6f} "
                        f"{confidence:.6f}"
                    )

            # -----------------------------------------
            # SAVE YOLO LABELS
            # -----------------------------------------

            label_file = (
                LABEL_DIR / Path(image_key)
            ).with_suffix(".txt")

            label_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            label_file.write_text(
                "\n".join(annotation_lines),
                encoding="utf-8"
            )

            # -----------------------------------------
            # SAVE ANNOTATED IMAGES
            # -----------------------------------------

            if SAVE_ANNOTATED_IMAGES:

                annotated_image = result.plot()

                output_image = (
                    ANNOTATED_DIR / Path(image_key)
                )

                output_image.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                success = cv2.imwrite(
                    str(output_image),
                    annotated_image
                )

                if not success:
                    raise IOError(
                        f"Image save failed: {image_path}"
                    )

            # -----------------------------------------
            # SAVE VEHICLE COUNTS
            # -----------------------------------------

            writer.writerow({

                "image_path": image_key,

                "total_vehicles": sum(
                    counts.values()
                ),

                **counts

            })

            csvfile.flush()

        except Exception as error:

            print(
                f"\nERROR: {image_path.name}: {error}"
            )

            failed_images.append(image_key)


# =====================================================
# 8. GENERATE COMBINED SUMMARY
# =====================================================

print("\nGenerating combined summary...")

df = pd.read_csv(CSV_FILE)

summary = {

    "Total processed images": len(df),

    "Images with vehicles": int(
        (df["total_vehicles"] > 0).sum()
    ),

    "Images without vehicles": int(
        (df["total_vehicles"] == 0).sum()
    ),

    "Total detected vehicles": int(
        df["total_vehicles"].sum()
    ),

    "Total cars": int(df["car"].sum()),

    "Total motorcycles": int(
        df["motorcycle"].sum()
    ),

    "Total buses": int(df["bus"].sum()),

    "Total trucks": int(df["truck"].sum()),

    "Failed images in this run": len(failed_images)

}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# =====================================================
# 9. SAVE FAILED IMAGE LIST
# =====================================================

if failed_images:

    pd.DataFrame({
        "image_path": failed_images
    }).to_csv(
        OUTPUT_DIR / "failed_images.csv",
        index=False
    )


# =====================================================
# 10. FINAL REPORT
# =====================================================

print("\n==========================================")
print("          ANNOTATION COMPLETED")
print("==========================================")

print(summary_df.to_string(index=False))

print("\nOutput directory:", OUTPUT_DIR)

print("Labels:", LABEL_DIR)

print("Annotated images:", ANNOTATED_DIR)

print("Counts CSV:", CSV_FILE)

print("Summary CSV:", SUMMARY_FILE)

print("\n==========================================")
