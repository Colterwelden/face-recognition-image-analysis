"""Find a reference face in group photographs and save annotated results.

This application uses pretrained face encoders; it does not train a model.
"""

from pathlib import Path

import face_recognition
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE = PROJECT_DIR / "known_person.jpg"
GROUP_DIR = PROJECT_DIR / "group_images"
OUTPUT_DIR = PROJECT_DIR / "output_images"
REPORT_FILE = PROJECT_DIR / "recognition_results.txt"

MATCH_TOLERANCE = 0.60
NUMBER_OF_JITTERS = 5
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def detect_faces(image):
    """Return face locations found by the HOG detector."""
    return face_recognition.face_locations(
        image, number_of_times_to_upsample=1, model="hog"
    )


def encode_face(image, location):
    """Encode a single detected face, returning None on encoding failure."""
    encodings = face_recognition.face_encodings(
        image,
        known_face_locations=[location],
        num_jitters=NUMBER_OF_JITTERS,
        model="small",
    )
    return encodings[0] if encodings else None


def load_reference_encoding():
    """Require exactly one detectable, encodable reference face."""
    if not REFERENCE_IMAGE.is_file():
        raise FileNotFoundError(
            f"Reference image not found: {REFERENCE_IMAGE}. "
            "Add a photograph named known_person.jpg to the project folder."
        )

    image = face_recognition.load_image_file(REFERENCE_IMAGE)
    locations = detect_faces(image)
    if len(locations) != 1:
        raise ValueError(
            "The reference image must contain exactly one detected face; "
            f"found {len(locations)}."
        )

    encoding = encode_face(image, locations[0])
    if encoding is None:
        raise ValueError("The detected reference face could not be encoded.")
    return encoding


def draw_label(drawing, box, label, color, image_width):
    """Draw a face boundary and a label within the image area."""
    top, right, bottom, left = box
    drawing.rectangle([(left, top), (right, bottom)], outline=color, width=4)

    text_bounds = drawing.textbbox((0, 0), label)
    text_width = text_bounds[2] - text_bounds[0]
    text_height = text_bounds[3] - text_bounds[1]
    label_left = max(0, min(left, image_width - text_width - 10))
    label_top = max(0, top - text_height - 10)
    drawing.rectangle(
        [(label_left, label_top), (label_left + text_width + 10, label_top + text_height + 6)],
        fill=color,
    )
    drawing.text((label_left + 5, label_top + 2), label, fill="white")


def analyze_group_image(image_path, reference_encoding):
    """Annotate all detected faces and summarize matches in one photograph."""
    image = face_recognition.load_image_file(image_path)
    locations = detect_faces(image)
    annotated = Image.fromarray(image).convert("RGB")
    drawing = ImageDraw.Draw(annotated)

    distances = []
    matches = 0
    encoding_failures = 0
    for number, location in enumerate(locations, start=1):
        encoding = encode_face(image, location)
        if encoding is None:
            encoding_failures += 1
            draw_label(drawing, location, f"Face {number}: not encoded", "#8b6508", annotated.width)
            continue

        distance = float(face_recognition.face_distance([reference_encoding], encoding)[0])
        is_match = distance <= MATCH_TOLERANCE
        distances.append(distance)
        if is_match:
            matches += 1
            label, color = f"MATCH {distance:.3f}", "green"
        else:
            label, color = f"Other face {number}: {distance:.3f}", "red"
        draw_label(drawing, location, label, color, annotated.width)

    output_path = OUTPUT_DIR / f"recognized_{image_path.stem}.png"
    annotated.save(output_path)
    return {
        "image": image_path.name,
        "faces_detected": len(locations),
        "faces_encoded": len(distances),
        "encoding_failures": encoding_failures,
        "matches": matches,
        "closest_distance": min(distances) if distances else None,
        "output": output_path.name,
    }


def format_result(result):
    """Format a report entry without treating non-detection as proof of absence."""
    distance = result["closest_distance"]
    distance_text = f"{distance:.4f}" if distance is not None else "N/A"
    if result["matches"]:
        status = "MATCH DETECTED"
    elif not result["faces_detected"]:
        status = "NO FACES DETECTED"
    elif not result["faces_encoded"]:
        status = "NO FACES ENCODED"
    else:
        status = "NO MATCH DETECTED"

    return "\n".join(
        [
            f"Image: {result['image']}",
            f"Faces detected: {result['faces_detected']}",
            f"Faces encoded: {result['faces_encoded']}",
            f"Encoding failures: {result['encoding_failures']}",
            f"Reference matches: {result['matches']}",
            f"Closest face distance: {distance_text}",
            f"Result: {status}",
            f"Annotated image: output_images/{result['output']}",
        ]
    )


def main():
    """Process supported images and write a human-readable report."""
    if not GROUP_DIR.is_dir():
        raise FileNotFoundError(
            f"Group image folder not found: {GROUP_DIR}. Create group_images/ "
            "and add JPG or PNG photographs."
        )

    group_images = sorted(
        path for path in GROUP_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not group_images:
        raise FileNotFoundError("No JPG or PNG photographs found in group_images/.")

    reference_encoding = load_reference_encoding()
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = [analyze_group_image(path, reference_encoding) for path in group_images]

    header = [
        "FACIAL RECOGNITION RESULTS",
        "=" * 28,
        f"Reference image: {REFERENCE_IMAGE.name}",
        f"Comparison tolerance: {MATCH_TOLERANCE:.2f}",
        f"Encoding jitters: {NUMBER_OF_JITTERS}",
        f"Group images analyzed: {len(results)}",
        "Note: 'No match detected' does not establish that a person is absent.",
    ]
    report = "\n\n".join(["\n".join(header)] + [format_result(r) for r in results]) + "\n"
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
