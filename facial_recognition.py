"""
Picks out a known person from a group photograph and saves the results.
"""

from pathlib import Path

import face_recognition
from PIL import Image, ImageDraw


PROGRAM_FOLDER = Path(__file__).resolve().parent
REFERENCE_IMAGE = PROGRAM_FOLDER / "known_person.jpg"
GROUP_FOLDER = PROGRAM_FOLDER / "group_images"
OUTPUT_FOLDER = PROGRAM_FOLDER / "output_images"
RESULTS_FILE = PROGRAM_FOLDER / "recognition_results.txt"

# The face_recognition library uses 0.60
MATCH_TOLERANCE = 0.60

NUMBER_OF_JITTERS = 5

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_faces(image):
    """Return the detected face locations and encodes the faces."""
    locations = face_recognition.face_locations(
        image,
        number_of_times_to_upsample=1,
        model="hog",
    )
    encodings = face_recognition.face_encodings(
        image,
        known_face_locations=locations,
        num_jitters=NUMBER_OF_JITTERS,
        model="small",
    )
    return locations, encodings


def load_reference_encoding():
    """Load the known person and one detectable face."""
    if not REFERENCE_IMAGE.exists():
        raise FileNotFoundError(
            f"The reference image was not found: {REFERENCE_IMAGE.name}"
        )

    reference_image = face_recognition.load_image_file(REFERENCE_IMAGE)
    locations, encodings = find_faces(reference_image)

    if len(encodings) == 0:
        raise ValueError("No face was detected in the reference image.")
    if len(encodings) > 1:
        raise ValueError(
            "The reference image must contain exactly one detectable face."
        )

    return encodings[0]


def draw_label(draw, box, text, color):
    """Draw a face box and a readable label immediately above it."""
    top, right, bottom, left = box
    draw.rectangle([(left, top), (right, bottom)], outline=color, width=5)

    text_box = draw.textbbox((left, top), text)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_top = max(0, top - text_height - 10)

    draw.rectangle(
        [(left, label_top), (left + text_width + 10, top)],
        fill=color,
    )
    draw.text((left + 5, label_top + 3), text, fill="white")


def analyze_group_image(image_path, reference_encoding):
    """Compare every detected face in one group image with the reference."""
    image = face_recognition.load_image_file(image_path)
    locations, encodings = find_faces(image)

    distances = face_recognition.face_distance(encodings, reference_encoding)
    matches = face_recognition.compare_faces(
        encodings,
        reference_encoding,
        tolerance=MATCH_TOLERANCE,
    )

    annotated_image = Image.fromarray(image)
    drawing = ImageDraw.Draw(annotated_image)

    for face_number, (location, distance, is_match) in enumerate(
        zip(locations, distances, matches),
        start=1,
    ):
        if is_match:
            label = f"MATCH {distance:.3f}"
            color = "green"
        else:
            label = f"Unknown {face_number}: {distance:.3f}"
            color = "red"

        draw_label(drawing, location, label, color)

    del drawing

    output_path = OUTPUT_FOLDER / f"recognized_{image_path.name}"
    annotated_image.save(output_path, quality=95)

    closest_distance = min(distances) if len(distances) > 0 else None
    person_is_present = any(matches)

    return {
        "image": image_path.name,
        "faces": len(locations),
        "closest_distance": closest_distance,
        "present": person_is_present,
        "output": output_path.name,
    }


def main():
    """Analyze all group images and write both visual and text results."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    group_images = sorted(
        path
        for path in GROUP_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not group_images:
        raise FileNotFoundError("No group photographs were found.")

    reference_encoding = load_reference_encoding()

    print(f"Reference image: {REFERENCE_IMAGE.name}")
    print(f"Comparison tolerance: {MATCH_TOLERANCE:.2f}")
    print(f"Group images found: {len(group_images)}\n")

    results = []
    for image_path in group_images:
        result = analyze_group_image(image_path, reference_encoding)
        results.append(result)

        distance_text = (
            f"{result['closest_distance']:.4f}"
            if result["closest_distance"] is not None
            else "N/A"
        )
        answer = "PRESENT" if result["present"] else "NOT PRESENT"

        print(f"Image: {result['image']}")
        print(f"Faces detected: {result['faces']}")
        print(f"Closest face distance: {distance_text}")
        print(f"Result: Individual is {answer}")
        print(f"Saved: output_images/{result['output']}\n")

    report_lines = [
        "FACIAL RECOGNITION RESULTS",
        "=" * 28,
        f"Reference image: {REFERENCE_IMAGE.name}",
        f"Comparison tolerance: {MATCH_TOLERANCE:.2f}",
        f"Encoding jitters: {NUMBER_OF_JITTERS}",
        "",
    ]

    for result in results:
        distance_text = (
            f"{result['closest_distance']:.4f}"
            if result["closest_distance"] is not None
            else "N/A"
        )
        answer = "PRESENT" if result["present"] else "NOT PRESENT"
        report_lines.extend(
            [
                f"Image: {result['image']}",
                f"Faces detected: {result['faces']}",
                f"Closest face distance: {distance_text}",
                f"Result: Individual is {answer}",
                "",
            ]
        )

    RESULTS_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Complete results saved as {RESULTS_FILE.name}.")


if __name__ == "__main__":
    main()
