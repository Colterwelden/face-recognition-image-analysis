# Face Recognition Image Analysis

**Python | Computer Vision | Facial Recognition | Image Processing**

## Project Overview

A Python-based computer vision application designed to identify a reference individual within group photographs using pretrained facial recognition models.

The application detects faces, generates facial embeddings, calculates similarity distances, and identifies potential matches against a supplied reference photograph. It produces annotated images and a text report summarizing the results.

Developed as part of the CSC526 academic portfolio, this project demonstrates practical implementation of computer vision techniques, integration of pretrained machine learning models, image processing, and automated reporting.

## Technologies Used

- **Python** — Application logic and file processing
- **face_recognition** — Facial detection and embedding generation
- **dlib** — Underlying pretrained facial recognition models
- **Pillow (PIL)** — Image annotation and output generation
- **HOG (Histogram of Oriented Gradients)** — Face detection method
- **Euclidean distance** — Comparison of facial embeddings

The application uses a pretrained model.

## How It Works

1. **Reference image validation:** Loads `known_person.jpg` and verifies that exactly one detectable, encodable face is present.
2. **Face detection:** Uses a HOG-based detector to locate faces within each group photograph.
3. **Facial encoding:** Generates a numerical representation of each detected face using pretrained facial embedding models.
4. **Identity comparison:** Calculates Euclidean distances between detected facial embeddings and the reference encoding.
5. **Match classification:** Classifies distances at or below the configured threshold of `0.60` as matches.
6. **Image annotation:** Draws labeled bounding boxes around detected faces, distinguishing matches, nonmatches, and encoding failures.
7. **Results reporting:** Generates a text report containing detection counts, encoding outcomes, matching distances, and image-level classifications.

Lower embedding distances indicate greater facial similarity. These values are not confidence percentages or calibrated probabilities.

## Observed Results

The original demonstration evaluated five group photographs containing the reference individual.

The reference individual was correctly identified in all five photographs, based on manual verification.

One additional face was not detected in one photograph. 

### Setup

Clone the repository:

```bash
git clone https://github.com/Colterwelden/face-recognition-image-analysis.git
cd face-recognition-image-analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Place a reference photograph named `known_person.jpg` in the project directory.

Create a folder named `group_images` containing the photographs to analyze.

The expected directory structure is:

```text
face-recognition-image-analysis/
├── facial_recognition.py
├── README.md
├── requirements.txt
├── .gitignore
├── known_person.jpg
└── group_images/
    ├── photo_01.jpg
    ├── photo_02.jpg
    └── photo_03.png
```

The reference photograph must contain exactly one detectable face.

Supported group image formats are JPG, JPEG, and PNG.

Run the application:

```bash
python facial_recognition.py
```

## Generated Outputs

The application automatically creates two types of output.

**1. Annotated photographs**

Saved to the `output_images/` directory.

Each detected face receives a colored bounding box:

- Green: Reference face match detected
- Red: Detected face does not meet the matching threshold
- Yellow/brown: Detected face could not be encoded

Faces that are not detected cannot be annotated.

**2. Recognition report**

Saved as `recognition_results.txt`.

The report includes:

- Input photograph filename
- Number of faces detected
- Number of faces successfully encoded
- Number of encoding failures
- Number of reference matches
- Closest facial embedding distance
- Image-level recognition result

The application distinguishes four outcomes:

| Result | Meaning |
|---|---|
| `MATCH DETECTED` | At least one detected face met the matching threshold |
| `NO MATCH DETECTED` | Faces were encoded, but none met the threshold |
| `NO FACES DETECTED` | The detector did not locate any faces |
| `NO FACES ENCODED` | Faces were detected, but none could be encoded |

A result of `NO MATCH DETECTED` does not establish that the reference individual is absent.

## Known Limitations

**Face detection sensitivity:** The HOG detector may fail to locate faces under challenging lighting, unusual viewing angles, small face sizes, or partial occlusion.

**Recognition threshold:** The default matching tolerance of `0.60` has not been calibrated against an independently labeled evaluation dataset.

**Limited validation:** Testing included five target-present photographs. False-match rates, false-nonmatch rates, and generalizable detection performance have not been established.

**Single reference identity:** The application searches for one reference individual at a time.

**Encoding failures:** Detected faces may occasionally fail to produce valid encodings, preventing identity comparison.

**Output filename collisions:** Group images sharing the same filename stem, such as `photo.jpg` and `photo.png`, generate the same output filename and may overwrite one another.

## Attribution

Developed as an academic computer vision application using established open-source libraries and pretrained models.

- [face_recognition — Adam Geitgey](https://github.com/ageitgey/face_recognition)
- [dlib](http://dlib.net/)
- [Pillow](https://pillow.readthedocs.io/)

**Project repository:** https://github.com/Colterwelden/face-recognition-image-analysis