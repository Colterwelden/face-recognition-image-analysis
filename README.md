# Face Recognition Image Analysis

An academic computer-vision project by Colter Welden, completed
for the CSC526 midterm portfolio.

## Objective

Compare a reference face against faces detected in group photographs,
then generate annotated images and a text report.

## Implementation

The program uses pretrained models through the face_recognition
library and Pillow for image annotation. It does not train a new model.

The workflow:

1. Validate that the reference photograph contains exactly one
   detectable face.
2. Detect faces in each group photograph using the HOG detector.
3. Generate face encodings and compare them with the reference.
4. Apply a matching tolerance of 0.60.
5. Save annotated photographs and a text results report.

Face distances indicate similarity between encodings.
They are not confidence percentages.

## Observed results

The project was demonstrated on five group photographs containing
the reference person.

The project author manually verified that the intended person was
correctly matched in all five photographs. Face detection was
incomplete in at least one photograph.

These observations describe a small demonstration, not a general
accuracy benchmark. Photographs without the reference person were
not evaluated.

## Requirements

- Python
- face_recognition
- Pillow

Tested package versions and Python version will be recorded before
the repository setup is finalized.

## Input files

Place the following beside facial_recognition.py:

- known_person.jpg: a reference photograph containing exactly one
  detectable face
- group_images/: a folder containing JPG, JPEG, or PNG photographs

Personal photographs are excluded from this repository.
Users must supply their own input images.

## Running the program

Install the dependencies listed in requirements.txt, then run:

    python facial_recognition.py

## Outputs

The program creates:

- output_images/: photographs with bounding boxes and match labels
- recognition_results.txt: detected face counts, closest distances,
  and image-level match decisions

## Limitations

- The demonstration uses one reference person and five photographs.
- No target-absent evaluation was performed.
- Missed detections prevent those faces from being compared.
- The matching threshold was not calibrated on a separate dataset.
- The current "NOT PRESENT" message means no matching face was
  detected; it does not establish that the person is absent.
- Image quality, face angle, lighting, and occlusion are potential
  factors to investigate in further testing.

## Planned improvements

- Distinguish "No faces detected" from "No matching face detected."
- Add target-absent photographs and record false matches.
- Evaluate face detection separately from identity matching.
- Add CSV output for systematic evaluation.
- Test installation in a fresh Python environment.

## Attribution

Academic implementation using the face_recognition library and Pillow.

- face_recognition: https://github.com/ageitgey/face_recognition
- Pillow: https://python-pillow.org/