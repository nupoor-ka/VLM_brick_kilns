# Bench-marking Vision Language Models on Detecting Brick Kilns in Satellite Imagery

This project explores the use of **Florence-2 Vision-Language Models (VLMs)** to detect **brick kilns** in **low-resolution satellite imagery**, addressing a challenging geospatial vision task.

---

## Overview

Brick kilns are small, spatially clustered structures, often located in remote or rural areas. Their detection in satellite imagery is difficult due to:

- Low image resolution
- Sparse annotations
- Small object scale

This project evaluates the performance of **four Florence-2 models**:

- `Florence2-base`
- `Florence2-base_finetuned`
- `Florence2-large`
- `Florence2-large_finetuned`

We experimented with:

- Zero-shot inference with prompt tuning  
- Super-resolution via **SwinIR**  
- Tiled inference (custom + SAHI library)  
- Fine-tuning (custom + Maestro library)

---

## Methods & Results

### 1. Zero-Shot Inference

- Used Florence-2 with task-specific prompts.
- Results limited by image quality and domain mismatch.

**F1 Score @ IoU 0.5**: `0.06`

---

### 2. Super-Resolution

- Applied **SwinIR** to enhance low-resolution inputs.

**F1 Score**: `0.07`

---

### 3. Tiled Inference

- Manual patching and slicing via the [SAHI](https://github.com/obss/sahi) library.
- Combined with super-resolution to improve small-object detection.

**F1 Score**: `0.0146`

---

### 4. Fine-Tuning (Maestro)

- Fine-tuned `Florence2-large` on kiln image dataset.
- Trained for **300–700 epochs**.
- Used varying numbers of labeled kiln images.
- Achieved **up to 0.314 F1 score**, a **32-point improvement** over zero-shot.

#### F1 Score vs Number of Kiln Images

| Kiln Images | F1 Score |
|-------------|----------|
| 1           | 0.034    |
| 2           | 0.076    |
| 3           | 0.139    |
| 4           | 0.109    |
| 5           | 0.166    |
| 7           | 0.137    |
| 10          | 0.182    |
| 20          | 0.172    |
| 30          | 0.255    |
| 40          | 0.269    |
| 60          | 0.248    |
| 100         | 0.326    |
| 200         | 0.314    |
| 297         | 0.306    |

---

## Repository Structure

├── finetuning_florence2.py # Fine-tuning Florence-2 model with Maestro

├── florence2_on_swinir_sr.py # Florence-2 inference with SwinIR super-resolution

├── florence2_slicing.py # Tiled inference using custom slicing and SAHI

├── florence2_zero_shot.py # Zero-shot inference with prompt tuning

└── README.md # Project documentation


The work was conducted under [Prof. Nipun Batra](https://nipunbatra.github.io/) at IIT Gandhinagar, in collaboration with Shataxi Dubey, Rishabh Mondal, and Devansh Lodha.
