# Human Action Recognition in Real-Time Video Stream

A real-time human action recognition pipeline focused on sport activities using skeleton-based graph neural networks.

## Project Idea

This project develops a complete pipeline for detecting and classifying human actions in real-time video streams. The system is specifically designed for sport activities and emphasizes the balance between accuracy and inference efficiency on consumer-grade hardware.

### Key Features
- **Real-time processing**: Optimized for consumer-grade GPUs (e.g., Apple M-series chips)
- **Skeleton-based approach**: Uses 2D pose estimation followed by graph-based action classification
- **Sport-focused**: Specialized for athletic movements and exercises
- **Single-person detection**: Designed for scenarios with one person in the frame

### Pipeline Overview
1. **Video Input**: Real-time video stream or recorded video
2. **Pose Estimation**: Extract 2D skeleton keypoints from each frame
3. **Graph Construction**: Convert skeleton sequences into spatial-temporal graphs
4. **Action Classification**: Classify actions using graph neural networks
5. **Output**: Real-time action predictions with confidence scores

## Models Used

1. Pretrained [YOLOv11](https://arxiv.org/pdf/2410.17725v1) for pose estimation
2. Finetuned [InfoGCN++](https://arxiv.org/pdf/2310.10547) for action recognition

## Data Used

We utilize a curated dataset combining three major action recognition datasets, focusing on the intersection of sport-related activities:

### Source Datasets
- **PennAction Dataset**: Human actions with detailed annotations
- **UCF101**: Large-scale action recognition dataset
- **Kinetics400**: Diverse human action video dataset (5% subset)

### Unified Action Classes
The following sport activities are included in our unified dataset:

| Action                 | PennAction | UCF101 | Kinetics400 | Count |
|------------------------|------------|--------|-------------|-------|
| squat                  | ✓          | ✓      | ✓           | 389   |
| bench_press            | ✓          | ✓      | ✓           | 344   |
| pullup                 | ✓          | ✓      | ✓           | 341   |
| pushup                 | ✓          | ✓      | ✓           | 333   |
| jump_rope              | ✓          | ✓      | ✗           | 241   |
| jumping_jacks          | ✓          | ✓      | ✗           | 235   |
| clean_and_jerk         | ✓          | ✓      | ✗           | 235   |
| lunges                 | ✗          | ✓      | ✓           | 154   |
| wall_pushups           | ✗          | ✓      | ✗           | 130   |
| situp                  | ✓          | ✗      | ✓           | 130   |
| handstand_pushups      | ✗          | ✓      | ✗           | 128   |
| handstand_walking      | ✗          | ✓      | ✗           | 111   |
| snatch_weight_lifting  | ✗          | ✗      | ✓           | 37    |
| running_on_treadmill   | ✗          | ✗      | ✓           | 11    |

## Quick Start

### Installation
1. Install all dependencies and the main package:
```bash
pip install -e .
```
2. Other steps will be added later.
