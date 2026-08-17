# Data layout

The repository does not redistribute the benchmark datasets. Prepare the
licensed datasets under this directory before training:

```text
data/
├── ats_train.json
├── totaltext_train.json
├── ic15_train.json
├── ctw1500_train.json
├── rects_train.json
├── test.json
└── images/
    └── ...
```

Each training JSON is a list of records with relative image paths and
coordinates normalized to the Qwen 0-1000 coordinate space:

```json
[
  {
    "image": "ats/train/example.jpg",
    "problem": "Spot all the text in the image with word-level, and output in JSON format.",
    "normalized_solution": [
      {"bbox_2d": [120, 210, 480, 330], "text_content": "example"}
    ]
  }
]
```

`test.json` uses the same fields and may additionally contain the pixel-space
`solution` field used by the evaluation scripts. Test annotations are never
read by the training entry point.

