
## Getting Started
- Check your python version, this is built on `python 3.6`
- Install `pytorch 1.7.1` and dependencies from https://pytorch.org/
- Install packages `cv2 4.1.1`

- Clone this repo:

```bash
git clone https://github.com/NeuralVFX/face-pose-estimation-pytorch-v2.git
```

## Train The Model
- Run [train_network.ipynb](train_network.ipynb) Jupyter Notebook

## Use The Model
- Take the `ptc` file from the `/output/`directory
- Load this model in LibTorch

## Included Face Data Usage


| **Filename**         | **Purpose**                                                        |  **Usage**             |
|------------------------------|--------------------------------------------------------------------|-------------------------|
| `data/face_model.mb`    |  Maya file of face, including blendshapes |    Open in Maya 2018 or later            |
| `data/bs_points_a.json`   | Face model points, and blendshape points exported  |   Loaded with JSON in `util/loaders.py`        |
| `data/maya_scripts.py`      |  Script to export data from Maya file to JSON file      |   Open `face_model.mb` and execute in script editor                   |

