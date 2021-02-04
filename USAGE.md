
## Getting Started
- Check your python version, this is built on `python 3.6`
- Install `pytorch 1.7.1` and dependencies from https://pytorch.org/
- Install packages `cv2 4.1.1`

- Clone this repo:

```bash
git clone https://github.com/NeuralVFX/face-pose-estimation-pytorch-v2.git
```
## Contents
- Some simple libraries for generating artificial training data
- Network, dataloader, and traning loop
- A Jupyter Notebook for training the model and exporting to Jit
- A Jupyter Notebook to export deformable 3d landmarks to Jit (For use in DLL for PnP solve) 

### Train The Model
- Run [train_network.ipynb](train_network.ipynb) Jupyter Notebook
- Jit exports will go into the `output` directory

### Export Landmark Blendshape Model
- Run [export_blendshape_model.ipynb](export_blendshape_model.ipynb) Jupyter Notebook
- Jit export will go into the `output` directory

### Use The Models
- Take the `ptc` files from the `/output/`directory
- Load these model in LibTorch

## Included Face Data Usage


| **Filename**         | **Purpose**                                                        |  **Usage**             |
|------------------------------|--------------------------------------------------------------------|-------------------------|
| `data/face_model.mb`    |  Maya file of face, including blendshapes |    Open in Maya 2018 or later            |
| `data/bs_points_a.json`   | Face model points, and blendshape points exported  |   Loaded with JSON in `util/loaders.py`        |
| `data/maya_scripts.py`      |  Script to export data from Maya file to JSON file      |   Open `face_model.mb` and execute in script editor                   |

