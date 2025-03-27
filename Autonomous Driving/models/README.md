# Lane Keeping models

This folder contains two sets of models (biased and generic) and the code used to define and train the models. The models were trained over data collected using the Carla simulator (https://carla.org/). We use Scenic (https://docs.scenic-lang.org/en/latest/quickstart.html) to generate scenarios.

The models receive as input images of the road in front of the car and a value indicating if the car has to turn (-1=left, 0=straight, 1=right). The output is the CTE (cross track error), which can then be provided to a PID controller to keep the car centered in a lane.  

- **Biased** (12 models): Each model is trained only on images of specific weather and light conditions, defined by the Carla weather presets. For example, ```model_clearnoon.pth``` is trained only on images of the ClearNoon weather preset.
- **Generic** (15 models): Each model is trained on a set of images collected randomly from the 14 Carla weather presets. The training datasets used were disjoint.

## How to use the models
The code used to define and train the models is contained in ```cnn_pytorch.py```. The models can be easily loaded using the following code, where ```model_path``` is the path of the model to be loaded:


```python
import torch
from cnn_pytorch import CNN

self.model = CNN(resnet=False).to("cuda")
self.model.load_state_dict(torch.load(model_path))

```

## Camera configuration

It is important to have the same configuration for the car camera and image preprocessing when using the models. The camera configuration and the image preprocessing are as follows:

```python
# --------------------------------
# Add front camera to agent
# --------------------------------

# Configuration
cam_config = self.carlaActor.get_world().get_blueprint_library().find('sensor.camera.rgb')
cam_config.set_attribute("image_size_x",str(320))
cam_config.set_attribute("image_size_y",str(160))
cam_config.set_attribute("fov",str(50))
cam_location = carla.Location(2,0,1)
cam_rotation = carla.Rotation(0,0,0)
cam_transform = carla.Transform(cam_location,cam_rotation)

self.front_cam = self.carlaActor.get_world().spawn_actor(cam_config, cam_transform, 
    attach_to = self.carlaActor, 
    attachment_type = carla.AttachmentType.Rigid)
```

```python
def _front_cam_callback(self,image):

    self.current_front_img_raw = image
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (image.height, image.width, 4)) # RGBA format
    array = array[:, :, :3] #  Take only RGB
    cropped_img = array[80:160,20:300]
    self.current_front_img = cropped_img[:, :, ::-1]

``` 


## Citation

The models can be freely used. We kindly ask to cite the following paper:

```
@inproceedings{DBLP:conf/atva/TorfahXJVS22,
  author       = {Hazem Torfah and
                  Carol Xie and
                  Sebastian Junges and
                  Marcell Vazquez{-}Chanlatte and
                  Sanjit A. Seshia},
  editor       = {Ahmed Bouajjani and
                  Luk{\'{a}}s Hol{\'{\i}}k and
                  Zhilin Wu},
  title        = {Learning Monitorable Operational Design Domains for Assured Autonomy},
  booktitle    = {Automated Technology for Verification and Analysis - 20th International
                  Symposium, {ATVA} 2022, Virtual Event, October 25-28, 2022, Proceedings},
  series       = {Lecture Notes in Computer Science},
  volume       = {13505},
  pages        = {3--22},
  publisher    = {Springer},
  year         = {2022},
  url          = {https://doi.org/10.1007/978-3-031-19992-9\_1},
  doi          = {10.1007/978-3-031-19992-9\_1},
  timestamp    = {Mon, 05 Feb 2024 20:35:20 +0100},
  biburl       = {https://dblp.org/rec/conf/atva/TorfahXJVS22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Questions?

Please reach out to Hazem Torfah (hazemto@chalmers.se)! 

