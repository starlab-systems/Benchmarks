# Lane Keeping models

This folder contains two sets of models trained to keep an autonomous car centered in a lane and the code used to define and train the models. The models were trained over data collected using the Carla simulator (https://carla.org/) and Scenic (https://docs.scenic-lang.org/en/latest/quickstart.html) to generate scenarios.

The models receive as input images of the road in front of the car and a value indicating if the car has to turn (-1=left, 0=straight, 1=right). The output is the CTE (cross track error), which can then be provided to a PID controller to keep the car centered in a lane.  

The two sets are in the folders:
- **Biased** (12 models): Each model is trained only on images of specific weather and light conditions, defined by the Carla weather presets. For example, ```model_clearnoon.pth``` is trained only on images of the ClearNoon weather preset.
- **Generic** (15 models): Each model is trained on a set of images collected from the 14 Carla weather presets. The training datasets used were disjoint.

Paper: [Learning Monitor Ensembles for Operational Design Domains](https://link.springer.com/chapter/10.1007/978-3-031-44267-4_14), appeared at [Runtime Verification 2023](https://rv23.csd.auth.gr/).

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

If you find this work is helpful in your research, please cite our work:

```
@inproceedings{torfah2023monitor,
    author="Torfah, Hazem
    and Joshi, Aniruddha
    and Shah, Shetal
    and Akshay, S.
    and Chakraborty, Supratik
    and Seshia, Sanjit A.",
    editor="Katsaros, Panagiotis
    and Nenzi, Laura",
    title="Learning Monitor Ensembles for Operational Design Domains",
    booktitle="Runtime Verification",
    year="2023",
    publisher="Springer Nature Switzerland",
    pages="271--290",
    isbn="978-3-031-44267-4"
}
```