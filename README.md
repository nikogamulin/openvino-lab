# Implementation of ML models for Detection and Recognition using OpenVINO

## Scripts

### exhibition_pipeline.py

```
python3 exhibition_pipeline.py --help
usage: exhibition_pipeline.py [-h] [-wd CAMERA_WIDTH] [-ht CAMERA_HEIGHT]
                              [-numncs NUMBER_OF_NCS] [-vidfps FPS_OF_VIDEO]
                              [-skpfrm NUMBER_OF_FRAME_SKIP] [-o OUTPUT_FILE]
                              [-i INPUT_VIDEO_FILE]
                              [-r REIDENTIFICATION_ENABLED]
                              [-fa FACE_ANALYSIS_ENABLED]

optional arguments:
  -h, --help            show this help message and exit
  -wd CAMERA_WIDTH, --width CAMERA_WIDTH
                        Width of the frames in the video stream. (USB Camera
                        Mode Only. Default=640)
  -ht CAMERA_HEIGHT, --height CAMERA_HEIGHT
                        Height of the frames in the video stream. (USB Camera
                        Mode Only. Default=480)
  -numncs NUMBER_OF_NCS, --numberofncs NUMBER_OF_NCS
                        Number of NCS. (Default=1)
  -vidfps FPS_OF_VIDEO, --fpsofvideo FPS_OF_VIDEO
                        FPS of Video. (USB Camera Mode Only. Default=30)
  -skpfrm NUMBER_OF_FRAME_SKIP, --skipframe NUMBER_OF_FRAME_SKIP
                        Number of frame skip. Default=0
  -o OUTPUT_FILE, --output OUTPUT_FILE
                        Recording Output File
  -i INPUT_VIDEO_FILE, --input INPUT_VIDEO_FILE
                        Input Video File
  -r REIDENTIFICATION_ENABLED, --reidentification REIDENTIFICATION_ENABLED
                        Enable Pedestrian Re-Identification. Default=1 (1 -
                        Enabled; 0 - Disabled)
  -fa FACE_ANALYSIS_ENABLED, --face-analysis FACE_ANALYSIS_ENABLED
                        Enable Face Analysis. Default=1 (1 - Enabled; 0 -
                        Disabled)
 ```
                        
On each frame, person and face detection is performed, using [face-person-detection-retail-0002 model](https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-person-detection-retail-0002/description/face-person-detection-retail-0002.md)
Upon detecting people's bodies and faces, the body and face rectangles are passed to models to recognize age, gender, head pose, and emotions and perform person re-identification.



# References
1. [MobileNet-SSD-RealSense](https://github.com/PINTO0309/MobileNet-SSD-RealSense)
2. [Overview of OpenVINO™ Toolkit Pre-Trained Models](https://github.com/opencv/open_model_zoo/blob/master/intel_models/index.md)
