# Development-of-rPPG-algorithm-to-monitor-Human-cardiac-activities-using-commercial-Webcams

Developed a face detector and carried out real time face tracking to extract the pulse information. 
Using the extracted RGB signal heart rate is measured with HeartPy library.

## Contents

Implementation of remote photoplethysmography algorithm in OpenCV for measuring heart rate, located in 'heart_rate.py'. The method employs multithreading to efficiently process images and accurately depict the heart rate of a subject. To enhance our work, we've utilized the VicarPPG-2 dataset from TU Delft.

## Requirements

* python==3.9
* opencv-python==4.5.1
* numpy==1.20.2
* scipy==1.7.1
* mediapipe==0.8.4.2
* matplotlib==3.4.1
* heartpy==1.2.7
* XlsxWriter==3.0.1

## Setup

1.  Install PyTorch and other required python libraries with:

    ```
    pip install -r requirements.txt
    ```

## Usage

To run the python file in terminal, use the following command:

`python heart_rate.py`
