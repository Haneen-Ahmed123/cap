# Oryx J022201
### Oryx the Stok M8 aims to automate inventory management in retail stores by utilizing computer vision and deep learning techniques to detect empty spaces on store shelves. Traditional manual methods for monitoring shelf stock can be time-consuming and error-prone. By applying YOLOv10, an advanced object detection algorithm, we can revolutionize this process, providing real-time and accurate identification of vacant shelf spaces.

### The key features include automated empty space detection and precise localization. The YOLOv10 model detects and localizes empty spaces on store shelves, eliminating the need for manual monitoring and reducing operational costs. The model operates in real-time, allowing for instantaneous detection and response to changes in shelf stock. YOLOv10 provides high-precision localization of empty spaces, offering valuable insights for optimizing shelf layouts and stock levels.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Model 
### Type: YOLOv10 Ultralytics
### purpose: can detection all empty space in stand in hapermarkt 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dataset used
### [Dataset Link](https://drive.google.com/file/d/1NmtMoH3OipUKnSgL8k9nrQPSteMmSXNG/view?usp=drive_link)

## preprocessing steps
### Step1: Labeling
![image](https://github.com/user-attachments/assets/1557f9f8-6f17-48db-9ec0-eb5cfc240fa8)
### we Labeling data by  Roboflow Annotate,Once the dataset version is generated, I clicked Export and select the YOLOv10 dataset format.

### Step2: Now that I have the images and annotations added, I generated a new dataset version. I used train-test split, by splitting 70% training, 20% validation and 10% testing. When generating a new version of data, you can elect to add preprocessing and augmentations. This step is completely optional, however, it can allow to significantly improve the robustness of the model.

### Step3:The preprocessing includes Resize, which downsizes the image for smaller file sizes and faster training. Auto-Orient discards any orientated images. The augmentation is a crucial process. It helps us create more images of the same dataset by generating augmented version of each images. Firstly, I used 90 Rotate, which adds 90-degree rotations to help the model be insensitive to camera orientations because a picture can be taken vertically and horizontally. And, Exposure includes adding variability to image brightness to help the model be more resilient to lighting and camera settings, because a good quality image isn't always available.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Model performance
## Metrics:
![image](https://github.com/user-attachments/assets/63d1f544-a170-4dcb-8e6f-817694531eae)

## F1:
![image](https://github.com/user-attachments/assets/a1483877-4bd5-4ee8-9743-0e509d6162fb)

## Precision:
![image](https://github.com/user-attachments/assets/03a04329-2e6c-4a9e-80b5-4e6ee35e9089)

## Recall:
![image](https://github.com/user-attachments/assets/346791d8-8ca7-40a1-9a1e-6c32805a86d6)

## Results:
![image](https://github.com/user-attachments/assets/8d6a061b-53d6-4cb4-9847-8639aa7e6fea)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# predict and detect
![image](https://github.com/user-attachments/assets/e2111e6a-a725-4a33-beb4-35ea83ffab29)
![image](https://github.com/user-attachments/assets/10a0dd03-ada4-496f-9a78-6f50d48aef3b)
![image](https://github.com/user-attachments/assets/9500cd7a-ca9b-4194-a02d-21153e8ec537)
![image](https://github.com/user-attachments/assets/03646dcc-61b3-4f82-81c9-f8cdfaf041b9)

# How Run Oryx_01_model.pt
### Frist Downloads Libraries
    
    pip install -r requirements.txt

### Second open app.py and run code
  #### note: must the path for model in divce same the path in code 
  #### ex:
  #### in code
![image](https://github.com/user-attachments/assets/560aa2fa-35a3-4544-b25b-b4c30d80ec8d)
  #### in divce
![image](https://github.com/user-attachments/assets/247f57b2-bb36-4003-b667-7e90d6fbed1c)


  

    


