# Brain Tumor Detection Using Deep Learning

## Overview
This project is a **full-stack web application** designed to classify brain tumors from MRI scan images. It utilizes a deep learning model based on **VGG16** with transfer learning and provides predictions with confidence scores through an intuitive user interface. The application is capable of identifying four classes of brain tumors: **meningiomas**, **gliomas**, **pituitary tumors**, and **no tumor**.

## Demo Video
[Brainalyzer](https://www.youtube.com/watch?v=6oDoeOF2FF0)


## Features
- Upload MRI scan images via a user-friendly web interface.
- Process images using a pre-trained **VGG16 model** fine-tuned for this classification task.
- Display tumor predictions with confidence scores.
## Dataset
The model was trained on a dataset containing over **1,300 MRI images**, categorized into four classes:
1. Meningiomas
2. Gliomas
3. Pituitary tumors
4. No tumor

## Technologies Used
### Backend:
- **Python**
- **Flask** (for building the web server and handling backend operations)
- **TensorFlow/Keras** (for deep learning model implementation)

### Frontend:
- **HTML/CSS**
- **JavaScript**
![Screenshot 2025-01-23 001453](https://github.com/user-attachments/assets/83eae89c-48f9-421e-83d8-b853da63b006)

### Model:
- **VGG16** pre-trained on ImageNet, fine-tuned for multi-class classification.

### Additional Libraries:
- **NumPy**, **Matplotlib**, **Seaborn** (for data processing and visualization)
- **Scikit-learn** (for model evaluation)
- **OpenCV** (for image processing)

## How It Works
1. **Preprocessing:** Images are resized to **128x128** pixels, normalized, and augmented to improve generalization.
2. **Model Training:** The VGG16 model was fine-tuned, freezing initial layers and training the last few layers for brain tumor classification.
3. **Evaluation:** Metrics such as **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC AUC** were used to assess model performance, achieving **95% accuracy** on unseen data.
4. **Web Application:** The trained model is deployed using Flask, allowing users to upload MRI scans and receive predictions with confidence scores.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Put the trained model.h5 file in the models directory.
4. Put the index.html file in the templates directory.
5. Run the application:
   ```bash
   python app.py
   ```
6. Open your browser and navigate to `http://localhost:5000`.

## Usage
- Upload an MRI scan image using the web interface.
- Wait for the model to process the image and display the tumor classification along with confidence scores.

## Results
![Screenshot 2025-01-23 001535](https://github.com/user-attachments/assets/e242cb07-5f29-4bb3-86f4-e1098a4529c2)
![Screenshot 2025-01-23 001600](https://github.com/user-attachments/assets/c213f1a9-db23-4b24-b6dd-a9a612b2e9c3)

### Classification Report
```
              precision    recall  f1-score   support

           0       0.97      0.98      0.98       300
           1       0.93      0.90      0.91       300
           2       0.95      1.00      0.97       405
           3       0.93      0.91      0.92       306

    accuracy                           0.95      1311
   macro avg       0.95      0.94      0.95      1311
weighted avg       0.95      0.95      0.95      1311
```

## Future Work
- Expand the dataset for improved generalization.
- Add support for additional tumor types.
- Integrate real-time image acquisition from medical imaging devices.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## Acknowledgments
- **VGG16** model pre-trained on ImageNet.
- Resources and datasets used for training and evaluation.

---
