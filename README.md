---

# Gender Age Detector 🎨

Welcome to the Gender Age Detector project! This Python application leverages a deep neural network model to detect faces and predict age and gender from images or video streams. Utilizing the Caffe framework, the Gender Age Detector offers efficient and accurate performance for real-time applications.

## Features

- **Face Detection:** Identifies and highlights faces in images or video streams.
- **Age Prediction:** Estimates the age group of detected faces.
- **Gender Prediction:** Determines the gender of detected faces.
- **Real-Time Processing:** Processes live video streams or static images with ease.

## Model Architecture

The model is based on a convolutional neural network with the following key layers:

- **Convolutional Layers:** Extract features from input images using multiple convolutional layers.
- **Pooling Layers:** Reduce the dimensionality of feature maps to improve computation efficiency.
- **Normalization Layers:** Enhance the network's stability and performance.
- **Fully Connected Layers:** Combine features to make final predictions.
- **Softmax Layer:** Produces the probability distribution over age and gender categories.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sarangs1621/gender-age-detector.git
   ```

2. Navigate to the project directory:
   ```bash
   cd gender-age-detector
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained models:
   - [Face Detection Model](https://github.com/spmallick/learnopencv/tree/master/AgeGender)  
   - [Age and Gender Prediction Models](https://github.com/spmallick/learnopencv/tree/master/AgeGender)

5. Place the model files in the project directory.

## Usage

Run the application using Python:

```bash
python gender_age.py 
```

or for video input:

```bash
python gender_age.py
```

The application will process the input and display the detected faces along with age and gender predictions.

## Contributing

We welcome contributions to improve the Gender Age Detector project! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
