# CIFAR10 Image Classifier

This project is a web-based application that allows users to upload images of objects belonging to one of the ten categories in the CIFAR10 dataset. A Convolutional Neural Network (CNN) model is used to recognize and predict the category of the uploaded images. The model is trained on the CIFAR10 dataset and achieves high accuracy in object recognition.

## Features
- Upload images of objects to receive predictions of their category.
- View the predicted category along with the confidence level.
- Uses a Flask backend to serve the model and handle predictions.

## Technologies Used
- Python
- Flask
- PyTorch
- torchvision
- HTML/CSS/JavaScript

## Project Structure
```
├── app.py 
├── data_loading.py 
├── models.py 
├── src 
│   ├── data_loading.py 
│   ├── training.py 
│   ├── evaluation.py 
│   ├── app.py 
├── templates 
│   └── index.html 
|   └── result.html 
├── static 
│   ├── style.css 
│   └── script.js 
├── requirements.txt 
└── README.md 
```

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/ThomasDoesAI/CIFAR10-Project.git
    cd CIFAR10-Project
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use \`venv\Scripts\activate\`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Train the Model (if needed):**
   If you haven't already trained the model or want to retrain it, you can use the training script. 
   ```bash
   python src/training.py
   ```

2. **Evaluate the Model:**
    If you want to evaluate the model, you can use the evaluation script.
    ```bash
    python src/evaluation.py
    ```

3. **Run the Flask Application:**
    ```bash
    python app.py
    ```

4. Open your web browser and navigate to `http://127.0.0.1:5000/` to use the image classifier.

## File Descriptions

- **app.py:** The main Flask application file that handles web requests, processes images, and serves predictions.
- **data_loading.py:** Contains functions to download and load the CIFAR10 dataset.
- **models.py:** Defines the CNN model architecture for object recognition.
- **templates/index.html:** HTML template for the web interface.
- **static/style.css:** CSS file for styling the web interface.
- **static/script.js:** JavaScript file for handling user interactions.

## How It Works

1. **Image Upload:** Users can upload images of objects to be classified.
2. **Submit Button:** When the user clicks "Submit", the uploaded image is sent to the Flask backend.
3. **Image Preprocessing:** The backend preprocesses the image (resizes, normalizes).
4. **Model Prediction:** The preprocessed image is fed into the CNN model to get the predicted object category and confidence level.
5. **Display Result:** The predicted category and confidence are displayed to the user.

This README provides a comprehensive guide to set up and use the CIFAR10 Image Classifier project. Let me know if you need any further assistance!
