from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CIFAR10CNN

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = CIFAR10CNN()
model.load_state_dict(torch.load('models/cifar10_cnn.pth'))
model.eval()

# Define the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the transformation to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Verify the file was saved
            if os.path.exists(filepath):
                print(f'File saved successfully: {filepath}')
            else:
                print('File not saved')
            
            # Process the uploaded image
            image = Image.open(filepath)
            image = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Predict the class of the image
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                class_name = classes[predicted.item()]
            
            # Render the result template with the image and classification
            return render_template('result.html', filename=filename, class_name=class_name)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Verify the file exists
    if os.path.exists(filepath):
        print(f'Serving file: {filepath}')
    else:
        print('File not found')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
