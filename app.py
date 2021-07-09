import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from PIL import Image
import numpy as np
from torchvision import transforms

# Application
app = Flask('MNIST-Classifier')
ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])
EVAL_PATH = '/eval'
# Is there the EVAL_PATH?
try:
    os.makedirs(EVAL_PATH)
except OSError:
    pass

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def image_preprocessing(fpath):
    """ Here, MNIST preprocessing for PyTorch model """
    img = Image.open(fpath)
    sqrWidth = np.ceil(np.sqrt(img.size[0] * img.size[1])).astype(int)
    img = img.convert('L').resize((sqrWidth, sqrWidth))
    img = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])(img)
    return img 

# Return an Image
@app.route('/', methods=['POST'])
def geneator_handler():
    """Upload an handwrittend digit image in range [0-9], then
    preprocess and classify"""
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")
    
    # Save Image into folder to process 
    filename = secure_filename(file.filename)    
    image_folder = os.path.join(EVAL_PATH, "images")
    try:
        os.makedirs(image_folder)
    except OSError:
        pass    
    input_filepath = os.path.join(image_folder, filename)
    file.save(input_filepath)
        
    # Get model 
    checkpoint = request.form.get("ckp") or "./checkpoints/mnist_convnet_model_epoch_10.pth" # FIX to    
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint))

    # Get image
    data = image_preprocessing(input_filepath)
    data = data.unsqueeze(0)
    data = data.to(device)
    
    # Make inference
    logit = model(data)
    pred = logit.argmax(dim=1, keepdim=True).item()     

    output = "Images: {file}, Classified as {pred}\n".format(file=file.filename, pred=int(pred))
    os.remove(input_filepath)

    return output

if __name__ == "__main__":
    # print("* Starting web server... please wait until server has fully started")
    # app.run(host='0.0.0.0', threaded=False)

    checkpoint = request.form.get("ckp") or "./checkpoints/mnist_convnet_model_epoch_10.pth" # FIX to    
    model = Net().to(device)
    model.load_state_dict(torch.load(checkpoint))
