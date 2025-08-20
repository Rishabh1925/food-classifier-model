# AI Vision Classifier

An intelligent web application powered by PyTorch and Transformers for real-time image classification with a sleek Flask interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This project combines cutting-edge deep learning models with a modern web interface to deliver accurate image classification. Built with PyTorch's latest features and Hugging Face Transformers, it provides an intuitive way to classify images through a responsive web application.

## Features

- **AI-Powered Classification**: Advanced neural networks for accurate predictions
- **Real-Time Processing**: Instant results with optimized inference
- **Web Interface**: Clean, modern UI built with Flask
- **Model Flexibility**: Support for various transformer architectures
- **Production Ready**: Gunicorn WSGI server integration

## Technology Stack

- **Core ML**: PyTorch 2.0+, Transformers
- **Backend**: Flask, Flask-CORS, Gunicorn
- **Image Processing**: Pillow, torchvision
- **Data Handling**: NumPy, requests
- **Progress Tracking**: tqdm

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (recommended for model loading)

## Installation & Setup

### Method 1: Using Git (Recommended)

If you have Git installed on your system:

1. **Clone the repository**
```bash
git clone https://github.com/Rishabh1925/food-classifier-model.git
cd foodvision-ai
```

### Method 2: Without Git (Manual Download)

If you don't have Git installed:

1. **Download the project manually**
   - Go to: https://github.com/Rishabh1925/food-classifier-model
   - Click the green "Code" button
   - Select "Download ZIP"
   - Extract the ZIP file to your desired location
   - Open terminal/command prompt and navigate to the extracted folder:
   ```bash
   cd path/to/extracted/food-classifier-model-main
   ```

### Method 3: Install Git (One-time setup)

**For Windows:**
- Download Git from: https://git-scm.com/download/windows
- Run the installer and follow the setup wizard
- Restart your command prompt/terminal
- Then use Method 1 above

**For macOS:**
```bash
# Using Homebrew (if you have it)
brew install git

# Or download from: https://git-scm.com/download/mac
```

**For Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install git
```

**For Linux (CentOS/RHEL/Fedora):**
```bash
sudo yum install git
# OR for newer versions
sudo dnf install git
```

---

### Continue with setup (after using any method above):

2. **Create and activate virtual environment**
```bash
python -m venv env

# Activate virtual environment:
# For Windows:
env\Scripts\activate

# For macOS/Linux:
source env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model and setup resources**
```bash
python download.py
```

5. **Launch the application**
```bash
python app.py
```

6. **Access the app**
Open your browser and navigate to `http://localhost:5000`

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
flask>=2.3.0
flask-cors>=4.0.0
Pillow>=9.5.0
numpy>=1.24.0
gunicorn>=21.0.0
tqdm==4.65.0
transformers>=4.30.0
requests>=2.28.0
```

## Project Structure

```
ai-vision-classifier/
├── app.py              # Flask application entry point
├── download.py         # Model and resource downloader
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
├── static/             # CSS, JS, images
├── models/             # Downloaded ML models
├── utils/              # Helper functions
└── README.md           # This file
```

## Web Interface

The application features a modern, responsive design with:
- Drag-and-drop image upload
- Real-time prediction display
- Confidence score visualization
- Mobile-friendly interface

## Model Information

The application supports various pre-trained models from Hugging Face:
- Vision Transformers (ViT)
- ResNet architectures  
- EfficientNet variants
- Custom fine-tuned models

## Important Notes

1. **First Run**: Always execute `python download.py` before starting the app
2. **Memory Usage**: Ensure sufficient RAM for model loading
3. **GPU Support**: CUDA-compatible GPU recommended for faster inference
4. **Model Updates**: Re-run `download.py` when switching models

## Troubleshooting

**Dependencies Issue:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**Model Download Fails:**
```bash
python download.py --retry 3 --verbose
```

**Port Already in Use:**
```bash
python app.py --port 8080
```

**Virtual Environment Issues:**
```bash
# If you get permission errors or path issues
python -m pip install --user virtualenv
python -m virtualenv env
```

**Python Not Found (Windows):**
- Make sure Python is added to your PATH
- Try using `python3` or `py` instead of `python`
- Reinstall Python and check "Add to PATH" during installation

## Alternative Download Methods

If you're having trouble with any of the above methods:

1. **Direct ZIP Download**: Download the project as ZIP from the GitHub repository
2. **GitHub CLI**: `gh repo clone Rishabh1925/food-classifier-model`
3. **Browser Download**: Save the repository files directly from GitHub's web interface

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` file for more information.

## Support

- [**GitHub Issues**](https://github.com/Rishabh1925/food-classifier-model/issues)
- [**Email**](rishabhranjansingh_mc24b06_019@dtu.ac.in)
