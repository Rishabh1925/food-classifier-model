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

1. **Clone the repository**
```bash
git clone https://github.com/Rishabh1925/food-classifier-model.git
cd foodvision-ai
```

2. **Create and activate virtual environment**
```bash
python -m venv env
source env/bin/activate  # Linux/macOS
# env\Scripts\activate     # Windows
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

## Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Docker Setup
```bash
# Build image
docker build -t ai-vision-classifier .

# Run container
docker run -p 5000:5000 ai-vision-classifier
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

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` file for more information.

## Support

- [**GitHub Issues**](https://github.com/yourusername/ai-vision-classifier/issues)
- [**Email**](rishabhranjansingh_mc24b06_019@dtu.ac.in)
