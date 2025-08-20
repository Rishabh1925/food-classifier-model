from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTConfig
from PIL import Image
import base64
import io
import numpy as np

app = Flask(__name__)
CORS(app)


FOOD_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

class FoodClassifier(nn.Module):
    def __init__(self, num_classes=101):
        super(FoodClassifier, self).__init__()
    
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        ).vit
        

        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):

        vit_outputs = self.vit(x)
        features = vit_outputs.last_hidden_state[:, 0]
        

        return self.classifier(features)
    

class HuggingFaceFoodClassifier(nn.Module):
    def __init__(self, model_path='food101_model.pth', num_classes=101):
        super(HuggingFaceFoodClassifier, self).__init__()

        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_labels=num_classes
        )
        self.model = ViTForImageClassification(config)
        
    def load_weights(self, model_path, device):
    
        state_dict = torch.load(model_path, map_location=device)
        
    
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('vit.'):
            
                new_key = key.replace('vit.', 'vit.')
                new_state_dict[new_key] = value
            elif key.startswith('classifier.'):
            
                new_key = key.replace('classifier.', 'classifier.')
                new_state_dict[new_key] = value
        
        self.model.load_state_dict(new_state_dict, strict=False)
        
    def forward(self, x):
        return self.model(x).logits
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def load_model():
    """Load the pre-trained model"""
    global model
    try:

        try:
            model = HuggingFaceFoodClassifier(num_classes=101)
            model.load_weights('food101_model.pth', device)
            model.to(device)
            model.eval()
            print("Hugging Face Food-101 model loaded successfully!")
            return
        except Exception as hf_error:
            print(f"Failed to load as Hugging Face model: {hf_error}")
    
        try:
            model = FoodClassifier(num_classes=101)
            state_dict = torch.load('food101_model.pth', map_location=device)
            

            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            
            model.to(device)
            model.eval()
            print("Custom ViT Food-101 model loaded successfully!")
            return
        except Exception as custom_error:
            print(f"Failed to load custom ViT model: {custom_error}")
        
        print("Using pre-trained ViT without Food-101 specific weights")
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=101,
            ignore_mismatched_sizes=True
        )
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def preprocess_image(image_data):
    """Preprocess base64 image data"""
    try:
       
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
       
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
   
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
       
        input_tensor = transform(image).unsqueeze(0).to(device)
        return input_tensor
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
       
        input_tensor = preprocess_image(data['image'])
        
       
        with torch.no_grad():
            if hasattr(model, 'logits'):  
                outputs = model(input_tensor).logits
            else:  
                outputs = model(input_tensor)
            
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
       
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            predictions = []
            for i in range(5):
                class_name = FOOD_CLASSES[top5_indices[i].item()]
                confidence = top5_prob[i].item()
                predictions.append({
                    'class': class_name.replace('_', ' ').title(),
                    'confidence': confidence
                })
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify({
            'error': f'Classification failed: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("Starting FoodVision AI...")
    load_model()
    
  
    import os
    os.makedirs('templates', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=False)
