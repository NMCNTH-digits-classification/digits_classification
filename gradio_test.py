import gradio as gr
import numpy as np
from PIL import Image
from src.models.model import DigitsClassifier
from torchvision import transforms
import torch

# Global variable to store the drawn PIL image
image = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitsClassifier().to(device)
with open('model.pth', 'rb') as f:
    model.load_state_dict(torch.load(f))
black_canvas = np.zeros((28, 28), dtype=np.uint8)
black_canvas_pil = Image.fromarray(black_canvas, mode="L")

def classify(editor_output):
    global image
    # ImageEditor returns a dict; extract the composited image
    if isinstance(editor_output, dict):
        pil_img = editor_output.get("composite")
    else:
        pil_img = editor_output
    img = pil_img  # Save the drawn image as a PIL object
    img_transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = img_transform(img).unsqueeze(0).to(device)
    output = model(img_tensor)
    predicted_digit = torch.argmax(output).item()
    return str(predicted_digit)

with gr.Blocks() as demo:
    editor = gr.ImageEditor(
        value=black_canvas_pil,
        canvas_size=(28, 28),
        image_mode="L",
        type="pil",
        brush=gr.Brush(default_color="#FFFFFF", default_size=3)
    )
    predict_btn = gr.Button("Predict")
    prediction_box = gr.Textbox(label="Prediction", interactive=False, lines=1)
    predict_btn.click(fn=classify, inputs=editor, outputs=prediction_box)
demo.launch()