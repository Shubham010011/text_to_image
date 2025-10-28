from dotenv import load_dotenv, find_dotenv
import requests, os, io
from PIL import Image
from datetime import datetime
import time
from flask import Flask, render_template, request, redirect, url_for

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = ("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")

def texttoimage(prompt: str, retries: int = 3, wait_time: int = 10):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payload = {"inputs": prompt}

    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        content_type = response.headers.get("content-type", "")

        if "image" in content_type:
           image_bytes = response.content
           img = Image.open(io.BytesIO(image_bytes))

            # Save the image inside static folder
           if not os.path.exists("static"):
                os.makedirs("static")

                filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
                save_path = os.path.join("static", filename)
                img.save(save_path)

                return save_path


        else:
            try:
                error_json = response.json()
                print("API Response:", error_json)

                # If the model is still loading, retry
                if "estimated_time" in error_json:
                    print(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            except Exception:
                print("Unexpected response (not image or JSON):", response.text[:200])
                return None
    return None

app = Flask(__name__, template_folder='.')

@app.route('/', methods=['GET'])
def index():
    # Renders the initial form
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    # 1. Get the prompt from the form data
    prompt = request.form.get('prompt')
    if not prompt:
        # Handle case where prompt is missing
        return redirect(url_for('index'))

    print(f"Generating image for prompt: '{prompt}'")
    
    # 2. Call the text-to-image function
    filename = texttoimage(prompt)

    # 3. Render the page again, passing the filename to display the image
    if filename:
        return render_template('index.html', image_filename=filename)
    else:
        # Handle error case
        return "Error generating image. Check server logs.", 500

if __name__ == '__main__':
    # Use environment port for deployment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
