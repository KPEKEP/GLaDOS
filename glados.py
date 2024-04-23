print("Importing libraries...")
import yaml
import time
from PIL import ImageGrab, Image
from TeraTTS import TTS
from ruaccent import RUAccent
import google.generativeai as genai
from google.generativeai import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from transliterate import translit
import random

def capture_screen(resize_dimensions):
    """Capture the screen, resize, and return PIL image"""
    screenshot = ImageGrab.grab()
    screenshot = screenshot.resize(resize_dimensions, Image.Resampling.LANCZOS)
    screenshot.save("screen.png")
    return screenshot

def describe_image(model, prompt, image, history):
    """Send the image to Gemini Pro Vision API and get the description."""
    message = {'role': 'user', 'parts': [prompt, image]}
    history.append(message)
    response = model.generate_content(history, safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    })
    history.append({'role':'model', 'parts':[response.text]})
    return response.text

def text_to_speech(text, tts, accentizer):
    """Process text and convert it to speech."""
    processed_text = accentizer.process_all(text.strip())
    audio = tts(processed_text, play=True, lenght_scale=1.1)
    tts.save_wav(audio, "./description.wav")

def main():
    print("GLaDOS starts...")
    
    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Create messages history
    history = []
    
    # Read parameters
    api_key = config['api_key']
    temperature = float(config['temperature'])
    top_p = float(config['top_p'])
    top_k = int(config['top_k'])
    sleep_min = int(config['sleep_min'])
    sleep_max = int(config['sleep_max'])
    max_tokens = int(config['max_tokens'])
    comment_chance = float(config['comment_chance'])
    resize_dimensions = tuple(config['resize_dimensions'])
    prompt = str(config['prompt'])


    # Init accentizer
    accentizer = RUAccent()
    custom_dict = {'ГЛаДОС' : 'ГЛ+А+ДОС', 
                   'ГЛАДОС' : 'ГЛ+АДОС', 
                   'ГлаДОС' : 'Гл+аДОС',
                   'ИИ' : '+И-+И'}
    accentizer.load(omograph_model_size='turbo', use_dictionary=True, custom_dict=custom_dict)

    # Init TTS
    tts = TTS("TeraTTS/glados2-g2p-vits", add_time_to_end=1.0, tokenizer_load_dict=True)

    # Configure and init Gemini
    genai.configure(api_key=api_key)
    generation_config = GenerationConfig(max_output_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest', generation_config = generation_config)
    
    try:
        print("GLaDOS started!")
        while True:
            if random.random() <= comment_chance:
                print("GLaDOS decides to comment!")
                try:
                    image = capture_screen(resize_dimensions)
                    description = describe_image(model, prompt, image, history)
                    description = translit(description, 'ru')
                    print(f"GLaDOS thinks: \"{description}\"")
                except Exception as e:
                    print("Failed to describe: "+e)
                else:
                    text_to_speech(description, tts, accentizer)
            else:
                print("GLaDOS decides to keep silence")                            
            sleep_time = random.randint(sleep_min, sleep_max)
            print(f"GLaDOS is sleeping for {sleep_time}s...")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopped by user. You're monster.")

if __name__ == '__main__':
    main()