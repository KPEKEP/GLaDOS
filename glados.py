print("Importing libraries...")
import os
import copy
import yaml
import time
import json
import random
import shutil
from TeraTTS import TTS
from ruaccent import RUAccent
from PIL import ImageGrab, Image
from transliterate import translit
import google.generativeai as genai
from google.generativeai import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
def capture_screen(folder, idx, all_screens):
    """Capture the screen, resize, and return PIL image"""
    screenshot = ImageGrab.grab(all_screens=all_screens)
    screenshot_filename = f"{folder}/screen_{idx}.png"
    
    screenshot.save(screenshot_filename)
    return screenshot_filename

def describe_image(model, prompt, image_path, history):
    """Send the image to Gemini Pro Vision API and get the description."""
    message = {'role': 'user', 'parts': [prompt, image_path]}
    history.append(message)
    
    history_with_images = copy.deepcopy(history)
    for item in history_with_images:
        if len(item['parts'])>1:
            item['parts'][1] = Image.open(item['parts'][1])
                                          
    response = model.generate_content(history_with_images, safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    })
    history.append({'role':'model', 'parts':[response.text]})

    with open('history.json', 'w', encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
        
    return response.text

def text_to_speech(text, tts, accentizer):
    """Process text and convert it to speech."""
    processed_text = accentizer.process_all(text.strip())
    audio = tts(processed_text, play=True, lenght_scale=1.1)
    #tts.save_wav(audio, "./description.wav")

def main():
    print("GLaDOS starts...")
    
    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Create messages history
    history = []
    images_folder = config['screenshots_folder']
    if os.path.exists(images_folder):
        shutil.rmtree(images_folder)
    os.makedirs(images_folder)

    # Read parameters
    api_key = config['api_key']
    temperature = float(config['temperature'])
    top_p = float(config['top_p'])
    top_k = int(config['top_k'])
    sleep_min = int(config['sleep_min'])
    sleep_max = int(config['sleep_max'])
    max_tokens = int(config['max_tokens'])
    comment_chance = float(config['comment_chance'])
    prompt = str(config['prompt'])


    # Init accentizer
    accentizer = RUAccent()
    custom_dict = {'ГЛаДОС' : 'ГЛ+А+ДОС', 
                   'ГЛАДОС' : 'ГЛ+АДОС', 
                   'ГлаДОС' : 'Гл+аДОС',
                   'ИИ' : '+И-+И',
                   'АИ': '+А-+И'}
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
                    image_path = capture_screen(images_folder, len(history), bool(config['capture_all_screens']))
                    description = describe_image(model, prompt, image_path, history)
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