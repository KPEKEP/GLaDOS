print("Importing libraries...")
import os
import copy
import yaml
import time
import json
import random
import shutil
import traceback
from TeraTTS import TTS
from ruaccent import RUAccent
from PIL import ImageGrab, Image
from transliterate import translit
import ollama
from transformers import SeamlessM4Tv2Model, AutoProcessor
import google.generativeai as genai
from google.generativeai import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def capture_screen(folder, idx, all_screens):
    """Capture the screen, resize, and return PIL image"""
    screenshot = ImageGrab.grab(all_screens=all_screens)
    screenshot_filename = f"{folder}/screen_{idx}.png"
    
    screenshot.save(screenshot_filename)
    return screenshot_filename

def describe_image_ollama(config, image_path, history):
    """Describe image and role-play using Ollama"""
    vision_message = {'role': 'user', 'content': config['ollama']['vision']['prompt'], 'images': [image_path]}
    screen_description = ollama.chat(model=config['ollama']['vision']['model_name'], 
                                    messages=[vision_message], 
                                    options=config['ollama']['vision']['options'])
    print("Screen description:", screen_description['message']['content'])
    
    role_message = {'role': 'user', 'content': config['ollama']['role']['prompt']}
    role_message['content'] = role_message['content'].replace('<SCREENSHOT>', screen_description['message']['content'])
    history.append(role_message)
    response = ollama.chat(model=config['ollama']['role']['model_name'], 
                            messages=[role_message], 
                            options=config['ollama']['role']['options'])
    print("GLaDOS rephrase:", response['message']['content'])
    
    return response['message']['content']

def describe_image_gemini(config, image_path, history):
    """Describe image and role-play using Gemini"""
    model = genai.GenerativeModel(config['gemini']['model_name'], generation_config=GenerationConfig(**config['gemini']['generation_config']))
    
    message = {'role': 'user', 'parts': [config['gemini']['prompt'], image_path]}
    history.append(message)
    
    history_with_images = copy.deepcopy(history)
    for item in history_with_images:
        if len(item['parts']) > 1:
            item['parts'][1] = Image.open(item['parts'][1])
    
    response = model.generate_content(history_with_images, safety_settings=config['gemini']['safety_settings'])
    print("GLaDOS rephrase:", response.text)
    
    return response.text

def update_history(config, history, content):
    """Update history based on configuration"""
    if bool(config.get('keep_history', False)):
        if config['inference_engine'] == 'ollama':
            history.append({'role': 'assistant', 'content': content})
        elif config['inference_engine'] == 'gemini':
            history.append({'role': 'model', 'parts': [content]})
        else:
            raise Exception(f"Unknown inference engine: {config['inference_engine']}")
        with open('history.json', 'w', encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    else:
        history.clear()
    return history

def text_to_speech(text, tts, accentizer):
    """Process text and convert it to speech."""
    processed_text = accentizer.process_all(text.strip())
    audio = tts(processed_text, play=True, lenght_scale=1.1)
    # tts.save_wav(audio, "./description.wav")

def translate(model, processor, text):
    """Translate from english to russian using Seamless M4T"""
    text_inputs = processor(text=text, src_lang="eng", return_tensors="pt").to("cpu")
    output_tokens = model.generate(**text_inputs, tgt_lang="rus", generate_speech=False)
    translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return translated_text

def main():
    print("GLaDOS starts...")
    
    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Initialize variables and create folders
    history = []
    images_folder = str(config['screenshots_folder'])
    if os.path.exists(images_folder):
        shutil.rmtree(images_folder)
    os.makedirs(images_folder)
    
    # Initialize TTS and accentizer
    accentizer = RUAccent()
    custom_dict = {
        'ГЛаДОС': 'ГЛ+А+ДОС', 
        'ГЛАДОС': 'ГЛ+АДОС', 
        'ГлаДОС': 'Гл+аДОС',
        'ИИ': '+И-+И',
        'АИ': '+А-+И'
    }
    accentizer.load(omograph_model_size='turbo', use_dictionary=True, custom_dict=custom_dict)
    tts = TTS("TeraTTS/glados2-g2p-vits", add_time_to_end=1.0, tokenizer_load_dict=True)

    # Initialize translation model if needed
    need_translation = (config['inference_engine'] == 'ollama') and bool(config['ollama']['role'].get('translate', False))
    if need_translation:
        processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        translation_model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    
    # Initialize Gemini if needed
    if config['inference_engine'] == 'gemini':
        genai.configure(api_key=config['gemini']['api_key'])

    try:
        print("GLaDOS started!")
        while True:
            if random.random() <= float(config['comment_chance']):
                print("GLaDOS decides to comment!")
                try:
                    image_path = capture_screen(images_folder, len(history), bool(config['capture_all_screens']))
                    
                    if config['inference_engine'] == 'ollama':
                        description = describe_image_ollama(config, image_path, history)
                    elif config['inference_engine'] == 'gemini':
                        description = describe_image_gemini(config, image_path, history)
                    else:
                        raise ValueError(f"Unknown inference engine: {config['inference_engine']}")
                    
                    history = update_history(config, history, description)
                    
                    if need_translation:
                        description = translate(translation_model, processor, description)
                    
                    description = translit(description, 'ru')
                    for k, v in custom_dict.items():
                        description = description.replace(k, v)
                    
                    print(f"GLaDOS thinks: \"{description}\"")
                    
                except Exception as e:
                    print("Failed to describe: " + str(e))
                    traceback.print_exc()
                else:
                    text_to_speech(description, tts, accentizer)
            else:
                print("GLaDOS decides to keep silence")
            
            sleep_time = random.randint(int(config['sleep_min']), int(config['sleep_max']))
            print(f"GLaDOS is sleeping for {sleep_time}s...")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopped by user. You're monster.")

if __name__ == '__main__':
    main()