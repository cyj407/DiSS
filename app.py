
import imp
from mimetypes import init
from flask import Flask, current_app, send_file, request, jsonify
from optimization.image_editor import ImageEditor
app = Flask(__name__, static_url_path='')

import io, re, base64, os
import numpy as np
from PIL import Image

# from test import evaluate, setup_model

import argparse
from optimization.image_editor import ImageEditor
# from optimization.arguments import get_arguments
from img_process import extract_black_contour, extract_color_strokes, extract_sketch_and_strokes
import cv2
from PIL import Image
import numpy as np

def setup_model():    
    init_setting = { 'prompt': '', 'init_sketch': '', 'init_stroke': '', 'mask': None, 'range_t': 20, 
        'sketch_guidance_scale': 1.0, 'stroke_guidance_scale': 1.0, 'realism_scale': 1.0, 'partial_edit': False, 
        'skip_timesteps': 0, 'local_clip_guided_diffusion': False, #'timestep_respacing': '100', 
        'model_output_size': 256, 'aug_num': 8,  'mode': 1,
        'clip_guidance_lambda': 1000, 'range_lambda': 50, 'lpips_sim_lambda': 1000, 'l2_sim_lambda': 10000,
        'background_preservation_loss': False, #'invert_mask': False, 'enforce_background': True, 
        'seed': 404, 'gpu_id': 0, 'output_path': 'output', 
        'output_file': 'output.png', 'iterations_num': 1, 'batch_size': 1, 'save_video': False, 'export_assets': False}
    args = argparse.Namespace(**init_setting)
    image_editor = ImageEditor(args)
    return image_editor, args

def evaluate(binary_img, image_editor, args):
    # image_editor, args = setup_model()
    print('receive input!')

    args.input = binary_img
    stroke_img = Image.open(binary_img).convert('RGB') 
    opencv_stroke = np.array(stroke_img)
    opencv_stroke = opencv_stroke[:, :, ::-1].copy() # RGB to BGR

    os.makedirs(os.path.join(os.getcwd(), 'input_example'), exist_ok=True)
    
    save_sketch_name = 'input_example/tmp_sketch' #+ args.init_stroke.split('/')[-1][:-4]
    save_stroke_name = 'input_example/tmp_stroke' #+ args.init_stroke.split('/')[-1][:-4]
        
    if image_editor.args.mode == 1:
        args.init_sketch, args.init_stroke = extract_sketch_and_strokes(opencv_stroke, 
            save_sketch_name, save_stroke_name
        )
    else:
        args.init_sketch = extract_black_contour(opencv_stroke, save_sketch_name) # get the sketch path
        args.init_stroke = extract_color_strokes(opencv_stroke, save_stroke_name)

    return image_editor.edit_image_by_prompt()


# static
class Model:
    image_editor, args = None, None

@app.before_first_request
def load_model():
    Model.image_editor, Model.args = setup_model()
    app.logger.info("Initialized Flask logger handler")

def setup_scale(r, sk, st):
    Model.args.realism_scale, Model.args.stroke_guidance_scale, Model.sketch_guidance_scale = \
        float(r), float(sk), float(st)

def setup_mode(mode):
    if mode == '1': # normal
        Model.args.mode = 1
        Model.args.partial_edit = False
        Model.args.range_t = 0
    elif mode == '2': # local editing
        Model.args.mode = 2
        Model.args.partial_edit = False
        Model.args.range_t = 100
    else: # partial 
        Model.args.mode = 3
        Model.args.partial_edit = True


@app.route('/generate', methods=['POST'])
def get_image():
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)#.decode('base64')
    # im = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    image_byte = io.BytesIO(base64.b64decode(image_data))
    # print(type(image_binary))

    setup_mode(request.values['mode'])
    setup_scale(request.values['realism'], request.values['sketch'], request.values['stroke'])
    pred_image = evaluate(image_byte, Model.image_editor, Model.args)

    img_byte = io.BytesIO()
    pred_image.save( img_byte, format('png'))
    image_binary = img_byte.getvalue()
    
    # save it 
    # THESE THREE LINE TO CHECK THE RESPONSE STATUS
    im = Image.open(image_byte).convert('RGB')
    im.save('tmp.png')
    # with open("tmp.png", 'rb') as bites:
    #     image_binary = bites.read()
    
    image = base64.b64encode(image_binary).decode("utf-8")
    return jsonify({'status': True, 'image': image})


@app.route('/')
def index():
    return current_app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(threaded=True, host='127.0.0.1')
