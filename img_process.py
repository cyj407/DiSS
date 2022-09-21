from ast import Return
from nturl2path import pathname2url
from pickle import FALSE
from PIL import Image
import torch
from torchvision.transforms import functional as TF
import os
import cv2
import numpy as np

device = torch.device(
    "cuda:7" if torch.cuda.is_available() else "cpu"
)


def get_top_sketch(img, name="tmp.png", save=False):
    '''
    create a top partial sketch
    '''
    img = img.convert('RGB')
    pixels = img.load() # create the pixel map

    for i in range(img.size[0]): # width
        for j in range(100, img.size[1]): # height
            pixels[i,j] = (255, 255, 255)

    if save:
        img.save(name)
    return img

def black_to_white_mask(img):
    img = img.convert('RGB')

    # set white part to transparent
    data = img.getdata()
    newData = []
    for item in data:
        if max(item) < 40:
        # if min(item) > 250:
            newData.append((255, 255, 255))
        else:
            newData.append(item)
    img.putdata(newData)
    # img.save('tmp.png')

    return img


def get_binary_sketch(sketch, name):
    sketch = sketch.convert("RGB")

    # set white part to transparent
    data = sketch.getdata()
    newData = []
    for item in data:
        if max(item) < 40:
            newData.append((255, 255, 255))
        else:
            newData.append(item)
    sketch.putdata(newData)
    sketch.save(name)



def get_overlay_img(sketch, gen_img, save=False):
    gen_img = gen_img.convert("RGBA")
    sketch = sketch.convert("RGBA")

    # set white part to transparent
    data = sketch.getdata()
    newData = []
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    sketch.putdata(newData)
    sketch.putalpha(255) # 0 : transparent
    gen_img.putalpha(130) # 0 : transparent

    comp_a = Image.alpha_composite(sketch, gen_img)
    
    if save:
        comp_a.save('overlay.png')

    return comp_a

'''good'''
def extract_black_contour(img, save_name):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, image2 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY) # cat, flower = 70
    # image2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    # cv2.imwrite('black_sketch.png', image2)
    path = os.path.join(os.getcwd(), '{}_sketch.png'.format(save_name))
    cv2.imwrite(path, image2)
    return path


def grab_cut(img, save_name):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0,0,img.shape[0]-1,img.shape[1]-1)

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask2_inv = cv2.bitwise_not(mask2*255)

    out = img*mask2[:,:,np.newaxis] + mask2_inv[:,:,np.newaxis]
    path = os.path.join(os.getcwd(), '{}_stroke.png'.format(save_name))
    cv2.imwrite(path, out)
    return out, path

def edge_detect(img, save_sketch_name, save_stroke_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 100, 230) # 200
    # cv2.imwrite('tmp.jpg', canny)

    # img[canny != 0, :] = 255
    # cv2.imwrite('tmp2.jpg', img)
        
    (cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contours = img.copy()
    cv2.drawContours(img, cnts, -1, (255, 255, 255), 2)
    stroke_path = os.path.join(os.getcwd(), '{}_stroke.png'.format(save_stroke_name))
    cv2.imwrite(stroke_path, img)
    
    white = np.ones_like(img) * 255
    cv2.drawContours(white, cnts, -1, (0, 0, 0), 2)
    sketch_path = os.path.join(os.getcwd(), '{}_sketch.png'.format(save_sketch_name))
    cv2.imwrite(sketch_path, white)
    # cv2.imwrite('tmp.jpg', white)
    return stroke_path, sketch_path

def extract_sketch_and_strokes(img, save_sketch_name, save_stroke_name):
    out, path = grab_cut(img, save_stroke_name)
    stroke_path, sketch_path = edge_detect(out, save_sketch_name, save_stroke_name)
    return sketch_path, stroke_path

'''good'''
def extract_color_strokes(img, save_name, save_mask=False):

    # convert image to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # extract non-black
    thresh1 = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY)[1]
    res = cv2.bitwise_and(img, img, mask= thresh1)
    # path3 = os.path.join(os.getcwd(), '{}_stroke.png'.format(save_name))
    # cv2.imwrite( path3, res)


    # black part to white
    res2 = res.copy()
    res2[thresh1 == 0] = (255, 255, 255)
    stroke_path = os.path.join(os.getcwd(), '{}_stroke.png'.format(save_name))
    cv2.imwrite( stroke_path, res2)

    if save_mask:
        mask_path = os.path.join(os.getcwd(), '{}_mask.png'.format(save_name))
        cv2.imwrite( mask_path, res)
    else:
        mask_path = None    
    # cv2.imwrite( save_name, res)
    return stroke_path


if __name__ == '__main__':
    import numpy as np
    import os
    import cv2

    paint = cv2.imread('input_example/dior_2.jpg')
    edge_detect(paint, '')
