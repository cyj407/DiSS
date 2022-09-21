"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os
from pickle import FALSE

import numpy as np
import torch
import torch as th
import torch.distributed as dist
from guided_diffusion.guided_diffusion.image_datasets import load_sketch, load_data, load_data_sketchstroke
from guided_diffusion.guided_diffusion import dist_util
from guided_diffusion.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from PIL import Image

def save_tensor(im_data, image_dir, image_name):
    im = tensor2im(im_data)
    save_path = os.path.join(image_dir, str(image_name)) #+ '.png'
    save_image(im, save_path)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    # h, w, _ = image_numpy.shape

    # if aspect_ratio is None:
    #     pass
    # elif aspect_ratio > 1.0:
    #     image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    # elif aspect_ratio < 1.0:
    #     image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)
    
def save_npz_img(name, img_array=None, path=None):

    if path is not None:
        data = np.load(path)
        img_array = data['arr_0']

    print(img_array.shape)
    img_array = np.squeeze(img_array)
    print(img_array.shape)

    im = Image.fromarray(img_array)
    # this might fail if `img_array` contains a data type that is not supported by PIL,
    # in which case you could try casting it to a different dtype e.g.:
    # im = Image.fromarray(img_array.astype(np.uint8))
    im.save(name)
    # im.show()
    
def postprocessing(image):
    image = ((image + 1) * 127.5).clamp(0, 255).to(th.uint8)
    image = image.permute(0, 2, 3, 1)
    image = image.contiguous()
    return image
    
def main():
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    print(torch.cuda.current_device())
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    
    print(args_to_dict(args, model_and_diffusion_defaults().keys()))
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    
    root = os.path.join(os.getcwd(), args.save_name)
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, 'test_stroke'), exist_ok=True)
    os.makedirs(os.path.join(root, 'test_sketch'), exist_ok=True)
    os.makedirs(os.path.join(root, 'test_input'), exist_ok=True)

    import cv2
    from img_process import extract_black_contour, extract_color_strokes, extract_sketch_and_strokes

    paint = cv2.imread(os.path.join(os.getcwd(), args.input_image))
    # input == comb
    cv2.imwrite(os.path.join(root, 'test_input', 'input_comb.png'), paint)
    
    extract_color_strokes(paint, os.path.join(root, 'test_stroke', 'input'), save_mask=False)
    extract_black_contour(paint, os.path.join(root, 'test_sketch', 'input'))
    
    sketch = load_sketch(
        data_dir=os.path.join(root, 'test_sketch'),
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )
    
    stroke = load_data(
        data_dir=os.path.join(root, 'test_stroke'),
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )
    
    comb = load_data(
        data_dir=os.path.join(root, 'test_input'),
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    sketch_input, _ = next(iter(sketch))
    sketch_input = sketch_input.to(dist_util.dev())
        
    stroke_input, _ = next(iter(stroke))
    stroke_input = stroke_input.to(dist_util.dev())
    
    comb_input, _ = next(iter(comb))
    comb_input = comb_input.to(dist_util.dev())

    for num in range(args.num_samples):
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        
        unconditional_sketch = torch.full(sketch_input.shape, 127.5/127.5-1)
        unconditional_sketch = unconditional_sketch.to(dist_util.dev())
        
        unconditional_stroke = torch.full(stroke_input.shape, 127.5/127.5-1)
        unconditional_stroke = unconditional_stroke.to(dist_util.dev())

        
        noise = th.randn((args.batch_size, 3, args.image_size, args.image_size), device=dist_util.dev())
        for sketch_guidance_scale in [1]:
            for stroke_guidance_scale in [1]:
                # Create a classifier-free guidance sampling function
                def model_fn(x_t, sketch, stroke, ts):
                    uncond_model_out = model(x_t, unconditional_sketch, unconditional_stroke, ts)
                    sketch_model_out = model(x_t, sketch, unconditional_stroke, ts)
                    stroke_model_out = model(x_t, unconditional_sketch, stroke, ts)

                    out = uncond_model_out \
                        + sketch_guidance_scale * (sketch_model_out-uncond_model_out) \
                        + stroke_guidance_scale * (stroke_model_out-uncond_model_out) 
                    return out


                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    model_fn,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    sketch = sketch_input,
                    stroke = stroke_input,
                    noise = noise,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    device = dist_util.dev(),
                    realism_scale=1.0,
                    range_t = 100,
                    partial_edit=True,
                )

                sample = postprocessing(sample)
                save_npz_img(
                    name=os.path.join(os.getcwd(), args.save_name, 'trans{}_sketch_{}_stroke_{}.jpg'.format(num, sketch_guidance_scale, stroke_guidance_scale)), 
                    img_array=sample.cpu().numpy())


    dist.barrier()


def create_argparser():
    defaults = dict(
        encode_step=250,
        clip_denoised=True,
        input_image = './test-examples/partial-example.png',
        save_name = 'partial-output',  # save output dir name
        num_samples=20,
        batch_size=1,
        use_ddim=False,
        # model_path = "/eva_data4/models/lhq_size512_step1000_finetune/ema_0.9999_330000.pt",
        # model_path = "/eva_data4/models/flower_foreground_finetune/ema_0.9999_400000.pt",
        # model_path = "/eva_data4/models/afhq_all_size512_step1000_finetune/ema_0.9999_400000.pt",
        # model_path="/eva_data5/models/cat_size512_step1000_finetune/ema_0.9999_400000.pt",
        model_path='./checkpoints/cat512.pt',
        schedule_sampler="uniform",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
