import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


# load sketch-stroke image 
def load_data_sketchstroke(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    # image
    all_image_files = sorted(_list_image_files_recursively(data_dir))
    
    # sketch
    sketch_dir = data_dir[:-1] + '_sketch/'
    all_sketch_files = sorted(_list_image_files_recursively(sketch_dir))
    
    # stroke
    stroke_dir = data_dir[:-1] + '_stroke/'
    # stroke_dir = "/eva_data1/Mavis/gd_sketch_stroke/test_stroke"
    all_stroke_files = sorted(_list_image_files_recursively(stroke_dir))
    
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        
    dataset = ImageDataset_sketchstroke(
        image_size,
        all_image_files,
        all_sketch_files,
        all_stroke_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    #return dataset
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader
        
# load sketch-image fine tune
def load_data_finetune(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    # image
    all_image_files = sorted(_list_image_files_recursively(data_dir))
    # sketch
    sketch_dir = data_dir[:-1] + '_sketch/'
    all_sketch_files = sorted(_list_image_files_recursively(sketch_dir))
    # stroke
    stroke_dir = data_dir[:-1] + '_stroke/'
    all_stroke_files = sorted(_list_image_files_recursively(stroke_dir))
    
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
        
    dataset = ImageDataset_finetune(
        image_size,
        all_image_files,
        all_sketch_files,
        all_stroke_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

# load sketch        
def load_sketch(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=True,
    random_crop=False,
    random_flip=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = SketchDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader
        


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict



# sketch-stroke-image pair
class ImageDataset_sketchstroke(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        sketch_paths,
        stroke_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
    ):
        super().__init__()
        self.resolution = resolution
        
        self.local_images = image_paths[shard:][::num_shards]
        self.local_sketches = sketch_paths[shard:][::num_shards]
        self.local_strokes = stroke_paths[shard:][::num_shards]
        
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        image_path = self.local_images[idx]
        sketch_path = self.local_sketches[idx]
        stroke_path = self.local_strokes[idx]
        
        flip_or_not = False
        if self.random_flip and random.random() < 0.5:
            flip_or_not = True
            
        # image
        with bf.BlobFile(image_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            image_arr = random_crop_arr(pil_image, self.resolution)
        else:
            image_arr = center_crop_arr(pil_image, self.resolution)

        if flip_or_not:
            image_arr = image_arr[:, ::-1]

        image_arr = image_arr.astype(np.float32) / 127.5 - 1
        
        # sketch
        with bf.BlobFile(sketch_path, "rb") as f:
            pil_sketch = Image.open(f)
            pil_sketch.load()
        pil_sketch = pil_sketch.convert("L")

        if self.random_crop:
            sketch_arr = random_crop_arr(pil_sketch, self.resolution)
        else:
            sketch_arr = center_crop_arr(pil_sketch, self.resolution)

        if flip_or_not:
            sketch_arr = sketch_arr[:, ::-1]
        
        sketch_arr = sketch_arr.astype(np.float32) / 127.5 - 1
        sketch_arr = np.expand_dims(sketch_arr, axis=2)
        
        # stroke
        with bf.BlobFile(stroke_path, "rb") as f:
            pil_stroke = Image.open(f)
            pil_stroke.load()
        pil_stroke = pil_stroke.convert("RGB")

        if self.random_crop:
            stroke_arr = random_crop_arr(pil_stroke, self.resolution)
        else:
            stroke_arr = center_crop_arr(pil_stroke, self.resolution)

        if flip_or_not:
            stroke_arr = stroke_arr[:, ::-1]

        stroke_arr = stroke_arr.astype(np.float32) / 127.5 - 1
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        #print(image_arr.shape)
        #print(sketch_arr.shape)
        return np.transpose(image_arr, [2, 0, 1]), np.transpose(sketch_arr, [2, 0, 1]), np.transpose(stroke_arr, [2, 0, 1]), out_dict    

# fine tune for classifier-free guidance
class ImageDataset_finetune(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        sketch_paths,
        stroke_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
    ):
        super().__init__()
        self.resolution = resolution
        
        self.local_images = image_paths[shard:][::num_shards]
        self.local_sketches = sketch_paths[shard:][::num_shards]
        self.local_strokes = stroke_paths[shard:][::num_shards]
        
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        image_path = self.local_images[idx]
        sketch_path = self.local_sketches[idx]
        stroke_path = self.local_strokes[idx]
        
        flip_or_not = False
        if self.random_flip and random.random() < 0.5:
            flip_or_not = True
            
        # image
        with bf.BlobFile(image_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            image_arr = random_crop_arr(pil_image, self.resolution)
        else:
            image_arr = center_crop_arr(pil_image, self.resolution)

        if flip_or_not:
            image_arr = image_arr[:, ::-1]

        image_arr = image_arr.astype(np.float32) / 127.5 - 1
        
        # sketch
        with bf.BlobFile(sketch_path, "rb") as f:
            pil_sketch = Image.open(f)
            pil_sketch.load()
        pil_sketch = pil_sketch.convert("L")

        if self.random_crop:
            sketch_arr = random_crop_arr(pil_sketch, self.resolution)
        else:
            sketch_arr = center_crop_arr(pil_sketch, self.resolution)

        if flip_or_not:
            sketch_arr = sketch_arr[:, ::-1]
        
        # for unconditional generation without sketch
        if random.random() < 0.3:
            m = np.max(sketch_arr) / 2
            sketch_arr.fill(m)
            #print(sketch_arr)
        
        sketch_arr = sketch_arr.astype(np.float32) / 127.5 - 1
        sketch_arr = np.expand_dims(sketch_arr, axis=2)
        
        # stroke
        with bf.BlobFile(stroke_path, "rb") as f:
            pil_stroke = Image.open(f)
            pil_stroke.load()
        pil_stroke = pil_stroke.convert("RGB")

        if self.random_crop:
            stroke_arr = random_crop_arr(pil_stroke, self.resolution)
        else:
            stroke_arr = center_crop_arr(pil_stroke, self.resolution)

        if flip_or_not:
            stroke_arr = stroke_arr[:, ::-1]
        
        # for unconditional generation without stroke
        if random.random() < 0.3:
            m = 255 / 2
            stroke_arr.fill(m)
        
        stroke_arr = stroke_arr.astype(np.float32) / 127.5 - 1
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        #print(image_arr.shape)
        #print(sketch_arr.shape)
        return np.transpose(image_arr, [2, 0, 1]), np.transpose(sketch_arr, [2, 0, 1]), np.transpose(stroke_arr, [2, 0, 1]), out_dict
    
# sketch
class SketchDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=False,
        unconditional=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.unconditional = unconditional

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("L")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        
        if self.unconditional:
            if random.random() < 0.1:
                m = np.max(arr) / 2
                arr.fill(m)
        
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.expand_dims(arr, axis=2)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict




def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
