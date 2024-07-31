#import pillow_jpls
from PIL import UnidentifiedImageError
from PIL import Image
import numpy as np
import io
import cv2
import os

# constants used for splitting files
JPG_LS_HEADER = b'\xFF\xD8\xFF\xF7'
JPG_LS_END = b'\xFF\xD9'

FFF_HEADER = b'\x46\x46\x46\x00'


def count_images(path: str) -> int:
    data = open(path,'rb').read().split(FFF_HEADER)
    data.pop(0)
    return len([d.split(JPG_LS_HEADER)[1] for d in data])


def export_invalid_images(path: str, export_path: str) -> int:
    data = open(path,'rb').read().split(FFF_HEADER)
    data.pop(0)
    for i, d in enumerate(data):
        # split by JPG-LS header
        temp = JPG_LS_HEADER + (d.split(JPG_LS_HEADER)[1].split(JPG_LS_END)[0] + JPG_LS_END)
        # convert to image
        try:
            img = Image.open(io.BytesIO(temp))
            np.array(img)
        except OSError:
            open(os.path.join(export_path, f"invalid-{i:05}.bin"), 'wb').write(temp)


def export_jpegls_images(path: str, export_path: str):
    data = open(path,'rb').read().split(FFF_HEADER)
    data.pop(0)
    for i, d in enumerate(data):
        # split by JPG-LS header
        temp = JPG_LS_HEADER + (d.split(JPG_LS_HEADER)[1].split(JPG_LS_END)[0] + JPG_LS_END)
        # convert to image
        try:
            Image.open(io.BytesIO(temp))
            open(os.path.join(export_path, f"image-{i:06d}.jpegls"),'wb').write(temp)
        except OSError:
            print(f"Image {i} is invalid!")


def export_csq_to_video(path: str, export_path: str):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = None
    # split by FFF0
    data = open(path,'rb').read().split(FFF_HEADER)
    # first part is always 0 bytes
    data.pop(0)
    for d in data:
        # split by JPG-LS header
        temp = JPG_LS_HEADER + (d.split(JPG_LS_HEADER)[1].split(JPG_LS_END)[0] + JPG_LS_END)
        # convert to image
        try:
            img = Image.open(io.BytesIO(temp))
            img_np = np.array(img)
            img_np = (img_np >> 8).astype("uint8")

            if video is None:
                video = cv2.VideoWriter(export_path,
                                fourcc,
                                30,
                                (img_np.shape[1], img_np.shape[0]),
                                0)
                if not video.isOpened():
                    raise ValueError("Failed to open video file!")
            video.write(img_np)
        except OSError:
            continue
    if video is not None:
        video.release()


class CSQImageSet:
    """
        Class for extracting images from recorded CSQ files

        This assumes the following
            - Images are saved as JPEG LS (losseless JPEG)

        The CSQ file is a series of Radiometric JPEG LS files stacked on top of each other
        The files are 16-bit grayscale images. Invalid images are discarded

        To extract the FLIR metadata, use exiftool to read the values

        The images are parsed by splitting by the FFF string that appears and then by the JPG_LS_HEADER to find the image data.
        The image data is read using Pillow (which provides a method to convert a byte array to an image) and converted to an 8-bit
        numpy array. If the user want to keep the 16-bit images, set keep_16 to True in constructor.

        The other methods are designed to help export the images more easily

        Example:
            # load images
            csq = CSQImageSet(path)
            # get number of images
            len(csq)
            # export the images to a video
            csq.to_video("output.avi")
    """
    def __init__(self, path: str, keep_16: bool = False):
        self.source = path
        self.image_stack = []
        self.num_invalid_images = 0
        # split by FFF0
        data = open(path,'rb').read().split(FFF_HEADER)
        # first part is always 0 bytes
        data.pop(0)
        for d in data:
            # split by JPG-LS header
            temp = JPG_LS_HEADER + (d.split(JPG_LS_HEADER)[1].split(JPG_LS_END)[0] + JPG_LS_END)
            # convert to image
            try:
                img = Image.open(io.BytesIO(temp))
                img_np = np.array(img)
                if not keep_16:
                    img_np = (img_np >> 8).astype("uint8")
                self.image_stack.append(img_np)
            except OSError:
                self.num_invalid_images += 1

    # write images to video file
    def to_video(self, path: str):
        """
            Export the images to a video file

            Uses MJPG codec and converts 16-bit images to 8-bit images

            Inputs:
                path : Output path for images
        """
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video = cv2.VideoWriter(path,
                                fourcc,
                                30,
                                (self.image_stack[0].shape[1], self.image_stack[0].shape[0]),
                                0)
        if not video.isOpened():
            raise ValueError("Failed to open video file!")
        for img in self.image_stack:
            # convert to 8-bit for writing
            if img.dtype == np.uint16:
                img = (img >> 8).astype("uint8")
            video.write(img)
        video.release()

    # get datatype of images
    def get_dtype(self) -> np.dtype:
        return self.image_stack[0].dtype

    # get number of images loaded
    def __len__(self) -> int:
        return len(self.image_stack)
    
    def num_images(self) -> int:
        return len(self.image_stack)
    
    # method to iterate over images in file
    # returns Pillow classes
    @classmethod
    def to_iter(cls, path: str):
        cls.source = path
        cls.image_stack = []
        # split by FFF0
        data = open(path,'rb').read().split(FFF_HEADER)
        data.pop(0)
        for d in data:
            # split by JPG-LS header
            temp = JPG_LS_HEADER + d.split(JPG_LS_HEADER)[1]
            # convert to image
            try:
                img = Image.open(io.BytesIO(temp))
                img_np = np.array(img)
                yield img_np
            except UnidentifiedImageError:
                break

if __name__ == "__main__":
    PATH = r"D:\Work\Plasma-Spray-iCoating\scripts\trenchcoat\src\sheffield_doe_flowrate_gasrate_0002.csq"
    #export_invalid_images(PATH, r"D:\Work\Plasma-Spray-iCoating\scripts\trenchcoat\src\pillow_export\invalid")
    #csq = CSQImageSet(PATH)
    #print(f"{len(csq)} images")
    #print(f"{csq.num_invalid_images} images")
    export_csq_to_video(PATH, r"D:\Work\Plasma-Spray-iCoating\scripts\trenchcoat\src\pillow_export\test.avi")