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
    """
        Count the number of images in the CSQ/SEQ file by counting the number of blocks
        between the FFF Header

        Inputs:
            path : File path
        
        Returns number of images
    """
    data = open(path,'rb').read().split(FFF_HEADER)
    # first section is always empty a can be ignored
    data.pop(0)
    return len([d.split(JPG_LS_HEADER)[1] for d in data])


def export_invalid_images(path: str, export_folder: str) -> int:
    """
        Export the invalid images found in the CSQ/SEQ file

        Sometimes when decoding, invalid images are encountered. This function is for
        exporting those images for inspecting and debuggging.

        Inputs:
            path: Input file path
            export_folder: Folder to save the invalid frames to

        Returns number of invalid frames
    """
    data = open(path,'rb').read().split(FFF_HEADER)
    data.pop(0)
    count = 0
    for i, d in enumerate(data):
        # split by JPG-LS header
        temp = JPG_LS_HEADER + (d.split(JPG_LS_HEADER)[1].split(JPG_LS_END)[0] + JPG_LS_END)
        # convert to image
        try:
            img = Image.open(io.BytesIO(temp))
            np.array(img)
        except OSError:
            open(os.path.join(export_folder, f"invalid-{i:05}.bin"), 'wb').write(temp)
            count += 1
    return count


def export_jpegls_images(path: str, export_folder: str) -> int:
    """
        Iterate over the SEQ/CSQ file, decode and export the valid images to the target folder

        NOTE: Not all CSQ files use JPEG-LS encoding for the images. Other files have been found to use TIFF.
        It is recommended to use a tool like exiftool to check the file header.

        The files are written as 16-bit JPEG-LS images.

        Inputs:
            path : File path to CSQ/SEQ file
            export_folder : Folder to export the images to

        Returns number of exported valid images
    """
    data = open(path,'rb').read().split(FFF_HEADER)
    data.pop(0)
    count = 0
    for i, d in enumerate(data):
        # split by JPG-LS header
        temp = JPG_LS_HEADER + (d.split(JPG_LS_HEADER)[1].split(JPG_LS_END)[0] + JPG_LS_END)
        # convert to image
        try:
            Image.open(io.BytesIO(temp))
            open(os.path.join(export_folder, f"image-{i:06d}.jpegls"),'wb').write(temp)
            count += 1
        except OSError:
            print(f"Image {i} is invalid!")
    return count


def export_csq_to_video(path: str, export_path: str):
    """
        Iterate over the SEQ/CSQ file, decode and export the valid images to a video file

        NOTE: Not all CSQ files use JPEG-LS encoding for the images. Other files have been found to use TIFF.
        It is recommended to use a tool like exiftool to check the file header.

        The images are originally encoded as 16-bit JPEG-LS images. To write it to OpenCV videos, the image are
        converted to an 8-bit grayscale image since I don't want to deal with finding one of the cases where 16-bit
        encoding works

        Inputs:
            path : File path to CSQ/SEQ file
            export_folder : Folder to export the images to

        Returns number of exported valid images
    """
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