import pims, cv2, os

def cine2avi(path: str, opath: str = None, **kwargs):
    if opath is None:
        opath = os.path.splitext(path)[0] + ".avi"
    codec = kwargs.get("codec",cv2.VideoWriter_fourcc(*"XVID"))
    frame_rate = kwargs.get("frame_rate",30)
    images = pims.open(path)
    if len(images) == 0:
        raise ValueError(f"File {path} contains no images!")
    h,w = images[0].shape
    output = cv2.VideoWriter(opath, codec, frame_rate, (w,h), 0)
    if not output.isOpened():
        raise ValueError(f"Failed to open filemat {opath}!")
    
    for im in images:
        output.write(im.astype("uint8"))
    output.release()