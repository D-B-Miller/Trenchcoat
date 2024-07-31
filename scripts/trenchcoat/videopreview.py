import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

def videoActionLabel(path,preview=20,fout='actions.txt'):
    # open video and check that it's opened
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file {path}")
    # read first frame
    ret,frame = cap.read()
    if not ret:
        raise ValueError("Failed to read first frame!")
    # get number of frames in file
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # setup actions list
    actions = nf*['',]
    # set preview height to 34
    prev_size=34
    # scale to aspect ratio of original shape
    asp = frame.shape[0]/frame.shape[1]
    prev_shape=(prev_size,int(prev_size*asp))

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(nrows=2,sharex=False)
    for aa in ax:
        aa.axis('off')
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.1, bottom=0.25)

    # Make a horizontal slider to control frame number
    axframe = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    
    frame_slider = Slider(
        ax=axframe,
        label='Frame',
        valmin=0,
        valmax=nf-1,
        valstep=1,
        valfmt="%d"
    )
    # text box for current action
    axbox = fig.add_axes([0.1, 0.2, 0.8, 0.075])
    text_box = TextBox(axbox, "Action")
    # button to apply action to current + preview frames
    baxprev = fig.add_axes([0.1, 0.1, 0.3, 0.05])
    bprev = Button(baxprev, 'Apply To Preview')
    # button to apply action to current frame only
    baxsingle = fig.add_axes([0.6, 0.1, 0.3, 0.05])
    bsingle = Button(baxsingle, 'Apply To Single')

    global fi, curr_action
    # current action
    curr_action = ''
    # current frame number
    fi = 0
    # method when slider is moved
    def update(val):
        global fi, curr_action
        # get target frame
        fi = int(frame_slider.val)
        # move frame pointer
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
        # read frame
        ret,frame = cap.read()
        if not ret:
            print(f"Failed to read frame {fi}!")
            return
        frame = frame[...,::-1]
        # display on first axes
        ax[0].imshow(frame)
        # downsample current frame to preview size
        prevs = [cv2.resize(frame,prev_shape,interpolation=cv2.INTER_AREA),]
        # collect remaining frames for preview
        for fic in range(preview-1):
            ret,frame = cap.read()
            if not ret:
                break
            frame = frame[...,::-1]
            prevs.append(cv2.resize(frame,prev_shape,interpolation=cv2.INTER_AREA))
        # fill remaining slot with black frames
        if len(prevs) != preview:
            prevs += abs(preview-len(prevs))*[np.zeros([prev_shape[1],prev_shape[0],3],np.uint8),]
        # stack + display preview
        ax[1].imshow(cv2.hconcat(prevs))
        fig.canvas.draw_idle()

    # set current action
    def set_action(val):
        global curr_action
        curr_action = val

    # button handlers
    def apply_preview(event):
        global fi, curr_action
        # action has to be non-empty
        if not curr_action:
            return
        if event.button==1:
            # if user clicked apply to preview
            if (event.inaxes == baxprev):
                for ii in range(fi,fi+preview+1):
                    actions[ii] = curr_action
                print("applied to preview!")
            # if user clicked applied to single
            elif (event.inaxes == baxsingle):
                actions[fi] = curr_action
                print("applied to current!")
    # assign handlers
    frame_slider.on_changed(update)
    text_box.on_submit(set_action)
    bprev.on_clicked(apply_preview)
    bsingle.on_clicked(apply_preview)
    
    update(0)
    plt.show()
    # if user gave filename for list of actions
    # write actions to file
    if fout:
        open(fout,'w').write('\n'.join(actions))
        print(f"Written actions to {fout}")
    return actions

if __name__ == "__main__":
    videoActionLabel("lsbu_doe_powder_stripes_0002_hot-pixel-marked.avi")
