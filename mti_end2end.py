import os
import glob
import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from ARX import *
from ARFDecoder import ARFDecoder
from utils import *

class LoadData():
    def __init__(self, file, blockSize=5, skip=5):
        self.arf = ARFDecoder(file)
        frames = [self.arf.readFrame(i) for i in range(self.arf.fileInfo["numFrames"])]
        self.frames = []
        tmp = []
        for idx in range(0, len(frames), skip):
            frame = frames[idx]
            if len(tmp) == blockSize:
                self.frames.append(tmp.copy())
                tmp = []
            else:
                tmp.append(frame)

        self.idx = 0

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.frames):
            raise StopIteration

        truth = []
        data = np.dstack(self.frames[self.idx])
        self.idx +=1
        return data, truth

def register_frames(cube, fixedIdx):
    for idx in range(cube.shape[-1]):
        if idx == fixedIdx:
            continue
        cube[:,:,idx], _ = alignImages(cube[:,:,idx], cube[:,:,fixedIdx])
    return

def runTopHat(image, kernelSize=7):
    # Defining the kernel to be used in Top-Hat
    filterSize = (kernelSize, kernelSize)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        filterSize)

    # Applying the Black-Hat operation
    image = cv2.morphologyEx(image,
                            cv2.MORPH_TOPHAT,
                            kernel)
    return image

def runOpticalFlow(cube):
    previous = cube[:,:,0]
    flow = np.zeros((cube.shape[0], cube.shape[1], 2), dtype=np.float32)
    for i in range(1, cube.shape[-1]):
        next = cube[:,:,i]
        flow = cv2.calcOpticalFlowFarneback(previous, next, flow, pyr_scale=0.5, levels=2, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        previous = next

    return flow

# def getBoxesFromARXOld(arx):
#     boxes = []
#     windowW, windowH = 40, 20
#     rows, cols, score = fastPdet2(arx, 10)
#     aah = np.std(score) * np.sqrt(6) / np.pi
#     cent = np.mean(score) - aah * 0.577216649
#     score = (score - cent) / aah
#     score = (score - np.mean(score)) / np.std(score)
#     score = 1 / (1 + np.exp(-score))
#     for i in range(len(rows)):
#         r, c = rows[i], cols[i]
#         xmin = c - windowW // 2
#         ymin = r - windowH // 2
#         xmax = c + windowW // 2
#         ymax = r + windowH // 2
#         boxes.append(BoundingBox(xmin, ymin, xmax, ymax, score=score[i]))
#     return boxes

def getBoxesFromARX(arx):
    boxes = []
    thresh = cv2.threshold(arx, arx.mean(), arx.max(), cv2.THRESH_BINARY)[1]
    thresh = thresh.astype(np.uint8)
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # hierarchy = heirarchy[0] if len(heirarchy) > 0 else []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        currScore = arx[y:y+h, x:x+w].max()/arx.max()
        boxes.append(BoundingBox(x, y, x+w, y+h, score=currScore)) # TODO: determine score differently?
    return boxes

def updateTracker(bboxes, tracker, cubeH, cubeW):
    adjusted = []
    boxes = np.asarray([box.getBox() for box in bboxes])
    tracked = tracker.update(boxes, [cubeH, cubeW], [cubeH, cubeW])
    for idx, track in enumerate(tracked):
        track.scoreHistory.append(track.score)
        xmin, ymin, xmax, ymax = track.tlbr.astype(np.int32)
        adjusted.append(BoundingBox(xmin, ymin, xmax, ymax, score=track.score)) # np.mean(track.scoreHistory)))
    return adjusted

def threshArxResponse(arxResponse, threshold=0.995):
    h, w = arxResponse.shape
    normed = cv2.normalize(arxResponse, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    counts, bins = np.histogram(normed, bins=256)
    pdf = counts/(h*w)
    cdf = np.cumsum(pdf)
    idx = np.where(cdf > threshold)
    val = idx[0][0]
    binary = np.zeros_like(arxResponse)
    binary[normed > val] = 1
    return binary

def mtiChain(cube, tracker=None, blockNum=0, convolve=False, tophat=False, plotArx=False, plotFlow=False):

    cubeH, cubeW, numFrames = cube.shape

    # register frames to middle index
    register_frames(cube, numFrames // 2)

    # run arx algorithm to find potential movers
    arxResponse = arx(cube) # arx(cube)

    # apply tophat, if applicable
    if tophat:
        arxResponse = runTopHat(arxResponse)

    # apply convolution of ones, if applicable
    if convolve:
        mask = np.zeros_like(arxResponse, dtype=np.int32)
        mask[10:-10, 10:-10] = 1
        arxResponse *= mask
        arxResponse = convolve2d(arxResponse, np.ones((6, 12)), mode='same')

    # threshold arx response via histogram
    arxResponse = threshArxResponse(arxResponse)

    # run optical flow on aligned images
    opticalFlow = runOpticalFlow(cube)
    if plotFlow:
        plotOpticalFlow(opticalFlow, "Optical Flow Block {}".format(blockNum))

    # threshold arx response via magnitude of optical flow vectors
    mag, ang = cv2.cartToPolar(opticalFlow[..., 0], opticalFlow[..., 1])
    arxResponse *= mag

    if plotArx:
        plotArxResponse(arxResponse, "ARX Block {}".format(blockNum))

    # get bounding boxes from ARX response
    bboxes = getBoxesFromARX(arxResponse)

    # apply nms
    bboxes = non_max_suppression_fast(bboxes, 0.45)

    # update tracker if applicable
    if tracker:
        bboxes = updateTracker(bboxes, tracker, cubeH, cubeW)

    return bboxes

def main():
    files = glob.glob(args.data + "/**/*.arf", recursive=True)[1:]
    # files = [x for x in files if "20211210_VOGE_0000_00057" in x]

    for file in files:
        generator = LoadData(file, skip=2)
        frames = []
        tracker = None
        for idx, (data, truth) in enumerate(generator):
            bboxes = mtiChain(data.copy(), convolve=True, tracker=tracker, blockNum=idx, tophat=False, plotArx=False, plotFlow=False)
            drawn = plotBoxes(data, bboxes, thresh=0, title="Block {}".format(idx))
            frames.append(drawn.copy())
        name = file.split("/")[-1].replace(".arf", "_drawnDets.avi")
        makeVideo(frames, name)
        break

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to folder containing ARF files", required=True)
    args = parser.parse_args()

    main()