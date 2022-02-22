import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt

class BoundingBox():
    def __init__(self, xmin, ymin, xmax, ymax, score=0.):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.asList = [self.xmin, self.ymin, self.xmax, self.ymax, self.score]

    def getList(self):
        return self.asList

    def getBox(self):
        return np.asarray(self.asList)

class TrackerArgs(object):
    def __init__(self):
        self.track_thresh = 0.4
        self.track_buffer = 5
        self.match_thresh = 0.9
        self.min_box_area = 10
        self.mot20 = False

def alignImages(im1, im2, max_features=500, good_match_percent=0.15, norm_bit=True, turnGray=False,
                warpMethod=cv2.RANSAC):
    """ Aligns image 1 to image 2, based off https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
        Inputs:
            im1 - misaligned image
            im2 - Image to align im1 too
            max_features - max number of features for alignment
            good_match_percent - percent of good features to choose
            norm_bit - boolean to convert images to 8 bit and normalize
            turnGray - boolean to convert images to grayscale
            warpMethod - method for transformation via opencv (best on SW data is cv2.RHO)
        Returns:
            im1Reg - Registered image 1
            h - calculated homography matrix
    """

    # Convert images to grayscale
    if turnGray:
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        im1Gray = copy.deepcopy(im1)
        im2Gray = copy.deepcopy(im2)

    if norm_bit:
        im1Gray = cv2.normalize(im1Gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        im2Gray = cv2.normalize(im2Gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Detect Sift features and compute descriptors.
    sift = cv2.SIFT_create(max_features)
    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    increment = 0.1
    while numGoodMatches < 4 and good_match_percent < 1.0:
        good_match_percent += increment
        numGoodMatches = int(len(matches) * good_match_percent)

    assert numGoodMatches >= 4, "Couldn't register images!"

    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, warpMethod)
    if h is None:
        return np.zeros_like(im1), None

    # Use homography
    height, width = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=np.asscalar(im1.min()))

    return im1Reg, h


def plotOpticalFlow(flow, title=""):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    plt.imshow(bgr)
    plt.title(title)
    plt.show()
    return


def plotBoxes(data, boxes, thresh=0., title="", show=False):
    # middle = data[..., data.shape[-1] // 2]
    middle = data[..., -1]
    middle = np.dstack([middle, middle, middle])
    middle = cv2.normalize(middle, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    for box in boxes:
        if box.score > thresh:
            middle = cv2.rectangle(middle, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 2)

    if show:
        plt.imshow(middle)
        plt.title(title)
        plt.show()
    return middle


def plotArxResponse(arx, title=""):
    plt.imshow(arx, cmap='gray', extent=[0, 1, 0, 1])
    plt.title(title)
    plt.show()
    return


def fastPdet2(x, nDets, windowH=20, windowW=40):
    h, w = x.shape
    r, c, score = [], [], []
    vmin = np.min(x)
    halfW = windowW // 2
    halfH = windowH // 2
    for i in range(nDets):
        currScore = np.max(x)
        idx = np.where(x == currScore)
        row, col = idx[0][0].item(), idx[1][0].item()
        r.append(row)
        c.append(col)
        score.append(currScore)
        r1 = max(row - halfH, 0)
        r2 = min(r1 + windowH, h)
        r1 = r2 - windowH
        c1 = max(col - halfW, 0)
        c2 = min(c1 + windowW, w)
        c1 = c2 - windowW
        x[r1:r2, c1:c2] = np.ones((windowH, windowW)) * vmin
    return r, c, np.asarray(score)


def makeVideo(frames, name):
    h, w, c = frames[0].shape
    outVideo = cv2.VideoWriter(name,
                               cv2.VideoWriter_fourcc(*'DIVX'), 1, (w, h))
    for frame in frames:
        outVideo.write(frame)

    outVideo.release()
    return

# Malisiewicz et al.
def non_max_suppression_fast(bboxes, overlapThresh):
    boxes = [box.getBox() for box in bboxes]
    boxes = np.asarray(boxes)
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return [bboxes[i] for i in pick]
