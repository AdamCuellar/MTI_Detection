import os
import sys
import numpy as np

class ARFDecoder():
    def __init__(self, arfPath):
        self.arfPath = arfPath
        self.byteSwapFn = self.swapBytesLittle if sys.byteorder == 'little' else self.swapBytesBig
        self.fid = None
        try:
             self.fid = open(arfPath, "rb")
        except:
            print("Error reading file: ", arfPath)

        self.fileInfo = self.getInfo()

    def swapBytesLittle(self, frame):
        frame.byteswap(inplace=True)
        return

    def swapBytesBig(self, frame):
        # arf files should be in big endian
        return

    def getInfo(self):
        headerDict = None
        if self.fid is None: return None

        header = np.fromfile(self.fid, dtype=np.dtype(np.uint32), count=8)
        self.byteSwapFn(header)

        # check the arf file is valid
        if header[0] == 3149642413:
            headerDict = {"magic_num":header[0], "version":header[1], "height":header[2], "width":header[3],
                "type":header[4], "numFrames":header[5], "offset":header[6], "flags":header[7]}

        return headerDict

    def getHumanReadableType(self, type, flags):
        types = ["GRAY8", "GRAY16_SIGNED", "GRAY16_SIGNED", "GRAY32_INT",
                 "GRAY16_UNSIGNED", "CHECKBIT", "GRAY32_FLOAT", "GRAY64_FLOAT"]

        if "CHECKBIT" in types[type]:
            if flags & 128 == 128:
                return "ARGB"
            else:
                return "GRAY32_UNSIGNED"
        else:
            return types[type]

    def getFrameRetrievalInfo(self):
        type = self.fileInfo["type"]
        dataTypeMap = { 0 : np.uint8,
                        1 : np.int16,
                        2 : np.int16,
                        3 : np.int16,
                        4 : np.int32,
                        5 : np.uint16,
                        6 : np.uint32,
                        7 : np.float32,
                        8 : np.double }

        return dataTypeMap[type]

    def readFrame(self, frameNum):
        bitType = np.dtype(self.getFrameRetrievalInfo())
        self.fid.seek(self.fileInfo['offset'] + (frameNum * bitType.itemsize * self.fileInfo['width'] * self.fileInfo['height']), 0)

        frame = np.fromfile(self.fid, dtype=bitType, count=(self.fileInfo["width"] * self.fileInfo["height"]))

        if frame.size < (self.fileInfo["width"] * self.fileInfo["height"]): return np.empty((0,0)) # reached end of file

        self.byteSwapFn(frame)
        frame.shape = (self.fileInfo["height"], self.fileInfo["width"])

        return frame