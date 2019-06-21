# Imports
import cv2
import os
import keras
import scipy.io
import bisect
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from slowai.util import *
from slowai.layers import *
from slowai.metrics import *
from keras.models import Model
from keras.layers import MaxPooling2D
from collections import deque
from moviepy.editor import VideoFileClip

import numpy as np

# Functions
def normalize(img):
    return np.expand_dims(np.expand_dims(((1.-img[:,:,0])-0.257)/0.288, axis=-1), axis=0)

def convertWiskStamps(wiskframe, mat):
    time = mat['frameTimeStampsWisk'][wiskframe][0]
    return bisect.bisect_left(np.squeeze(mat['frameTimeStamps']), time)

def getTrialFrames(trackedFeatures, mat):
    obsOnTimes = np.squeeze(mat['obsOnTimes'])
    obsOnFrames = []
    clipLen = len(mat['frameTimeStampsWisk'])
    otherclipLen = len(mat['frameTimeStamps'])
    beforeAfter = True
    for i in range(len(mat['frameTimeStampsWisk'])):
        if np.isnan(mat['frameTimeStampsWisk'][i][0]):
            if beforeAfter:
                mat['frameTimeStampsWisk'][i][0]=-1
            else:
                mat['frameTimeStampsWisk'][i][0]=INF
        else:
            beforeAfter = False
    beforeAfter = True
    for i in range(len(mat['frameTimeStamps'])):
        if np.isnan(mat['frameTimeStamps'][i][0]):
            if beforeAfter:
                mat['frameTimeStamps'][i][0] = -1
            else:
                mat['frameTimeStampsWisk'][i][0]=INF
        else:
            beforeAfter = False
    firstIdx = bisect.bisect(np.squeeze(mat['frameTimeStamps']), -1)
    if firstIdx == clipLen:
        firstIdx = 0
    otherfirstIdx = bisect.bisect(np.squeeze(mat['frameTimeStampsWisk']), -1)
    if otherfirstIdx == otherclipLen:
        otherfirstIdx = 0

    lastIdx = -1
    otherlastIdx = -1

    while mat['frameTimeStamps'][lastIdx][0]==INF:
        lastIdx-=1
    while mat['frameTimeStampsWisk'][otherlastIdx][0]==INF:
        otherlastIdx-=1

    otherFirstTime = mat['frameTimeStamps'][firstIdx][0]
    firstTime = mat['frameTimeStampsWisk'][otherfirstIdx][0]
    for i, obsOnTime in enumerate(obsOnTimes):
        if not (obsOnTime>min(mat['frameTimeStampsWisk'][otherlastIdx][0], mat['frameTimeStamps'][lastIdx][0]) or obsOnTime<max(firstTime, otherFirstTime)):
            obsOnFrames.append((i, bisect.bisect_left(np.squeeze(mat['frameTimeStampsWisk']), obsOnTime)))
        else:
            obsOnFrames.append((i, -1))

    obsOffTimes = np.squeeze(mat['obsOffTimes'])
    obsOffFrames = []
    for i, obsOffTime in enumerate(obsOffTimes):
        if not (obsOffTime>min(mat['frameTimeStampsWisk'][otherlastIdx][0], mat['frameTimeStamps'][lastIdx][0]) or obsOffTime<max(firstTime, otherFirstTime)):
            obsOffFrames.append((i, bisect.bisect_left(np.squeeze(mat['frameTimeStampsWisk']), obsOffTime)))
        else:
            obsOffFrames.append((i, -1))

    if len(obsOnTimes) != len(obsOffTimes):
        raise ValueError('obsOnTimes and obsOffTimes have different lengths! Please check no trials have been skipped')

    temptrialFrames = list(zip(obsOnFrames, obsOffFrames))
    trialFrames = []
    for temp in temptrialFrames:
        if temp[0][0] != temp[1][0]:
            raise ValueError('some trial got shifted somewhere, exiting')
        trialFrames.append((temp[0][0], temp[0][1], temp[1][1]))

    return trialFrames

if __name__ == '__main__':
    # Argument reading
    ap = argparse.ArgumentParser()
    ap.add_argument('sessionDir', help="path to sessions")
    ap.add_argument('-d', '--dataDir', default="data", help="desired path to output data (optional)")
    ap.add_argument('-c', '--cropmodel', default="models/cropWhiskers.h5", help="path to cropmodel (optional)")
    ap.add_argument('-t', '--timesteps', type=int, default=10, help="number of timesteps per window (optional, default 10)")
    ap.add_argument('-vs', '--valsplit', type=float, default=0.2, help="percent of frames to include in validation set (between 0 and 1, default 0.2)")
    ap.add_argument('--crop-size', type=int, default=200, help="size of cropped image (optional, default 200)")
    ap.add_argument('--mean', type=float, default=0.257, help="image mean for normalization (optional, default 0.257)")
    ap.add_argument('--std', type=float, default=0.288, help="image std for normalization (optional, default 0.288)")
    ap.add_argument('--width-offset', type=int, default=-50, help="width offset for cropping (optional, default -50)")
    ap.add_argument('--height-offset', type=int, default=0, help="height offset for croppign (optional, default 0)")
    args = vars(ap.parse_args())

    # Some variables
    INF = sys.maxsize

    # Load and assemble crop model
    crop_model = keras.models.load_model(args['cropmodel'], custom_objects={'meanDistance': meanDistance})

    x = crop_model.output
    x = Maxima2D()(x)
    x = PointCrop2D(crop_size=args['crop_size'], mean=args['mean'], std=args['std'], wOffset=args['width_offset'], hOffset=args['height_offset'])([x, crop_model.input])
    x = MaxPooling2D((2,2), padding="same", name="downsampler")(x)

    model = Model(inputs=crop_model.input, outputs=x)

    for layer in crop_model.layers:
        layer.trainable = False

    # Load session directories
    sessions = {}

    for root, path, files in os.walk(args['sessionDir']):
        if 'whiskerLabels.csv' in files:
            sessions[root[-10:]] = root

    # Process sessions
    for session in sessions:
        print("Exporting images from", session)
        clip = VideoFileClip(os.path.join(sessions[session], 'runWisk.mp4'))
        trackedFeatures = np.loadtxt(os.path.join(sessions[session], 'trackedFeaturesRaw.csv'), delimiter=',', skiprows=1)
        mat = scipy.io.loadmat(os.path.join(sessions[session], 'runAnalyzed.mat'))
        size = [clip.size[1], clip.size[0]]

        trialFrames = getTrialFrames(trackedFeatures, mat)

        labeledTrials = {}
        for line in open(os.path.join(sessions[session], 'whiskerLabels.csv')):
            elts = line.split(',')
            labeledTrials[int(elts[0])] = int(elts[1])

        frames = []

        print('Analyzing')

        for idx, framenum, endframe in tqdm(trialFrames):
            if labeledTrials[idx] == -1:
                continue
            if framenum == -1 or endframe == -1:
                continue
            features = trackedFeatures[convertWiskStamps(framenum, mat)]
            obsPos = list(map(int, [features[22], features[23]]))
            obsConf = features[24]
            nosePos = list(map(int, [features[19], features[20]]))
            image = clip.get_frame(framenum * (1 / clip.fps))
            while sum(sum(i<10 for i in image[:, (size[1]-2):, 0]))<40:
                try:
                    framenum+=1
                    features = trackedFeatures[convertWiskStamps(framenum, mat)]
                    obsPos = list(map(int,[features[22],features[23]]))
                    obsConf = features[24]
                    nosePos = list(map(int, [features[19], features[20]]))
                    image = clip.get_frame(framenum * (1 / clip.fps))
                except IndexError:
                    framenum = endframe+1
                    break
            if framenum >= endframe:
                print("Could not find obstacle within session. Skipping session...")
                continue
            minFrame = INF
            maxFrame = 0
            while (nosePos[0]-obsPos[0]<50 or obsConf!=1) and framenum < endframe:
                minFrame = min(framenum, minFrame)
                maxFrame = max(framenum, maxFrame)
                framenum+=1
                features = trackedFeatures[convertWiskStamps(framenum, mat)]
                obsPos = list(map(int, [features[22], features[23]]))
                obsConf = features[24]
                nosePos = list(map(int, [features[19], features[20]]))
            for i in range(args['timesteps']):
                maxFrame = max(framenum, maxFrame)
                framenum+=1
            if labeledTrials[idx]<=maxFrame and labeledTrials[idx]>=minFrame:
                frames.append((minFrame, labeledTrials[idx], maxFrame+1))

        width = len(str(np.max(frames)))

        print('Exporting')

        testDir = os.path.join(args['dataDir'], 'test', session)
        trainDir = os.path.join(args['dataDir'], 'train', session)

        if not os.path.exists(testDir):
            os.makedirs(testDir)
        else:
            for f in os.listdir(testDir):
                os.remove(os.path.join(testDir, f))
        if not os.path.exists(trainDir):
            os.makedirs(trainDir)
        else:
            for f in os.listdir(trainDir):
                os.remove(os.path.join(trainDir, f))

        testFile = open(os.path.join(testDir, 'frameInfo.csv'), 'w')
        trainFile = open(os.path.join(trainDir, 'frameInfo.csv'), 'w')

        for trial in tqdm(frames):
            randTrial = random.random()
            if randTrial<args['valsplit']:
                testFile.write(','.join(map(str, [trial[0], trial[2], trial[1]]))+'\n')
            else:
                trainFile.write(','.join(map(str, [trial[0], trial[2], trial[1]]))+'\n')
            for frame in range(trial[0], trial[2]):
                img = clip.get_frame(frame*1./clip.fps)
                img = img/255.
                newImg = model.predict(normalize(img))*0.288+0.257
                newImg = np.concatenate([newImg, newImg, newImg], axis=3)
                plt.imsave(os.path.join(testDir if randTrial<args['valsplit'] else trainDir, str(frame).zfill(width)+'.png'), np.squeeze(newImg))

        clip.close()
        testFile.close()
        trainFile.close()
