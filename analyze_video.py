from moviepy.editor import VideoFileClip
import keras
import numpy as np
import operator
import keras.backend as K
import scipy.io
import bisect
from keras.models import Model, load_model
from keras.layers import Input, MaxPooling2D
import time
import csv
from tqdm import tqdm
import os
import sys
from collections import deque
import argparse

from slowai.layers import *
from slowai.metrics import *

# Functions
def meanError(y_true, y_pred):
    ground_truth = K.argmax(y_true, axis=-1)
    preds = K.argmax(y_pred, axis=-1)
    nines = K.ones_like(preds) * 9
    masked = K.equal(preds - ground_truth, nines)
    return K.mean(preds - ground_truth, axis=-1)

def meanSquared(y_true, y_pred):
    ground_truth = K.argmax(y_true, axis=-1)
    preds = K.argmax(y_pred, axis=-1)
    return K.mean(K.square(preds - ground_truth), axis=-1)

def meanAbsError(y_true, y_pred):
    ground_truth = K.argmax(y_true, axis=-1)
    preds = K.argmax(y_pred, axis=-1)
    return K.mean(K.abs(preds - ground_truth), axis=-1)

def convertWiskStamps(wiskframe, mat):
    time = mat['frameTimeStampsWisk'][wiskframe][0]
    return bisect.bisect_left(np.squeeze(mat['frameTimeStamps']), time)

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument('session', help="session ID")
ap.add_argument('-s', '--session-dir', required=True,
                help="path to directory of sessions")
ap.add_argument('-m', '--whisker-model', required=True,
               help="path to whisker model file")
ap.add_argument('-cm', '--crop-model', required=True,
               help="path to crop model file")
ap.add_argument('-t', '--timesteps', type=int, default=10,
               help="number of timesteps (OPTIONAL) (default 10)")
ap.add_argument('-bs', '--batch-size', type=int, default=64,
               help="batch size when analyzing (OPTIONAL) (default 64)")
ap.add_argument('--mean', type=float, default=0.257,
                   help="OPTIONAL: adjust image normalization mean (default 0.257)")
ap.add_argument('--std', type=float, default=0.288,
                   help="OPTIONAL: adjust image normalization std (default 0.288)")
ap.add_argument('--crop-size', type=int, default=200,
                help="size of cropped image (optional, default 200)")
ap.add_argument('--width-offset', type=int, default=-50,
                help="width offset for cropping (optional, default -50)")
ap.add_argument('--height-offset', type=int, default=0,
                help="height offset for cropping (optional, default 0)")
args = vars(ap.parse_args())
INF = sys.maxsize
# TODO: make this flexible for multiple sizes of sliding window
probDistribution = [0.5, 0.707, 0.867, 0.966, 1.0, 1.0, 0.966, 0.867, 0.707, 0.5]

print("[INFO] loading models...")
crop_model = load_model(args['crop_model'],
                       custom_objects={'meanDistance': meanDistance})
model = load_model(args['whisker_model'],
                   custom_objects={'meanError': meanError,
                                   'meanSquared': meanSquared,
                                   'meanAbsError': meanAbsError})

print("[INFO] assembling end-to-end-model")
num_features = int(model.layers[2].input.shape[2])

conv_net = model.layers[1].layer
v = crop_model.output
v = Maxima2D()(v)
v = PointCrop2D(crop_size=args['crop_size'], mean=args['mean'],
                std=args['std'], wOffset=args['width_offset'],
                hOffset=args['height_offset'])([v, crop_model.input])
v = MaxPooling2D((2, 2), padding='same', name='downsampler')(v)
v = conv_net(v)

visual_model = Model(input=crop_model.input, outputs=v)

interm_input = Input((10, num_features,))
layers = [l for l in model.layers]
x = layers[2](interm_input)
for i in range(3,len(layers)):
    x = layers[i](x)

linear_model = Model(inputs=interm_input, outputs=x)

print("[INFO] loading session metadata...")
frames = {}
softmaxs = {}
clip = VideoFileClip(os.path.join(args['session_dir'], args['session'],
                     'runWisk.mp4'))
print("[INFO] Duration of video (s): {}, FPS: {}, Dimensions: {}".format(clip.duration, clip.fps, clip.size))

trackedFeatures = np.loadtxt(os.path.join(args['session_dir'], args['session'],
                             'trackedFeaturesRaw.csv'), delimiter=',',
                             skiprows=1)
mat = scipy.io.loadmat(os.path.join(args['session_dir'], args['session'],
                       'runAnalyzed.mat'))

print("[INFO] preprocessing session...")
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
            mat['frameTimeStamps'][i][0]=INF
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

size = [clip.size[1], clip.size[0]]

fieldnames = ["framenum", "confidence"]
answers = [{"framenum":-1, "confidence": 0}]*len(mat['obsOnTimes'])

print("[INFO] analyzing video...")
for idx, framenum, endframe in tqdm(trialFrames):
    try:
        if framenum == -1 or endframe == -1:
            continue
        findTime = 0
        initialStart = time.time()
        start = time.time()
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
        frameProbs = {}
        oldFrame = framenum
        if framenum >= endframe:
            print("Could not find obstacle within session. Skipping session...")
            continue
        findTime = time.time()-start
        loadTime = 0
        resizeTime = 0
        bottleneckTime = 0
        predictTime = 0
        cacheTime = 0
        incrementTime = 0
        needFrames = deque()
        while (nosePos[0]-obsPos[0]<50 or obsConf!=1) and framenum < endframe:
            needFrames.append(framenum)
            framenum+=1
            features = trackedFeatures[convertWiskStamps(framenum, mat)]
            obsPos = list(map(int, [features[22], features[23]]))
            obsConf = features[24]
            nosePos = list(map(int, [features[19], features[20]]))
        for i in range(args['timesteps']):
            needFrames.append(framenum)
            framenum+=1
        frames.clear()
        while needFrames:
            frameBatch = []
            for i in range(args['batch_size']):
                if not needFrames:
                    break
                frameBatch.append(needFrames.popleft())
            actualBatch = [((1.-np.reshape((clip.get_frame(i*1./clip.fps)/255.)[:,:,0], tuple(size)+(1,)))-0.257)/0.288 for i in frameBatch]
            actualBatch = np.asarray(actualBatch)
            predBatch = visual_model.predict(actualBatch)
            frames.update({i:j for (i, j) in zip(frameBatch, predBatch)})
        framenum = oldFrame
        features = trackedFeatures[convertWiskStamps(framenum, mat)]
        obsPos = list(map(int, [features[22], features[23]]))
        obsConf = features[24]
        nosePos = list(map(int, [features[19], features[20]]))
        needAnal = deque()
        while (nosePos[0]-obsPos[0]<50 or obsConf!=1) and framenum < endframe:
            session = []
            if framenum<args['timesteps']:
                framenum+=1
                features = trackedFeatures[convertWiskStamps(framenum, mat)]
                obsPos = list(map(int, [features[22], features[23]]))
                obsConf = features[24]
                nosePos = list(map(int, [features[19], features[20]]))
                continue
            if framenum>int(clip.duration*clip.fps-50):
                framenum+=1
                features = trackedFeatures[convertWiskStamps(framenum, mat)]
                obsPos = list(map(int, [features[22], features[23]]))
                obsConf = features[24]
                nosePos = list(map(int, [features[19], features[20]]))
                continue
            for i in range(args['timesteps']):
                frame = framenum+i
                session.append(frame)
            needAnal.append(session)
            framenum+=1
            features = trackedFeatures[convertWiskStamps(framenum, mat)]
            obsPos = list(map(int, [features[22], features[23]]))
            obsConf = features[24]
            nosePos = list(map(int, [features[19], features[20]]))
        softmaxs.clear()
        while needAnal:
            frameBatch = []
            for i in range(args['batch_size']):
                if not needAnal:
                    break
                frameBatch.append(needAnal.popleft())
            actualBatch = np.asarray([[frames[i] for i in j] for j in frameBatch])
            predBatch = linear_model.predict(actualBatch)
            softmaxs.update({i[0]:j for (i,j) in zip(frameBatch, predBatch)})
        for framenum in softmaxs:
            softmax = softmaxs[framenum]
            prediction = np.argmax(softmax)
            if prediction == args['timesteps']:
                continue
            if framenum+prediction in frameProbs:
                frameProbs[framenum+prediction]+=probDistribution[prediction]
            else:
                frameProbs[framenum+prediction]=probDistribution[prediction]
        try:
            predictedFrame = max(frameProbs.items(), key=operator.itemgetter(1))[0]
        except ValueError:
            predictedFrame=-1
        if predictedFrame != -1:
            answers[idx] = {"framenum": predictedFrame, "confidence": frameProbs[predictedFrame]}
    except:
        print("exception in trial, skipping")

print("[INFO] saving data...")
with open(os.path.join(args['session_dir'], args['session'],
          'whiskerAnalyaed.csv'), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for answer in answers:
      writer.writerow(answer)
