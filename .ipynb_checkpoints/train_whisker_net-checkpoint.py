from slowai.util import *
from slowai.layers import *
from slowai.metrics import *

import keras
import losswise
import os
import multiprocessing
import time
import argparse

from PIL import Image as pil_image

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed, Conv3D, MaxPooling3D
from keras.models import Model, Sequential
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from losswise.libs import LosswiseKerasCallback
from keras.callbacks import ModelCheckpoint

from moviepy.editor import VideoFileClip
import numpy as np
from skimage.transform import resize

import keras.backend as K
import random

# Constants
COLOR_MODE = 'grayscale'

# Function definitions
def getImg(image):
    img = pil_image.open(image)
    x = np.asarray(img, dtype=K.floatx())[:,:,0]/255.
    x = x.reshape((x.shape[0], x.shape[1], 1))
    return x

def meanError(y_true, y_pred):
    groundTruth = K.argmax(y_true, axis=-1)
    preds = K.argmax(y_pred, axis=-1)
    nines = K.ones_like(preds)*9
    masked = K.equal(preds-groundTruth, nines)
    return K.mean(preds-groundTruth, axis=-1)
    
def meanSquared(y_true, y_pred):
    groundTruth = K.argmax(y_true, axis=-1)
    preds = K.argmax(y_pred, axis=-1)
    return K.mean(K.square(preds-groundTruth), axis=-1)

def meanAbsError(y_true, y_pred):
    groundTruth = K.argmax(y_true, axis=-1)
    preds = K.argmax(y_pred, axis=-1)
    return K.mean(K.abs(preds-groundTruth), axis=-1)

class WhiskerGenerator(keras.utils.Sequence):
    
    def __init__(self, baseDir, timesteps, batch_size, rotation_range=0, width_shift_range=0, height_shift_range=0, zoom_range=0, mean=0.257, std=0.288):
        self.baseDir = baseDir
        if not os.path.exists(baseDir):
            raise ValueError("Invalid data path")
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.mean = np.reshape([mean], [1,1,1])
        self.std = np.reshape([std], [1,1,1])
        self.augment = rotation_range or width_shift_range or height_shift_range or zoom_range
        if self.augment:
            self.datagen = ImageDataGenerator(rotation_range=rotation_range, width_shift_range=width_shift_range, height_shift_range=height_shift_range, zoom_range=zoom_range)
        extensions = {'png', 'PNG'}
        self.sessions = []
        for subdir in sorted(os.listdir(self.baseDir)):
            if os.path.isdir(os.path.join(self.baseDir, subdir)):
                self.sessions.append(subdir)
        self.frameInfo = {}
        for session in self.sessions:
            if os.path.exists(os.path.join(self.baseDir, session, "frameInfo.csv")):
                self.frameInfo[session] = np.loadtxt(os.path.join(self.baseDir, session, "frameInfo.csv"), delimiter=",")
            else:
                print(session, "is not a valid session")
                continue
        self.training_examples = []
        self.width = {}
        for session in self.frameInfo:
            counter = 0
            for trial in self.frameInfo[session]:
                if int(trial[1])-int(trial[0])<self.timesteps:
                    print("Trial not long enough, skipping...")
                    print("Frame:", trial[0])
                    continue
                for frame in range(int(trial[0]), int(trial[1])-self.timesteps+1):
                    self.onehot = np.zeros(self.timesteps+1)
                    if int(trial[2])-frame<self.timesteps and int(trial[2])-frame>=0:
                        self.onehot[int(trial[2])-frame] = 1
                    else:
                        self.onehot[self.timesteps] = 1
                    self.training_examples.append((session, frame, self.onehot))
                    counter+=1
            self.width[session] = len(os.listdir(os.path.join(self.baseDir, session))[0].split(".")[0])
            print("Found", counter, "images in", session)
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.training_examples)/self.batch_size))
    
    def __getitem__(self, idx):
        if idx>=int(np.ceil(len(self.training_examples)/self.batch_size)):
            print("index too large for number of training examples")
            return -1
                                            
        self.x = [list(map(getImg, [os.path.join(self.baseDir, image[0], str(frame).zfill(self.width[image[0]])+".png") for frame in range(image[1], image[1]+self.timesteps)])) for image in self.training_examples[idx*self.batch_size:min((idx+1)*self.batch_size, len(self.training_examples))]]
        if self.augment:
            xfrms = [self.datagen.get_random_transform(self.x[0][0].shape[:2]) for i in self.x]
            self.x = [[self.datagen.apply_transform(n, xfrms[i]) for n in x] for i, x in enumerate(self.x)]
        self.x = np.asarray(self.x)
        self.x-=self.mean
        self.x/=self.std
        self.y = np.asarray([ex[2] for ex in self.training_examples[idx*self.batch_size:min((idx+1)*self.batch_size, len(self.training_examples))]])
        
        return self.x, self.y
    
    def getClassWeights(self):
        self.counts = np.zeros(self.timesteps+1)
        for example in self.training_examples:
            self.counts[np.argmax(example[2])]+=1
        self.counts = 1/self.counts
        self.scaling = (self.timesteps+1)/np.sum(self.counts)
        self.counts*=self.scaling
        self.toReturn = {}
        for i, weight in enumerate(self.counts):
            self.toReturn[i]=weight
        return self.toReturn
    
    def on_epoch_end(self):
        random.shuffle(self.training_examples)

if __name__ == '__main__':
    
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('modelname', help="name of trained model")
    ap.add_argument('-d', '--data-dir', default="data",
                    help="desired path to input data (optional)")
    ap.add_argument('-m', '--model-dir', default="models",
                   help="desired path to model directory (optional)")
    ap.add_argument('-t', '--timesteps', type=int, default=10,
                    help="number of timesteps per window (optional, default 10)")
    ap.add_argument('-w', '--workers', type=int, default=8,
                   help="number of workers to load images during training (default 8)")
    ap.add_argument('-ws', '--width-shift', type=int, default=0,
                    help="number of width pixels to shift each training example by (data aug)")
    ap.add_argument('-hs', '--height-shift', type=int, default=0,
                   help="number of height pixels to shift each training example by (data aug)")
    ap.add_argument('-z', '--zoom-range', type=float, default=0,
                    help="allowable zoom range (data aug)")
    ap.add_argument('-r', '--rotation-range', type=int, default=0,
                   help="allowable rotation range (data aug)")
    ap.add_argument('-b', '--batch-size', type=int, default=32,
                    help="batch size (default 32)")
    ap.add_argument('-e', '--epochs', type=int, default=20,
                   help="number of training epochs (default 20)")
    ap.add_argument('-l', '--losswise-api',
                    help="OPTIONAL: Losswise API key to track training progress")
    ap.add_argument('--continue-training', help="OPTIONAL: model path to continue training")
    ap.add_argument('--mean', type=float, default=0.257,
                   help="OPTIONAL: adjust image normalization mean (default 0.257)")
    ap.add_argument('--std', type=float, default=0.288,
                   help="OPTIONAL: adjust image normalization std (default 0.288)")
    ap.add_argument('--disable-class-weighting', action='store_true',
                   help="OPTIONAL: disable class weighting in loss function")
    args = vars(ap.parse_args())
    
    # Argument validation
    if args['timesteps'] == 0:
        raise ValueError("Timesteps cannot be 0")
    if args['batch_size'] == 0:
        raise ValueError("Batch size cannot be 0")
    if args['epochs'] == 0:
        raise ValueError("Epoch count cannot be 0")
    if args['std'] == 0:
        raise ValueError("STD cannot be 0")
    if args['workers'] == 0:
        raise ValueError("Cannot have 0 workers")
    
    # Losswise compatibility
    if args['losswise_api']:
        losswise.set_api_key(args['losswise_api'])
        
    # Load whisker images
    print('[INFO] Loading whisker images...')
    test_gen = WhiskerGenerator(os.path.join(args['data_dir'], 'test'), args['timesteps'], args['batch_size'], mean=args['mean'], std=args['std'])
    train_gen = WhiskerGenerator(os.path.join(args['data_dir'], 'train'), args['timesteps'], args['batch_size'], width_shift_range=args['width_shift'], height_shift_range=args['height_shift'], zoom_range=args['zoom_range'], rotation_range=args['rotation_range'], mean=args['mean'], std=args['std'])
    
    if len(test_gen) == 0 or len(train_gen) == 0:
        raise ValueError("No training or test images")
        
    # Test whisker image load
    print('[INFO] Testing whisker image load...')
    x, y = train_gen[0]
    input_size = x[0][0].shape[:2]
    
    # Model definition and loading
    if not os.path.exists(args['model_dir']):
        os.makedirs(args['model_dir'])
    
    if args['continue_training']:
        print('[INFO] Loading existing model...')
        model = keras.models.load_model(args['continue_training'], custom_objects={"meanError": meanError, "meanSquared": meanSquared, "meanAbsError": meanAbsError})
    else:
        print('[INFO] Creating model...')
        x_in1 = Input(input_size+(1,))
        x = Conv2D(32, (7, 7), strides=2, activation='relu', padding='same', name='conv2d_1_1')(x_in1)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_2_1')(x)
        x = MaxPooling2D((2,2), padding='same', name='maxpooling2d_1_1')(x)
        x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same', name='conv2d_1_2')(x)
        x = MaxPooling2D((2,2), padding='same', name='maxpooling2d_1_2')(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        vision_model = Model(inputs=x_in1, outputs=x)
        
        x_in = Input((args['timesteps'],)+input_size+(1,))
        encoded_frames = TimeDistributed(vision_model)(x_in)
        encoded_frames = Flatten()(encoded_frames)
        
        x4 = Dense(64, activation='relu')(encoded_frames)
        x4 = Dropout(0.5)(x4)
        predictions = Dense(args['timesteps']+1, activation='softmax')(x4)
        
        model = Model(inputs=x_in, outputs=predictions)
        model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy', meanError, meanSquared, meanAbsError])
    
    # Training
    print('[INFO] Training...')
    callbacks = [ModelCheckpoint(os.path.join(args['model_dir'], args['modelname']+".{epoch:02d}-{val_loss:.2f}.h5"), period=5, verbose=1)]
    if args['losswise_api']:
        callbacks.append(LosswiseKerasCallback(tag=args['modelname'], display_interval=1))
    
    kwargs = {
        'verbose': 1,
        'epochs': args['epochs'],
        'validation_data': test_gen,
        'use_multiprocessing': args['workers'] > 1,
        'callbacks': callbacks
    }
    
    if args['workers'] > 1:
        kwargs['workers'] = args['workers']
    
    if not args['disable_class_weighting']:
        kwargs['class_weight'] = train_gen.getClassWeights()
        
    model.fit_generator(train_gen, **kwargs)
    
    # Final model save
    print('[INFO] Saving final model...')
    model.save(os.path.join(args['model_dir'], args['modelname']+'_final.h5'))
    
    print('[INFO] Done!')