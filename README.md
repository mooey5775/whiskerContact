Whisker Contact Detector
========================

Detects mouse whisker contacts with an obstacle for Sawtell Lab. Uses LEAP and a shallow CNN to detect time of first touch to perform reaction time analyses for mice.

Two example models are provided.

To generate training data, [whiskLabeler](https://github.com/mooey5775/whiskLabeler) is used.

#### To get started with training your whisker net, visit the [Quick Start](https://github.com/mooey5775/whiskerContact/quick_start.md)

Files
-----

`export_images.py`: Exports training data created by [whiskLabeler](https://github.com/mooey5775/whiskLabeler)  
`train_whisker_net.py`: Trains whisker network after exporting trained images  
`analyze_video.py`: Analyzes new whisker contact videos

Usage
-----

All files should be run with Python3. To get a list of arguments to pass in, run `python3 [FILE] -h`.

Most available arguments are optional and can (probably should) be left default. If they are changed, make sure the parameters are the same across all three training files

Improvements
------------

 - [x] Initial release
 - [x] Tested code
 - [ ] Better handling of argument preservation across scripts
 - [ ] More flexibility in model structure
 - [ ] Training performance visualization
 - [ ] Make it not necessary to have trackedfeaturesraw and runAnalyzed
