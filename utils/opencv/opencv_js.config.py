# Classes and methods whitelist
# from generator.embindgen import makeWhiteList

core = {
  '': ['bitwise_and', 'countNonZero'],
}

imgproc = {
  '': [
    'adaptiveThreshold',
    'boundingRect',
    'cvtColor',
    'dilate',
    'drawContours',
    'findContours',
    'getStructuringElement',
    'morphologyEx',
    'threshold'
  ]
}

white_list = makeWhiteList([core, imgproc])

# white_list = makeWhiteList([core, imgproc, objdetect, video, dnn, features2d, photo, calib3d])

# namespace_prefix_override['dnn'] = ''  # compatibility stuff (enabled by default)
# namespace_prefix_override['aruco'] = ''  # compatibility stuff (enabled by default)
