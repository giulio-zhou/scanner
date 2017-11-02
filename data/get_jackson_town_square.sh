#!/bin/bash

# mkdir -p data
# wget https://storage.googleapis.com/noscope-data/videos/jackson-town-square.mp4
# mv jackson-town-square.mp4 data

ffmpeg -i data/jackson-town-square.mp4 -ss 00:00:00 -t 00:30:00 data/jackson-town-square-30-minutes.mp4
