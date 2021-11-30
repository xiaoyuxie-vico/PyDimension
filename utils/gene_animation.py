import cv2
import os

import glob


# video_name = '../results/keyhole_trajectory/video.avi'
# images = glob.glob('../results/keyhole_trajectory/*jpg')

# video_name = '../results/rb_trajectory/rb_video.avi'
# images = glob.glob('../results/rb_trajectory/*jpg')

video_name = '../results/ps_results/ps_keyhole.avi'
images = glob.glob('../results/ps_results/ps_keyhole_*.jpg')

# video_name = '../results/ps_results/ps_rb.avi'
# images = glob.glob('../results/ps_results/ps_rb_*.jpg')

# video_name = '../results/porosity_ps_vis/porosity.avi'
# images = glob.glob('../results/porosity_ps_vis/iter_*.jpg')


images.sort()

frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width, height))
# video = cv2.VideoWriter(video_name, 0, 5, (width, height))

for img_path in images:
    video.write(cv2.imread(img_path))

cv2.destroyAllWindows()
video.release()
