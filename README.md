# Ultrapong

ping pong + AI.

## Setup Instructions

We recommend using a virtual environment. To set up a virtual environment, run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Inspiration

We were playing ping pong and trying to think of hackathon ideas. The path forward suddenly became clear to us.

## What it does

It tracks people playing ping pong using a simple video camera setup and commentates on it. It requires a green ping pong ball. Orange and white ping pong balls may also work, but after 10 hours of trying to get it to work, we decided to use green ping pong balls instead.

## How we built it

### Detecting the balls
We started by creating a method for detecting ping pong balls. After several iterations, we arrived at a solution involving two stages of masking:
1. Masking for motion: Static pixels are excluded as candidates for being a ping pong ball.
2. Masking for color: Pixels which do not meet a certain threshold of "greenness" (measured as a range of hues in HSV color space) are also filtered out.
This gives us a general mask of which pixels may belong to ping pong balls. From here, we apply a Gaussian blur and then contour detection to separate the pixels into distinguishable masses of points. These points are further filtered through two methods:
1. Area: Assuming the camera is a certain distance from the table, any masses of points that fall outside an acceptable range for a number of pixels are removed.
2. Convexity: We divide the area of the contour by the area of its convex hull. This allows us to filter out contours that are sparse and/or noisy.

### Detecting the table
We detect our table via another round of thresholding: We check to see if there is less than a certain level of luminance, and the proportions of red light to green and blue light. This gives us a mask containing a set of candidate pixels. From here, we apply contour detection to identify likely contours. We filter out contours that are too small, as well as contours that cannot be represented as quadrilaterals within 2% error.

### Detecting ball bounces
We used a simple technique to detect ball bounces: We check whether the coordinate (x, or y) has changed from increasing to decreasing. In practice, this simply means checking for `x[0] < x[1] > x[2]`, or similar. We debounce these measurements by introducing a minimum delay period between bounces of 0.7 seconds.

Using our table detection, we are able to create an axis resembling a net. We can then compare whether the ball was on the left side or right side of the table upon bouncing.

### State machine
We create a finite state machine to keep track of game state.

## Challenges we ran into

Locating the ball consistently was the most difficult portion, as it had to be differentiated from other moving objects including our bodies and paddles. It required a high degree of precision because all of our detections, e.g. tracking when the ball bounces, when the player hits the ball, and when the ball is lost, all depend on the location of the ball.

## Accomplishments that we're proud of

We’re proud that we were able to implement this project so it detects ping-pong gameplay accurately and robustly. It took over 30 iterations to correct the game logic and fine-tune the detection thresholds.

## What we learned

We learned about methods for filtering object detections, like Kalman filters, as well as several computer vision models, like Gaussian blurring, contour detection, eroding/dilation, HSV color spaces, and motion detection. Additionally, we learned that DFAs can actually have real-world applications!

On a more general note, we learned that simple tasks for humans, like locating a ping pong ball, can be surprisingly difficult to write as computer programs. While we did not train our own deep learning models for this project, we are optimistic about deep learning-based methods to be able to encode behaviors that are difficult to model explicitly.

## What's next for ultrapong

We hope to be able to generalize the functionality of ultrapong to balls of different colors and improve the consistency of results despite poor lighting, faster ball speeds, and noisier backgrounds. We would also hope to bring ultrapong to mobile devices and create a user-friendly interface to create seamless setup and gameplay.   

Moreover, we are interested in using deep learning approaches to improve the robustness of our detection algorithms.


