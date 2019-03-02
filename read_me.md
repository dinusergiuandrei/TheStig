
Videos from training:

https://www.youtube.com/watch?v=T3svN3MMMCc
https://www.youtube.com/watch?v=zECiPLo45b4

If the input is the entire image it is too large for a ANN to learn from it. (w * h * 3) <br>
If we scale down the image, the input is still to large. (w/r * h/r * 3) <br>
If we approximate the pixel to closest from green/red/grey, the input size is smaller, but still too large. (w/r * h/r) <br>
Considering what is required in making a decision in the game, we can trim the size of the viewport to a smaller  <br>
rectangle. Now the ANN starts to work, but we can make the input even smaller. <br>
Consider the width of the track as t pixels. <br>
We observed that the car occupies roughly the same pixels in a race. <br>
Consider a line that goes over the front of the car. (on a straight road, it should be perpendicular to the road) <br>
On that line, we are interested only in the t pixels to the right of the car and the t to the left. <br>
From those pixels, we go up and count how many pixels there are until we hit green (off-track). <br>
For example, if a left turn follows, the array will be similar to this: <br>
0 0 250 ... 200 ... 150 ... 100 ... 0 0 0 <br>
These numbers are divided by the maximum numbers of pixels from the car to the top of the image (to normalize). <br>
The resulting array is the input for the ANN. <br>
The reward is larger when the car is at the center of the road. (feature_extractor.get_reward()) <br>
The rewards from the game environment is not as responsive.