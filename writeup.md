# **Finding Lane Lines on the Road**

[image1]: ./pipeline.jpg "Pipeline"

## Reflection

### 1. Description of my pipeline and modified draw_lines() function.

My pipeline consisted of 5 steps:

1. Grayscale convertion
2. Image blurring
3. Edge detection using Canny algorithm
4. Mask our region of interest over the image. 
	- I define a trapezoid with the lower side extending all the lenght of the bottom of the image, and the upper side placed 80 pixels below the center and the corners 100 pixels to each side of the center.
5. Detect lines using the Hough transform

After this pipeline I get several lines segments placed over the edges within the original image. Mainly they are placed around the road lanes, but edges of other objects, like cars, may still remain. To group the lines segments of the road lanes into single left and right lanes, I modified the draw_lines() function implementing these steps:

- Distinguish between lines on the left and the right based on position and slope
- Calculate the fitting line for each left and right groups of line segments
- Exclude line segments that are not close to the fitted lines. 
	- As most of the edges belong to the road lanes, the fitted line will be close to the lane, while edges belonging to other objects, like cars, will be discarded on this step
- Fit again the lines over the remaining points. This second fitting will be more accurate, using more edges belonging to the road lanes

The full pipeline with the final highlighting of the road lanes is shown in the image below:
 ![alt text][image1]
### 2. Potential shortcomings in my current pipeline

There are many fixed parameters in my pipeline, for example the region of interest to mask, the estimated slopes of the lines, or the Canny algorithm and Hough transform parameters. That can be a shortcoming with different ilumination settings (like night or fog) where different Canny or Hough parameter would be best suited, or unusual camera angles.

Another shortcoming, well seen on the challenge video, appears with different ilumination settings on the same picture (shadows). The algorithm properly detects the lane before the tree shadow, but fails on it.

A further problem presents when other cars get into the region of interest, as some of its edges are not filtered through the pipeline.

Sharper curves can present a challenge too, as the fitting is linear in this pipeline.

### 3. Possible improvements to the current pipeline

A possible improvement could be to have different parameter sets adjusted for different ilumination settings. If the number of lines segments out of the Hough transform is too low, we could iterate through the different sets until we get a minimum of data to extrapolate the road lanes.

Also, if the data is too scattered and does not resemble a line we can repeat iterations narrowing down the region of interest or the slope limits.

Another possible improvement would be to fit our lines to a higher degree polynom and better fit road lanes inside a curve.
