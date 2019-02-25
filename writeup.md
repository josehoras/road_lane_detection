# **Finding Lane Lines on the Road**

[//]: # (Image References)

[image1]: ./pipeline.jpg "Pipeline"

## Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps:

- Grayscale convertion
- Image blurring  
- Edge detection using Canny algorithm
- Mask our region of interest over the image
- Detect lines with Hough transform

After this pipeline I get many lines segments (around 400 to 500) placed on the edges within the original image. Mainly they are placed around the road lines, but other edges of other objects, like cars, may still remain. To group the lines segments of the road lanes into single left and right lanes, I modified the draw_lines() function implementing these steps:

- Distinguish between lines on the left and the right based first on position only
- Calculate the mean value of the slope for both groups of lines and keep only lines around of sigma of this value. This assumes that the mayority of lines belong to the road lane and the mean slope value is very close to the slope of the lane
- Fit a line to these points and exclude all points that are not over these line. At this point we have excluded lines whose slope is not close to the mean and that are not close to the fitted line
- Calculate the horizon by the point where our left and right lines cross, and set a buffer below this horizon to excluse the edges present on the horizon
- Fit again the lines over the remaining points

An example on the steps is shown in the image below:
 ![alt text][image1]
### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
