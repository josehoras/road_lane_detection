#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    def keep_slopes_around_mean(lines):
        slopes = []
        # calculate all slopes
        for line in lines:
            for x1,y1,x2,y2 in line:
                slopes.append((y2-y1)/(x2-x1+1e-8))
        # Calculate the mean and standard deviation of all slopes
        mean = np.mean(slopes)
        std = np.std(slopes)
        # keep only lines with slope within mean +/- one sigma
        # make a list of indices that will be deleted
        new_lines = []
        for i in range(len(slopes)):
            if  mean - std < slopes[i] < mean + std:
                new_lines.append(lines[i])
        return new_lines
    
    def fit_lines(lines):
        # Group together Xs and Ys to make the lineal fit
        all_x = []
        all_y = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                all_x.extend((x1, x2))
                all_y.extend((y1, y2))
        # Calculate lineal fit
        m, b = np.polyfit(all_x, all_y, 1)  # y = m*x + b
        return m, b
           
    def keep_points_over_line(lines, m, b):

        distance = 20
        in_lines = []
        for line in lines:
            for x1,y1,x2,y2 in line:
#                 d = abs(m*x1 - y1 + b) / np.sqrt(m**2+1)
#                 if d < distance:
                if abs(m*x1 + b - y1) < distance and abs(m*x2 + b - y2) < distance:
                    in_lines.append(line)
        return in_lines

    def exclude_lines_on_horizon(lines, y_max):
        in_lines = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if y1 > y_max and y2 > y_max:
                    in_lines.append(line)
        return in_lines

    def draw_lanes(img, left_m, left_b, right_m, right_b, color=[255, 0, 0], thickness=10):
        y_left_1 = img.shape[0]
        x_left_1 = int((img.shape[0]-left_b)/left_m)
        y_right_1 = img.shape[0]
        x_right_1 = int((img.shape[0]-right_b)/right_m)
        y_horizon = int( (left_m * right_b - right_m * left_b) / (left_m - right_m))
        y_horizon = int(y_horizon + 20)
        x_left_2 = int((y_horizon - left_b) / left_m)
        x_right_2 = int((y_horizon - right_b) / right_m)
        cv2.line(img, (x_left_1, y_left_1), (x_left_2, y_horizon), color, thickness)
        cv2.line(img, (x_right_1, y_right_1), (x_right_2, y_horizon), color, thickness)
    
    def group_left_right(lines, center):
        lines_on_left = []
        lines_on_right = []
        # discriminate between left and right lanes only by position
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1+1e-8)
                if x1 < center - 40 and -1 < slope < -0.3:
                    lines_on_left.append(line)
                elif x1 > center + 40 and 1 > slope > 0.3:
                    lines_on_right.append(line)
        return lines_on_left, lines_on_right
            
    
    img_y= img.shape[0]
    img_x= img.shape[1]
    lines_on_left, lines_on_right = group_left_right(lines, img_x/2)
    # Calculate lines equation fitting points with slopes within mean +/- sigma
    left_m, left_b = fit_lines(lines_on_left)
    right_m, right_b = fit_lines(lines_on_right)
    draw_lanes(img, left_m, left_b, right_m, right_b,  color=[0, 0, 255], thickness=10)      
    # Exclude points that are not close to the fitted lines
    lines_on_left = keep_points_over_line(lines_on_left, left_m, left_b)
    lines_on_right = keep_points_over_line(lines_on_right, right_m, right_b)
#     # calculate horizon on the intersection of the two lines and substract some
#     y_horizon = int( (left_m * right_b - right_m * left_b) / (left_m - right_m))
#     y_horizon = int(y_horizon + 20)
#     # Exclude points above our horizon
#     lines_on_left = exclude_lines_on_horizon(lines_on_left, y_horizon)
#     lines_on_right = exclude_lines_on_horizon(lines_on_right, y_horizon)
    
    
    # calculate the mean slope and exclude lines with slope outside +/- sigma
    lines_on_left = keep_slopes_around_mean(lines_on_left)
    lines_on_right = keep_slopes_around_mean(lines_on_right)  
    
    if len(lines_on_left) > 2 and len(lines_on_right) > 2:
        # Recalculate with the new data set
        left_m, left_b = fit_lines(lines_on_left)
        right_m, right_b = fit_lines(lines_on_right)
        draw_lanes(img, left_m, left_b, right_m, right_b,  color=[0, 255, 0], thickness=10)

    
    # Draw small red lines
    for line in lines_on_left:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    for line in lines_on_right:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#     print(len(lines))
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

import os
os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

#printing out some stats and plotting
# print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)

f, ([[ax1, ax2, ax3], [ax4, ax5, ax6]]) = plt.subplots(nrows=2, ncols=3, figsize=(16,10))
plt.subplots_adjust(top=1, bottom=0, hspace=-0.4)
#reading in an image
image = mpimg.imread('test_images/solidWhiteCurve.jpg')

# 1. Convert image to grayscale
img_gray = grayscale(image)
ax1.set_title('Grayscale', {'fontsize': 16})
ax1.imshow(img_gray)

# 2. Apply Gaussian smoothing
kernel_size = 7
img_blur = gaussian_blur(img_gray, kernel_size)
# ax2.set_title('Blurred')
# ax2.imshow(img_blur)

# 3. Apply Canny edge detection
low_threshold = 50
high_threshold = 150
img_edges = cv2.Canny(img_blur, low_threshold, high_threshold)
ax2.set_title('Edges', {'fontsize': 16})
ax2.imshow(img_edges)

# 4. Mask region of interest
im_X = img_edges.shape[1]
im_Y = img_edges.shape[0]

vertices = np.array([[(0, im_Y), (im_X/2, im_Y/2), (im_X, im_Y)]], dtype=np.int32)

vertices = np.array([[(0, im_Y), 
                      (400, 320), (600, 320), 
                      (im_X, im_Y)]], dtype=np.int32)

img_masked = region_of_interest(img_edges, vertices)
ax3.set_title('Masked', {'fontsize': 16})
ax3.imshow(img_masked)

# 5. Apply Hough transform to extract the edges forming lines
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 1     # minimum number of votes (intersections in Hough grid cell)
min_line_len = 4  #minimum number of pixels making up a line
max_line_gap = 10    # maximum gap in pixels between connectable line segments

# Show image with lines from Hough transform
hough = cv2.HoughLinesP(img_masked, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
img_hough = np.zeros_like(image)
for line in hough:
    for x1,y1,x2,y2 in line:
        cv2.line(img_hough, (x1, y1), (x2, y2), [255,0,0], 2)
ax4.set_title('Hough', {'fontsize': 16})
ax4.imshow(img_hough)
# Filter Hough lines and draw road lines
img_lines = hough_lines(img_masked, rho, theta, threshold, min_line_len, max_line_gap)
ax5.set_title('Drawn', {'fontsize': 16})
ax5.imshow(img_lines)


img_weight = weighted_img(img_lines, image, α=0.8, β=1., γ=0.)
ax6.set_title('Result', {'fontsize': 16})
ax6.imshow(img_weight)
f.savefig("pipeline.jpg")


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    # 1. Convert image to grayscale
    img_gray = grayscale(image)

    # 2. Apply Gaussian smoothing
    kernel_size = 7
    img_blur = gaussian_blur(img_gray, kernel_size)

    # 3. Apply Canny edge detection
    low_threshold = 50
    high_threshold = 150
    img_edges = cv2.Canny(img_blur, low_threshold, high_threshold)

    # 4. Mask region of interest
    im_X = img_edges.shape[1]
    im_Y = img_edges.shape[0]

    vertices = np.array([[(0, im_Y), (im_X/2, im_Y/2), (im_X, im_Y)]], dtype=np.int32)

    vertices = np.array([[(0, im_Y), 
                          (400, 300), (600, 300), 
                          (im_X, im_Y)]], dtype=np.int32)
    img_masked = region_of_interest(img_edges, vertices)

    # 5. Apply Hough transform to extract the edges forming lines
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 1  #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    img_lines = hough_lines(img_masked, rho, theta, threshold, min_line_len, max_line_gap)

    result = weighted_img(img_lines, image, α=0.8, β=1., γ=0.)
    return result

#white_output = 'test_videos_output/solidWhiteRight.mp4'
### To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
### To do so add .subclip(start_second,end_second) to the end of the line below
### Where start_second and end_second are integer values representing the start and end of the subclip
### You may also uncomment the following line for a subclip of the first 5 seconds
###clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)
