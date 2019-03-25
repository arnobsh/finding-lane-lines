# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
--------

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).The code file is called P1.ipynb and the writeup template is writeup_template.md 

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)


# Ideas and Implementation

Self-driving car does not have any visions like we do. So it needs to depend on the sensors, algorithms. It’s more obvious to identify a road only with sensor and algorithm will complex. Also, the algorithm needs constant adjusting while it processes. Because, we are driving car and process the data from the camera at the same time.

Using computer vision technique (OpenCV library) and some famous algorithm we can overcome that challenge. I will discuss those algorithm in a fewer lines later.

 
# Input and Output

From the Udacity provided the test input and test output (both images and videos). The final output will be a video file where we can identify lane on real time

Input:

![Input][image1]

Output:

![Output][image2]

# Setup

I downloaded all files locally. I used pyCharm for my development, developed my project using IDE and then re-run the pipelines through the 
workspace.

### Reflection

# Pipelines:

I used several algorism for my project. Here are the list of the algorithm those I used

* Convert original image to Greyscale image
* Apply Gaussian Blur to remove the extra noises
* Apply Canny Edge Detection on blur image
* Trace Region Of Interest and discard all other lines identified by our previous step that are outside this region
* Hough Transform to find lanes within our region of interest
* Extrapolate and intraplate the lines segment to be continued line


### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I use Gaussian Blur algorithm with kernel size 7 to remove the extra noises from the image.

After that I apply Canny Edge Detection on blur image to find out the edges of the lines. For doing so I applied low threshold 50 and high threshold 150.

After identifying the canny edges, I try to find the region of interest which I need to draw the lines. To do so I use polyfit algorithm with the vertices (X1, Y1) = (480, 315) and (X2, Y2) = (490, 315) respectively.

Then I run Hough on edge detected image. Output "lines" is an array containing endpoints of detected line segments. Here as parameter I use
threshold = 1, min_line_length = 1 and max_line_gap = 2 to identify the image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by finding out the slopes of each line segment and connection those segments. Here I use different technique to identify the slope. I used original image to calibrate the lines with generated Hough space masked image. I use canny edge and Hough space algorithm and also cropped the image for smooth processing. I used color mask [255, 0, 0]

After connected lines of both left lane line and right lane line, I generate the output image by masking over the original image

# Steps and Algorithms:

## Convert image to greyscale

For smooth finding edges at the very beginning we transform the original colored image to greyscale image. 

The conversion from RGB to a different space helps in reducing noise from the original three color channels. This is a necessary pre-processing steps before we can run more powerful algorithms to isolate lines.

![Greyscale Image][image4]

## Gaussian Blur

[Gaussian blur](https://en.wikipedia.org/wiki/Gaussian_blur) (also referred to as Gaussian smoothing) is a pre-processing technique used to smoothen the edges of an image to reduce noise. We counter-intuitively take this step to reduce the number of lines we detect, as we only want to focus on the most significant lines (the lane ones), not those on every object. We must be careful as to not blur the images too much otherwise it will become hard to make up a line.

The OpenCV implementation of Gaussian Blur takes a integer kernel parameter which indicates the intensity of the smoothing. For our task we choose a value of _7_.

![Gaussian blur Image][image3]

## Region Of Interest

 We need to determine a region of interest and discard any lines outside of this polygon. One crucial assumption in this task is that the camera remains in the sample place across all these image, and lanes are flat, therefore we can identify the critical region we are interested in.
 
 For finding the region of interest we apply polyfit algorithm . We use fillPoly() to determine the region of interest. We assume (X1,Y1) and (X2,Y2) (480, 315), (490, 315) respectively.
 
 ![Region Of Interest Image][image8]
 
## Canny Edge Detection

we can apply a [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector), whose role it is to identify lines in an image and discard all other data.

The OpenCV implementation requires passing in two parameters in addition to our blurred image, a low and high threshold which determine whether to include a given edge or not. A threshold captures the intensity of change of a given point (you can think of it as a gradient). Any point beyond the high threshold will be included in our resulting image, while points between the threshold values will only be included if they are next to edges beyond our high threshold. Edges that are below our low threshold are discarded. Recommended low:high threshold ratios are 1:3 or 1:2. We use values _50_ and _150_ respectively for low and high thresholds.

![Canny Edge Detection Image][image5]

## Hough Transform

The next step is to apply the [Hough Transform](https://en.wikipedia.org/wiki/Hough_transform) technique to extract lines and color them. The goal of Hough Transform is to find lines by identifying all points that lie on them. This is done by converting our current system denoted by axis (x,y) to a _parametric_ one where axes are (m, b). In this plane:

 * lines are represented as points
 * points are presented as lines (since they can be on many lines in traditional coordinate system)
 * intersecting lines means the same point is on multiple lines

Therefore, in such plane, we can more easily identify lines that go via the same point. We however need to move from the current system to a _Hough Space_ which uses _polar coordinates_ one as our original expression is not differentiable when m=0 (i.e. vertical lines). In polar coordinates, a given line will now be expressed as (ρ, θ), where line L is reachable by going a distance ρ at angle θ from the origin, thus meeting the perpendicular L; that is ρ = x cos θ + y sin θ.

All straight lines going through a given point will correspond to a sinusoidal curve in the (ρ, θ) plane. Therefore, a set of points on the same straight line in Cartesian space will yield sinusoids that cross at the point (ρ, θ). This naturally means that the problem of detecting points on a line in Cartesian space is reduced to finding intersecting sinusoids in Hough space.

![Hough Spaces][image6]

More information about the implementation of Hough Transform in OpenCV can be found [here](http://docs.opencv.org/trunk/d6/d10/tutorial_py_houghlines.html)

# Line Extrapolation and Connection lines

To trace a full line from the bottom of the screen to the highest point of our region of interest, we must be able to interpolate the different points returned by our Hough transform function, and find a line that minimizes the distance across those points. Basically this is a [linear regression](https://en.wikipedia.org/wiki/Regression_analysis) problem. We will attempt to find the line on a given lane by minimizing the [least squares](https://en.wikipedia.org/wiki/Least_squares) error. 

![Line Extrapolation][image7]

To extrapolate the line from the image , we first identify the slope of the left and right lanes . Then we determine the line intersection points. Then for each of the line vertices those are not connected we connect the points an draw lines.


[//]: # (Image References)
[image1]: ./input.jpg "Input Image"
[image2]: ./final_output.jpg "Final Output Image"
[image3]: ./gaussian_blur.jpg "Gaussian Blur"
[image4]: ./greyscale_images.jpg "Grey Scale"
[image5]: ./canny_edge_detection.jpg "Canny Edge Detection"
[image6]: ./hough_transform.jpg "Hough Transform"
[image7]: ./connecting_lines.jpg "Connecting Lines"
[image8]: ./region_of_interest.jpg "Region Of Interest"


---

### 2. Potential Shortcomings

One potential shortcoming would be what would happen when there is a 360 degree curve my lane goes out of the white or yellow box.

Again, for sometimes when I was solving the challenge video, I found for the tree shadow images whose edge point cannot be removed fully from canny edge.

The adjusting Hough space by implementing different threshold and line gap always gave me different result which is interesting. Not sure my settings is the best in that case. 

Another shortcoming was when I am extrapolating lines and connecting them, using the existing masked image from Hough does not give me the good result. I have to modify the parameters and as well as need more cropping before sending image to Hough space. I used different technique to identify the connection slope at there

### 3. Suggest possible improvements to your pipeline

I need to solve the challenge video. I think the video resolution is high which I did not adjust when I am generating image from the canny edge. I need to adjust that

For 360 degree curve I need to remove extra noises from the image and also always make sure the lane lines always in the section of the road.

All code is available on [Github](https://github.com/arnobsh/finding-lane-lines.git).
