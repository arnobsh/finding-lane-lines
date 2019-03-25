#importing some useful packages
# Import everything needed to edit/save/watch video clips
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import math
import os

#reading in an image
from django.conf.locale import pl

image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')




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
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
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
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
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


def get_vertices_for_img(img):
    img_shape = img.shape
    height = img_shape[0]
    width = img_shape[1]

    vert = None

    if (width, height) == (960, 540):
        region_bottom_left = (130, img_shape[0] - 1)
        region_top_left = (410, 330)
        region_top_right = (650, 350)
        region_bottom_right = (img_shape[1] - 30, img_shape[0] - 1)
        vert = np.array([[region_bottom_left, region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    else:
        region_bottom_left = (200, 680)
        region_top_left = (600, 450)
        region_top_right = (750, 450)
        region_bottom_right = (1100, 650)
        vert = np.array([[region_bottom_left, region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

    return vert


os.listdir("test_images/")


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
def connecting_lines(image):
    """
    An function to connect the different section of lane lines
    """
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = int(image.shape[0] * (3 / 5))
    max_y = int(image.shape[0])
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    img = np.copy(image)
    thickness = 5
    color = [255, 0, 0]
    lines = [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y],
    ]]
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)


    return line_img


# function to save the output images into directory
def processBulkImages(input_dir, output_dir):
    """ Perform the operations on every images and then will save into output directory
    """
    for file in os.listdir(input_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            # print(os.path.join(directory, filename))
            # temporarily save the image into output directory
            original_image = np.copy(mpimg.imread(os.path.join(input_dir, filename)))
            # convert image to grayscale.

            gray_temp_image = grayscale(original_image)  # grayscale conversion
            mpimg.imsave(os.path.join("test_images_output/greyscale_images/", filename), gray_temp_image)
            # Define a kernel size for Gaussian smoothing / blurring
            # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
            kernel_size = 7
            blur_gray = gaussian_blur(gray_temp_image, kernel_size)
            mpimg.imsave(os.path.join("test_images_output/gaussian_blur/", filename), blur_gray)
            # Define parameters for Canny and run it
            low_threshold = 50
            high_threshold = 150
            output_image_edges = canny(blur_gray, low_threshold, high_threshold)
            mpimg.imsave(os.path.join("test_images_output/canny_edge_detection/", filename), output_image_edges)
            # This time we are defining a four sided polygon to mask
            imshape = original_image.shape
            vertices = np.array([[(0, imshape[0]), (480, 315), (490, 315), (imshape[1], imshape[0])]], dtype=np.int32)
            # Next we'll create a masked edges image using fillPoly()
            masked_tmp_image = region_of_interest(output_image_edges, vertices)
            mpimg.imsave(os.path.join("test_images_output/region_of_interest/", filename), masked_tmp_image)
            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            threshold = 1  # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 1  # minimum number of pixels making up a line
            max_line_gap = 2  # maximum gap in pixels between connectable line segments

            lines = hough_lines(masked_tmp_image, rho, theta, threshold,
                                min_line_length, max_line_gap)
            mpimg.imsave(os.path.join("test_images_output/hough_transform/", filename), lines)
            # extrapolate the lines from the image
            image_with_connected_lines = connecting_lines(original_image)
            # plt.imshow(image_with_connected_lines)
            mpimg.imsave(os.path.join("test_images_output/connecting_lines/", filename), image_with_connected_lines)
            # # Draw the lines on the edge image
            output_image = weighted_img(image_with_connected_lines, original_image)
            mpimg.imsave(os.path.join(output_dir+"final_images/", filename), output_image)
        else:
            continue


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # convert image to grayscale.

    gray_temp_image = grayscale(image)  # grayscale conversion

    # Define a kernel size for Gaussian smoothing / blurring
    # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
    kernel_size = 7
    blur_gray = gaussian_blur(gray_temp_image, kernel_size)

    # Define parameters for Canny and run it
    low_threshold = 50
    high_threshold = 150
    output_image_edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (480, 315), (490, 315), (imshape[1], imshape[0])]], dtype=np.int32)
    # Next we'll create a masked edges image using fillPoly()
    masked_tmp_image = region_of_interest(output_image_edges, vertices)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 1  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 1  # minimum number of pixels making up a line
    max_line_gap = 2  # maximum gap in pixels between connectable line segments

    lines = hough_lines(masked_tmp_image, rho, theta, threshold,
                        min_line_length, max_line_gap)
    # extrapolate the lines from the image
    image_with_connected_lines = connecting_lines(image)
    # plt.imshow(image_with_connected_lines)

    # # Draw the lines on the edge image
    result = weighted_img(image_with_connected_lines, image)
    return result

def process_videos(input_dir, output_dir):
    # processing all files of the input directory
    for file in os.listdir(input_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".mp4"):
            clip1 = VideoFileClip(os.path.join(input_dir, filename))
            white_clip = clip1.fl_image(process_image)
            white_clip.write_videofile(os.path.join(output_dir,filename), audio=False)
        else:
            continue

# iterate over files
input_directory = "test_images/"
output_directory = "test_images_output/"
processBulkImages(input_directory, output_directory)

def play_video(video):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video  file")

        # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

# for video files
# white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# %time white_clip.write_videofile(white_output, audio=False)
# white_clip.write_videofile(white_output, audio=False)

# process all videos, by single function
# video_input_dir = 'test_videos/';
# video_output_dir = 'test_videos_output/';
# play_video('test_videos/solidWhiteRight.mp4')
# process_videos(video_input_dir,video_output_dir)