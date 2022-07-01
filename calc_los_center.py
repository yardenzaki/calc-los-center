## README:
"""
This Script takes a movie of 2 vibrating object and outputs each object center X,Y displacements to "centers_displ.csv"
steps:
1. Pick ROIs in the following order:
    a. Crop the frame to smaller ROI of the frame that includes the 2 objects -> Press enter/space to verify the ROI -> Press ESC
    b. The cropped region is opened now, Pick ROI for the first object center moving region (i.e. , the region which in every frame the center would be found inside) -> verify by pressing Enter/Space
    c. do the same as b. for the second object ROI
    d. press ESC

2. The script now is running and computing center displacements for objects 1 and 2
3. In the output folder images you will find:
    a. the output video
    b. "centers_displ.csv"

4. "centers_displ.csv" could be analyzed with "Analyze_centers_disp_DF.py" script


@Author: Yarden Zaki
@Date: 07/01/2022
@Version: 1.0
@Links: https://github.com/yardenzaki
@License: MIT
"""

import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
# import datetime
import imutils
from imutils import paths
import argparse
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog


def select_ROI(frame):
    """
    Select a ROI and then press SPACE or ENTER button!
    Cancel the selection process by pressing c button!
    Finish the selection process by pressing ESC button!

    """
    fromCenter = False
    ROIs = cv.selectROIs('Select ROIs', frame, fromCenter)

    return ROIs


def plot_contours_ff(firstFrame, min_max_countour, save_frames, min_max_thresh):
    # filter_2 = np.array([[3, -2, -3], [-4, 8, -6], [5, -1, -0]])
    # firstFrame=cv.filter2D(firstFrame,-1,filter_2)
    firstFrame_blur = cv.medianBlur(firstFrame, 3)
    first_frame_thresh = cv.threshold(firstFrame_blur, min_max_thresh[0], min_max_thresh[1], cv.THRESH_BINARY)[1]
    # first_frame_thresh = cv.dilate(first_frame_thresh, None, iterations=1)
    cv.imshow('first_frame_thresh', first_frame_thresh)
    text = 'Fisrt Frame'
    feed = firstFrame.copy()
    cnts = cv.findContours(first_frame_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centers = []
    t = 1

    range_x1 = range(ROIs[1][0], ROIs[1][0] + ROIs[1][2])
    range_y1 = range(ROIs[1][1], ROIs[1][1] + ROIs[1][3])

    range_x2 = range(ROIs[2][0], ROIs[2][0] + ROIs[2][2])
    range_y2 = range(ROIs[2][1], ROIs[2][1] + ROIs[2][3])

    for c in cnts:
        print("Found Contours:", "with Area:", cv.contourArea(c))
        if cv.contourArea(c) < min_max_countour[1] and cv.contourArea(c) > min_max_countour[0]:
            print("Contour Area", cv.contourArea(c))
            cv.drawContours(feed, [c], 0, (256, 256, 256), 1)
            # compute the center of the contour
            M = cv.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the contour and center of the shape on the image
            (x, y, w, h) = cv.boundingRect(c)
            # cX=int(x+w/2)
            # cY=int(y+h/2)
            cond_x_1 = cX in range_x1
            cond_x_2 = cX in range_x2
            cond_x = cond_x_1 or cond_x_2
            cond_y_1 = cY in range_y1
            cond_y_2 = cY in range_y2
            cond_y = cond_y_1 or cond_y_2

            if cond_x and cond_y:
                cv.rectangle(feed, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.circle(feed, (cX, cY), 2, (255, 255, 255), -1)
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                circle_text = "center" + str(t) + ":" + str(cX) + "," + str(cY)
                cv.putText(feed, circle_text, (x - 20, y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # centers.append((x+w/2,y+h/2))
                centers.append((cX, cY))
                t = t + 1
                cv.putText(feed, "Frame #" + text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if save_frames is True:
                cv.imwrite("first_frame_cnts.jpg", feed)
                cv.imwrite("first_frame_thresh.jpg", first_frame_thresh)
            else:
                cv.imshow('first_frame_cnts', feed)
                cv.imshow('first_frame_thresh', first_frame_thresh)
        else:
            continue

    return (centers)


def plot_blur_level(feed, count, min_max_countour):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    gray = cv.cvtColor(feed, cv.COLOR_BGR2GRAY)  # frame in Gray
    fm = variance_of_laplacian(gray)
    text = "Not Blurry"
    return


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv.Laplacian(image, cv.CV_64F).var()


def plot_contours(feed, count, min_max_countour, min_max_thresh):
    feed_blur = cv.medianBlur(feed, 3)
    feed_thresh = cv.threshold(feed_blur, min_max_thresh[0], min_max_thresh[1], cv.THRESH_BINARY)[1]
    # feed_thresh = cv.dilate(feed_thresh, None, iterations=1)
    text = str(count)
    cnts = cv.findContours(feed_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centers = []
    t = 1

    range_x1 = range(ROIs[1][0], ROIs[1][0] + ROIs[1][2])
    range_y1 = range(ROIs[1][1], ROIs[1][1] + ROIs[1][3])

    range_x2 = range(ROIs[2][0], ROIs[2][0] + ROIs[2][2])
    range_y2 = range(ROIs[2][1], ROIs[2][1] + ROIs[2][3])

    for c in cnts:
        if cv.contourArea(c) < min_max_countour[1] and cv.contourArea(c) > min_max_countour[0]:
            print("Contour Area", cv.contourArea(c))
            cv.drawContours(feed, [c], 0, (256, 256, 256), 1)
            # compute the center of the contour
            M = cv.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            (x, y, w, h) = cv.boundingRect(c)
            # print("x,y,w,h",x,y,w,h)
            # cX=int(x+w/2)
            # cY=int(y+h/2)
            print("cX", cX, "cY", cY)
            cond_x_1 = cX in range_x1
            cond_x_2 = cX in range_x2
            cond_x = cond_x_1 or cond_x_2
            cond_y_1 = cY in range_y1
            cond_y_2 = cY in range_y2
            cond_y = cond_y_1 or cond_y_2
            if cond_x and cond_y:
                cv.rectangle(feed, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.circle(feed, (cX, cY), 2, (255, 255, 255), -1)
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                circle_text = "center" + str(t) + ":" + str(cX) + "," + str(cY)
                cv.putText(feed, circle_text, (x - 20, y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # centers.append((x+w/2,y+h/2))
                centers.append((cX, cY))

                t = t + 1
                cv.putText(feed, "Frame #" + text, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # cv.imwrite("feed_thresh%d.jpg" % count , feed_thresh)
        else:
            continue
    return (feed, centers)


def record_vid(frameA, frameB, frameC, frameD):
    output = np.zeros((h * 2, w * 2, 3), dtype="uint8")
    output[0:h, 0:w] = cv.cvtColor(frameA, cv.COLOR_GRAY2BGR)
    output[0:h, w:w * 2] = cv.cvtColor(frameB, cv.COLOR_GRAY2BGR)
    output[h:h * 2, w:w * 2] = cv.cvtColor(frameC, cv.COLOR_GRAY2BGR)
    output[h:h * 2, 0:w] = cv.cvtColor(frameD, cv.COLOR_GRAY2BGR)

    writer.write(output)
    cv.imshow("Output", output)
    return


def create_folder():
    image_folder_name = '\images'
    path = os.getcwd()
    Newdir_path = path + image_folder_name
    try:
        os.mkdir(Newdir_path)
    except OSError:
        print("Creation of the directory %s failed" % Newdir_path)
    os.chdir(Newdir_path)
    print("current dir.", os.getcwd())


def add_marker(frame, x, y):
    # Create a named colour
    black = 0
    # Change one pixel
    if x + 10 < frame.shape[0] and y + 10 < frame.shape[0]:
        for i in range(1, 10):
            frame[x + i, y + i] = black
            frame[x - i, y - i] = black
            frame[x + i, y - i] = black
            frame[x - i, y + i] = black
    return


def calc_centers_movements(Frames_centers):
    print("----------------------------------------------------------------", "\n", "\n")
    print("Calculating Results...", "\n")
    print("----------------------------------------------------------------", "\n", "\n")
    for cen in range(len(Frames_centers[0][1])):
        print("Center#", cen + 1, "(originally was at", Frames_centers[0][1][cen], ")", "\n")
        RSS_list = []
        DX_list = []
        DY_list = []
        for frame in range(1, len(Frames_centers)):
            if len(Frames_centers[frame][1]) == len(Frames_centers[0][1]):
                print("The diffrence between frames:", (Frames_centers[0][0], "Frame" + str(Frames_centers[frame][0])),
                      "is:")
                tup1 = Frames_centers[0][1][cen]  # First Frame - benchmark
                tup2 = Frames_centers[frame][1][cen]  # Any other frames
                distances = calc_distance(tup1, tup2)
                RSS = distances[0]
                DX = distances[1]
                DY = distances[2]
                RSS_list.append(RSS)
                DX_list.append(DX)
                DY_list.append(DY)
                print("Magnitude (RSS):", '%.2f' % RSS, '[Pixel]\n', "dx=", DX, '[Pixel]', "dy=", DY, '[Pixel]\n')
        print("Average displacement for Center #", cen + 1, ":")
        print("Avg. RSS - ", '%.2f' % np.mean(RSS_list), '[Pixel]')
        print("Avg. DX - ", '%.2f' % np.mean(DX_list), '[Pixel]')
        print("Avg. DY - ", '%.2f' % np.mean(DY_list), '[Pixel]\n')
        print("RMS displacement for Center #", cen + 1, ":")
        print("RMS. RSS - ", '%.2f' % np.median(RSS_list), '[Pixel]')
        print("RMS. DX - ", '%.2f' % np.median(DX_list), '[Pixel]')
        print("RMS. DY - ", '%.2f' % np.median(DY_list), '[Pixel]\n')
        print("----------------------------------------------------------------", "\n", "\n")

    return


def plot_center_motion(center_number, Frames_centers):
    x_movements = []
    y_movements = []
    frames_count = []  # [0(ff),1,2,3.....]
    # initieallizing ff movements....
    x_movements.append(0)
    y_movements.append(0)
    frames_count.append(0)
    for frame in range(1,
                       len(Frames_centers)):  # NOTE--- notation: Frames_centers[frame_number][0=frame_number_ID 1=centers_list][center_number][0=c_x 1=cy]
        if len(Frames_centers[frame][1]) == len(Frames_centers[0][1]):
            frame_count = Frames_centers[frame][0]
            frames_count.append(frame_count + 1)  # [0(ff),1,2,3.....]
            x_mov = Frames_centers[frame][1][center_number][0] - Frames_centers[0][1][center_number][
                0]  # positive X is to thr right
            y_mov = -1 * (Frames_centers[frame][1][center_number][1] - Frames_centers[0][1][center_number][
                1])  # positive Y is down
            x_movements.append(x_mov)
            y_movements.append(y_mov)
    print("center Number", center_number, "is", Frames_centers[0][1][center_number])
    plt.plot(frames_count, x_movements, label="X disp.")
    plt.plot(frames_count, y_movements, label="Y disp.")
    plt.xlabel('Frame #')
    # Set the y axis label of the current axis.
    plt.ylabel('Center Displacement [Pixel]')
    # Set a title of the current axes.
    plt.title('X,Y center displacement ')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()
    return (frames_count, x_movements, y_movements)


def calc_distance(tuple_a, tuple_b):
    dist = math.sqrt((tuple_b[0] - tuple_a[0]) ** 2 + (tuple_b[1] - tuple_a[1]) ** 2)
    dx = abs(tuple_b[0] - tuple_a[0])
    dy = abs(tuple_b[1] - tuple_a[1])
    return dist, dx, dy


firstFrame = None
Frames_centers = []
count = 0
######------------######------------######------------######------------######------------

# Fill Analysis Parameters Here:
root = tk.Tk()
root.withdraw()
cap_path = filedialog.askopenfilename()
print("cap_path", cap_path)
# cap_path = os.path.join(os.getcwd(),cap_path)
cap = cv.VideoCapture(cap_path)
# cap = cv.VideoCapture(0)
wait_key = 100  # Time in ms between frames

save_frames = True
analyze_frames = True

# a=420## Set analysis window bounds [pixels]
# b=220# #y 1080q
# width=None #2600 #160 #300 #1050 #200 #desired window's width ---- For original frame size without cropping ---> width=None
min_max_countour = (500, 30000)  # defines the min and max contour areas (pixel**2) to look for
min_max_thresh = (150, 255)  # (120,255) # defines the thresh level for image contour

######------------######------------######------------######------------######------------

# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv.VideoWriter_fourcc(*'MJPG')
writer = None
(h, w) = (None, None)
zeros = None

if save_frames is True:
    create_folder()

global ROIs
ret, frame1 = cap.read()
cap.set(cv.CAP_PROP_POS_FRAMES, 1)

ROIs = select_ROI(frame1)
print(ROIs, type(ROIs))

frame1 = frame1[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
ROIs2 = select_ROI(frame1)
ROIs = np.append(ROIs, ROIs2, axis=0)

# ROI_1 = frame1[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]  ####11 Crop area of the genral frame
# ROI_2 = frame1[ROIs[1][1]:ROIs[1][1] + ROIs[1][3], ROIs[1][0]:ROIs[1][0] + ROIs[1][2]]  ####2.region for center 1
# ROI_3 = frame1[ROIs[2][1]:ROIs[2][1] + ROIs[2][3], ROIs[2][0]:ROIs[2][0] + ROIs[2][2]]  ####3. region for center 2

# cv.imshow('1', ROI_1)
# cv.imshow('2', ROI_2)
# cv.imshow('3', ROI_3)

# cv.waitKey(0)
# cv.destroyAllWindows()
print(ROIs, type(ROIs))

prvs_ff = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)  # frame1 frame in Gray
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
ret = True
while (ret):
    ret, frame2 = cap.read()
    # check if ret is False
    if ret is False:
        print("Done!\n")
        cap.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()
        break

    frame2 = frame2[ROIs[0][1]:ROIs[0][1] + ROIs[0][3], ROIs[0][0]:ROIs[0][0] + ROIs[0][2]]
    if firstFrame is None:
        firstFrame = prvs_ff
        cv.imwrite('firstframe.jpg', firstFrame)
        # add_marker(firstFrame,60,61)
        print("Reading First Frame:")
        print("Frame #", "First Frame")
        ff_centers = plot_contours_ff(firstFrame, min_max_countour, save_frames, min_max_thresh)
        Frames_centers.append(["First Frame", ff_centers])  # (frame#,(center_x,center_y))
    if not ret:
        cap.release()
        cv.destroyAllWindows()
        print('Done!\n')
    nex = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    flow = cv.calcOpticalFlowFarneback(firstFrame, nex, None, 0.5, 5, 20, 5, 7, 1.5, 2)  # Dens Flow
    """Each element in that flow matrix is a point that represents the displacement 
    of that pixel from the prev frame. Meaning that you get a point with x and y values
    (in pixel units) that gives you the delta x and delta y from the last frame"""

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])  # flow[...,0] is delta_x array flow[...,1] is delta_y array
    delta_x = flow[..., 0]
    delta_y = flow[..., 1]

    # add_marker(nex,97,113)# (1080, 1920)

    # Capture Writer Settings
    if writer is None and save_frames is True:
        (h, w) = nex.shape[:2]
        output = 'Output.avi'
        fps = int(3 * wait_key / 1000)
        writer = cv.VideoWriter(output, fourcc, 5, (w * 2, h * 2), True)
        zeros = np.zeros((h, w), dtype="uint8")

    # Masks
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Operation On Frame -------
    frameDelta = cv.absdiff(firstFrame, nex)
    thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]
    # thresh = cv.dilate(thresh, None, iterations=2)

    # Frame Counter
    print('Read a new frame: ', nex.shape)
    print('Frame #', str(count) + '\n')
    count += 1

    # Show The Frames
    if save_frames is not True:
        print("Displaying frames...")
        cv.imshow('first frame', firstFrame)
        contour_results = plot_contours(nex, count - 1, min_max_countour, min_max_thresh)
        feed, centers = contour_results[0], contour_results[1]
        cv.imshow('feed', feed)
        # cv.imshow('Dense Optical Flow',bgr)
        # cv.imshow('thresh',thresh)
        # cv.imshow('Absdiff',frameDelta)

    # Saving The frames as JPEG files and Output Video
    if save_frames is True:
        print("saving frames...")
        contour_results = plot_contours(nex, count - 1, min_max_countour, min_max_thresh)
        feed, centers = contour_results[0], contour_results[1]
        count -= 1
        # cv.imwrite("feed%d.jpg" % count , feed)
        # cv.imwrite("bgr%d.jpg" % count , bgr)
        # cv.imwrite("thresh%d.jpg" % count , thresh)
        # cv.imwrite("frameDelta%d.jpg" % count , frameDelta)
        record_vid(feed, firstFrame, frameDelta, thresh)
        count += 1

    # print to the screen the deta of movements
    if analyze_frames is True:
        Frames_centers.append([count - 1, centers])  # (frame#,(center_x,center_y))

    key = cv.waitKey(wait_key) & 0xFF

    # Release
    if key == ord("q"):
        cap.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()
        break
    elif key == ord('s'):
        cv.imwrite('feedsave.png', feed)
    prvs = next
    continue

# Final Results:
if analyze_frames is True:
    print("\n-------------------------------\n")
    print("-------------------------------\n")
    print("Final Results-------------------------------\n")
    print("-------------------------------\n")
    # calc_centers_movements(Frames_centers)

    # frames_count,x_movements,y_movements=plot_center_motion(0,Frames_centers)

cwd = os.getcwd()
df_title = os.path.join(cwd, "centers_displ.csv")

centers_df = pd.DataFrame()
print("no of centers:", Frames_centers[1])

frame_no_lst = [f[0] for f in Frames_centers]
centers = [f[1] for f in Frames_centers]
cent_columns = {}

for i in range(len(centers[0])):
    key_str = "Center_" + str(i) + "_X"
    c_x = [c[i][0] for c in centers]
    cent_columns[key_str] = c_x
    key_str = "Center_" + str(i) + "_Y"
    c_y = [c[i][1] for c in centers]
    cent_columns[key_str] = c_y

centers_df["Frame"] = frame_no_lst
for k, v in cent_columns.items():
    centers_df[k] = v

centers_df.to_csv(df_title)










