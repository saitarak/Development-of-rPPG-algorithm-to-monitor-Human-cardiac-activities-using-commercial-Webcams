import cv2
import heartpy as hp
import xlsxwriter
from threading import *
from queue import Queue
import mediapipe as mp
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

q1 = Queue()
q2 = Queue()
q3 = Queue()
ROI = []
Time = []
count = []

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Creating empty lists for storing mean values of all channels
red_forehead_mean_values = []
green_forehead_mean_values = []
blue_forehead_mean_values = []
red_right_cheek_mean_values = []
green_right_cheek_mean_values = []
blue_right_cheek_mean_values = []
red_left_cheek_mean_values = []
green_left_cheek_mean_values = []
blue_left_cheek_mean_values = []

# Processing thread
def Processing():
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        height, width, _ = image.shape
        # Flip the image horizontally for a later selfie-view display, and convert BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        print(type(image))
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        results = face_mesh.process(image)
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        # Convert the RGB image to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Connecting lines using these points for forehead, right cheek, left cheek regions
        forehead_line = [284, 298, 9, 68, 54, 103, 67, 109, 10, 338, 297, 332, 284]
        right_cheek_line = [391, 423, 266, 330, 280, 401, 361, 288, 397, 410, 322, 391]
        left_cheek_line = [165, 203, 36, 101, 50, 147, 132, 58, 172, 186, 92, 165]
        forehead_line_xy = []
        left_cheek_line_xy = []
        right_cheek_line_xy = []

        if results.multi_face_landmarks:
            x = [54, 284, 391, 361, 132, 165]
            y = [10, 8, 330, 397, 101, 172]
            for i in range(len(x)):
                # Landmark boundaries for all the region of interests are appended to ROI list
                ROI.append(int((results.multi_face_landmarks[0].landmark[x[i]].x) * width))
                ROI.append(int((results.multi_face_landmarks[0].landmark[y[i]].y) * height))
            im = image.copy()
            for face_landmarks in results.multi_face_landmarks:

                for i in forehead_line:
                    # Taking XY boundary coordinate points on forehead region
                    forehead_line_x = int((face_landmarks.landmark[i].x) * width)
                    forehead_line_y = int((face_landmarks.landmark[i].y) * height)
                    forehead_line_xy.append((forehead_line_x, forehead_line_y))
                for point1, point2 in zip(forehead_line_xy, forehead_line_xy[1:]):
                    # Drawing boundary line around forehead region
                    cv2.line(im, point1, point2, [0, 255, 0], 1)

                for i in left_cheek_line:
                    # Taking XY boundary coordinate points on left cheeks region
                    left_cheek_line_x = int((face_landmarks.landmark[i].x) * width)
                    left_cheek_line_y = int((face_landmarks.landmark[i].y) * height)
                    left_cheek_line_xy.append((left_cheek_line_x, left_cheek_line_y))
                for point1, point2 in zip(left_cheek_line_xy, left_cheek_line_xy[1:]):
                    # Drawing boundary line around left cheeks region
                    cv2.line(im, point1, point2, [0, 255, 0], 1)

                for i in right_cheek_line:
                    # Taking XY boundary coordinate points on right cheeks region
                    right_cheek_line_x = int((face_landmarks.landmark[i].x) * width)
                    right_cheek_line_y = int((face_landmarks.landmark[i].y) * height)
                    right_cheek_line_xy.append((right_cheek_line_x, right_cheek_line_y))
                for point1, point2 in zip(right_cheek_line_xy, right_cheek_line_xy[1:]):
                    # Drawing boundary line around right cheeks region
                    cv2.line(im, point1, point2, [0, 255, 0], 1)

            # Collecting the contours at all the region of interests
            forehead_pts = np.array([forehead_line_xy], dtype=np.int32)
            right_cheek_pts = np.array([right_cheek_line_xy], dtype=np.int32)
            left_cheek_pts = np.array([left_cheek_line_xy], dtype=np.int32)

            # Creating mask with zeros in array
            mask = np.zeros(image.shape[:2], np.int8)
            forehead_mask = np.zeros(image.shape[:2], np.int8)
            right_cheek_mask = np.zeros(image.shape[:2], np.int8)
            left_cheek_mask = np.zeros(image.shape[:2], np.int8)

            # The regions are filled with the contours taken and filled with white color
            cv2.fillPoly(forehead_mask, [forehead_pts], 255)
            cv2.fillPoly(right_cheek_mask, [right_cheek_pts], 255)
            cv2.fillPoly(left_cheek_mask, [left_cheek_pts], 255)
            cv2.fillPoly(mask, [forehead_pts,right_cheek_pts,left_cheek_pts], 255)

            # Take the values of pixels that are not with the value of zero
            maskimage_forehead = cv2.inRange(forehead_mask, 1, 255)
            maskimage_right_cheek = cv2.inRange(right_cheek_mask, 1, 255)
            maskimage_left_cheek = cv2.inRange(left_cheek_mask, 1, 255)
            maskimage = cv2.inRange(mask, 1, 255)

            # Results of all the region of interests after combining the original image and masking
            forehead_result = cv2.bitwise_and(image, image, mask=maskimage_forehead)
            right_cheek_result = cv2.bitwise_and(image, image, mask=maskimage_right_cheek)
            left_cheek_result = cv2.bitwise_and(image, image, mask=maskimage_left_cheek)
            result = cv2.bitwise_and(image, image, mask=maskimage)

            # Taking mean of all the pixels at the region of interest
            mean_forehead = cv2.mean(forehead_result, mask=maskimage_forehead)
            mean_right_cheek = cv2.mean(right_cheek_result, mask=maskimage_right_cheek)
            mean_left_cheek = cv2.mean(left_cheek_result, mask=maskimage_left_cheek)

            # All region of interests are taken into consideration
            ROI1 = forehead_result[ROI[1]:ROI[3], ROI[0]:ROI[2]]
            ROI2 = right_cheek_result[ROI[5]:ROI[7], ROI[4]:ROI[6]]
            ROI3 = left_cheek_result[ROI[9]:ROI[11], ROI[8]:ROI[10]]

            # Stacking all the regions RGB channels for better visualization
            # The values will be in RGB order in hstack
            forehead_channels = np.hstack([ROI1[:, :, 2], ROI1[:, :, 1], ROI1[:, :, 0]])
            right_cheek_channels = np.hstack([ROI2[:, :, 2], ROI2[:, :, 1], ROI2[:, :, 0]])
            left_cheek_channels = np.hstack([ROI3[:, :, 2], ROI3[:, :, 1], ROI3[:, :, 0]])

            # Appending mean values of each region RGB.
            red_forehead_mean_values.append(mean_forehead[2])
            green_forehead_mean_values.append(mean_forehead[1])
            blue_forehead_mean_values.append(mean_forehead[0])
            red_right_cheek_mean_values.append(mean_right_cheek[2])
            green_right_cheek_mean_values.append(mean_right_cheek[1])
            blue_right_cheek_mean_values.append(mean_right_cheek[0])
            red_left_cheek_mean_values.append(mean_left_cheek[2])
            green_left_cheek_mean_values.append(mean_left_cheek[1])
            blue_left_cheek_mean_values.append(mean_left_cheek[0])
            count.append((len(red_forehead_mean_values))/30)

        # Store values into list and converted to array
        dictionary = [cap.isOpened(), cap, forehead_channels, right_cheek_channels, left_cheek_channels,
                      red_forehead_mean_values, green_forehead_mean_values, blue_forehead_mean_values,
                      red_right_cheek_mean_values, green_right_cheek_mean_values, blue_right_cheek_mean_values,
                      red_left_cheek_mean_values, green_left_cheek_mean_values, blue_left_cheek_mean_values,count,image]
        arr_dictionary = np.asarray(dictionary, dtype=object)
        q2.put(arr_dictionary)


        # Images are printed below
        cv2.imshow('forehead', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #arr_dictionary[0] = False
            #q2.put(arr_dictionary)
            break

    cap.release()
    cv2.destroyAllWindows()


# Output thread
def animate(i=None):
    samples = q2.get()
    while cap.isOpened():

        # Bandpass filter is used to get filtered data
        def bandpass(data, fs, order, fc_low, fc_hig):
            nyq = 0.5 * fs  # Calculate the Nyquist frequency.
            cut_low = fc_low / nyq  # Calculate the lower cutoff frequency (-3 dB).
            cut_hig = fc_hig / nyq  # Calculate the upper cutoff frequency (-3 dB).
            bp_b, bp_a = sig.butter(order, (cut_low, cut_hig), btype='bandpass')  # Design and apply the band-pass filter.
            bp_data = list(sig.filtfilt(bp_b, bp_a, data))  # Apply forward-backward filter with linear phase.
            return bp_data

        # Plotting of RGB for Forehead as Region of Interest every 10 frames
        if (len(samples[5]) > 15) and ((len(samples[5]) % 10) == 0):
            print(samples[5])
            fig = plt.figure(1)
            plt.cla()
            plt.plot(samples[14],bandpass(samples[5], 30, 2, 0.9, 1.8), color = 'red')
            plt.plot(samples[14],bandpass(samples[6], 30, 2, 0.9, 1.8), color = 'green')
            plt.plot(samples[14],bandpass(samples[7], 30, 2, 0.9, 1.8),color = 'blue',label = 'Left Cheeks')
            plt.xlabel('Time in seconds')
            plt.ylabel('Average intensity of RGB values at Forehead region')
            plt.title('Forehead as Region of Interest filtered RGB plot')
            #legend = plt.legend(loc='lower right', fontsize='x-small')
            #legend.get_frame().set_facecolor('C0')
            plt.pause(0.01)

    plt.ioff()
    plt.show()

    # Fast Fourier Transform Plotting
    data_win_red = bandpass(samples[5], 30, 2, 0.9, 1.8) * np.hanning(len(samples[5]))
    data_win_green = bandpass(samples[6], 30, 2, 0.9, 1.8) * np.hanning(len(samples[6]))
    data_win_blue = bandpass(samples[7], 30, 2, 0.9, 1.8) * np.hanning(len(samples[7]))
    mag_red = 2.0 * np.abs(np.fft.rfft(tuple(data_win_red)) / len(data_win_red))
    mag_green = 2.0 * np.abs(np.fft.rfft(tuple(data_win_green)) / len(data_win_green))
    mag_blue = 2.0 * np.abs(np.fft.rfft(tuple(data_win_blue)) / len(data_win_blue))
    bin_red = np.fft.rfftfreq(len(data_win_red), 1.0 / 30)
    bin_green = np.fft.rfftfreq(len(data_win_green), 1.0 / 30)
    bin_blue = np.fft.rfftfreq(len(data_win_blue), 1.0 / 30)
    plt.subplot(3,1,1)
    plt.plot(bin_red,mag_red)
    plt.xlim((0.0, 5.0))
    plt.xlabel('Number of bins')
    plt.ylabel('Magnitude(Red)')
    plt.title('FFT Plot for Forehead region')
    plt.subplot(3,1,2)
    plt.plot(bin_green,mag_green)
    plt.xlim((0.0, 5.0))
    plt.xlabel('Number of bins')
    plt.ylabel('Magnitude(Green)')
    plt.subplot(3,1,3)
    plt.plot(bin_blue,mag_blue)
    plt.xlim((0.0, 5.0))
    plt.xlabel('Number of bins')
    plt.ylabel('Magnitude(Blue)')
    plt.show()

    # Sending data to Excel sheet
    outWorkbook = xlsxwriter.Workbook('Forehead_filter1.xlsx')
    outSheet = outWorkbook.add_worksheet()
    outSheet.write('A1', 'Red forehead')
    outSheet.write('B1', 'Green forehead')
    outSheet.write('C1', 'Blue forehead')
    outSheet.write('D1', 'Red Right Cheek')
    outSheet.write('E1', 'Green Right Cheek')
    outSheet.write('F1', 'Blue Right Cheek')
    outSheet.write('G1', 'Red Left Cheek')
    outSheet.write('H1', 'Green Left Cheek')
    outSheet.write('I1', 'Blue Left Cheek')

    for item in range(len(red_forehead_mean_values)):
        outSheet.write(item + 1, 0, samples[14][item])
        outSheet.write(item + 1, 0, samples[5][item])
        outSheet.write(item + 1, 1, samples[6][item])
        outSheet.write(item + 1, 2, samples[7][item])
        outSheet.write(item + 1, 3, samples[8][item])
        outSheet.write(item + 1, 4, samples[9][item])
        outSheet.write(item + 1, 5, samples[10][item])
        outSheet.write(item + 1, 6, samples[11][item])
        outSheet.write(item + 1, 7, samples[12][item])
        outSheet.write(item + 1, 8, samples[13][item])
    outWorkbook.close()
    print("excel export completed")

    # Plotting Heart rate and measuring Heart rate parameters
    working_data_red, measures_red = hp.process(np.asarray(bandpass(samples[5], 30, 2, 0.9, 1.8)), sample_rate = 30)
    working_data_green, measures_green = hp.process(np.asarray(bandpass(samples[6], 30, 2, 0.9, 1.8)), sample_rate=30)
    working_data_blue, measures_blue = hp.process(np.asarray(bandpass(samples[7], 30, 2, 0.9, 1.8)), sample_rate=30)
    hp.plotter(working_data_red, measures_red, title='Heart Beat Detection on Forehead Red channel Filtered Signal')
    hp.plotter(working_data_green, measures_green, title='Heart Beat Detection on Forehead Green channel Filtered Signal')
    hp.plotter(working_data_blue, measures_blue, title='Heart Beat Detection on Forehead Blue channel Filtered Signal')
    plt.show()
    for measure_red,measure_green,measure_blue in zip(measures_red.keys(),measures_green.keys(),measures_blue.keys()):
        print('%s: %f' % (measure_red, measures_red[measure_red]))
        print('%s: %f' % (measure_green, measures_green[measure_green]))
        print('%s: %f' % (measure_blue, measures_blue[measure_blue]))

# Threads are called from here and started
if __name__ == '__main__':
    # Webcam capture
    cap = cv2.VideoCapture('01-base.mp4')
    # FPS will be printed
    print("FPS for Webcam Capture: ", cap.get(cv2.CAP_PROP_FPS))
    plt.ion()
    # Threads are started here
    Thread(target=Processing).start()
    Thread(target=animate).start()
    # ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)