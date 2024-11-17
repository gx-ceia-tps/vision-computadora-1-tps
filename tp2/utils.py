import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def get_img_center(img):
    m, n = img.shape
    return m//2, n//2

def generate_deltas(n):
    return list(range(-(n - 1) * 2, (n - 1) * 2 + 1, 4))

def get_focus_matrix(img, rects_i, rects_j, dj, di):
    i, j = get_img_center(img[:, :, 0])
    modified_img = img.copy()
    color = (0, 255, 0)  # bgr
    points = []
    for z in generate_deltas(rects_i):
        for k in generate_deltas(rects_j):
            start_point = (j + k * dj - dj, i + z * di - di)
            end_point = (j + k * dj + dj, i + z * di + di)
            points.append((start_point, end_point))
            cv2.rectangle(modified_img, start_point, end_point, color, thickness=2)

    plt.imshow(modified_img)
    return points

def get_centered_roi(img, area_percentage):
    m, n = img.shape
    center_m, center_n = get_img_center(img)
    dm, dn = int(m*area_percentage)//2, int(n*area_percentage)//2
    # print(dm,dn)
    return img[center_m-dm:center_m+dm, center_n-dn:center_n+dn]

def store_frames(vid):
    print('Storing video frames...')
    count = 0
    success = 1
    while success:
        success, image = vid.read()
        count += 1
        if success:
            cv2.imwrite(f'frames/frame-{count:03}.jpg', image)
    print('Frames stored successfully')


def read_video_and_store_frames(video_path):
    video = cv2.VideoCapture(video_path)
    fps = round(video.get(cv2.CAP_PROP_FPS))

    if os.path.exists('frames'):
        print('Frames folders already created.')
    else:
        print('Creating frames folder.')
        try:
            os.makedirs('frames')
        except Exception as e:
            print(f"An error occurred: {e}")

    store_frames(video)
    return fps

def plot_metric(curr_metric):
    plt.plot(curr_metric)
    plt.ylabel('Image Sharpness Measure')
    plt.xlabel('# frame')
    absolute_max = np.max(curr_metric)
    plt.axhline(absolute_max, linestyle='dashed', color='orange', label=f'max: {absolute_max:0.4f}')
    plt.legend()
    return absolute_max


def get_img_center(img):
    m, n = img.shape
    return m//2, n//2

def get_centered_roi(img, area_percentage):
    m, n = img.shape
    center_m, center_n = get_img_center(img)
    dm, dn = int(m*area_percentage)//2, int(n*area_percentage)//2
    # print(dm,dn)
    return img[center_m-dm:center_m+dm, center_n-dn:center_n+dn]

def get_gray_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_metric_for_centered_roi(frame, metric_func, area_percentage=0.05):
    gray_img = get_gray_img(cv2.imread(frame))
    centered_roi = get_centered_roi(gray_img, area_percentage)
    return metric_func(centered_roi)


def generate_deltas(n):
    return list(range(-(n - 1) * 2, (n - 1) * 2 + 1, 4))


def get_focus_matrix(img, rects_i, rects_j, dj, di):
    i, j = get_img_center(img[:, :, 0])
    modified_img = img.copy()
    color = (0, 255, 0)  # bgr
    points = []
    for z in generate_deltas(rects_i):
        for k in generate_deltas(rects_j):
            start_point = (j + k * dj - dj, i + z * di - di)
            end_point = (j + k * dj + dj, i + z * di + di)
            points.append((start_point, end_point))
            cv2.rectangle(modified_img, start_point, end_point, color, thickness=2)

    # plt.imshow(modified_img)
    return modified_img, points

def get_delta_points(start,end):
    x1,y1 = start
    x2,y2 = end
    dx = x2-x1
    dy = y2-y1
    return x1,y1,dx,dy

def get_delta_points(start,end):
    x1,y1 = start
    x2,y2 = end
    dx = x2-x1
    dy = y2-y1
    return x1,y1,dx,dy

def group_metric(img, points, metric_func, group_func):
    # we might want to group by min, max, mean
    gray_img = get_gray_img(img)
    calculations = []
    for start_point, end_point in points:
        x1,y1,dx,dy = get_delta_points(start_point,end_point)
        calculations.append(metric_func(gray_img[y1:y1+dy,x1:x1+dx]))
    return group_func(calculations)

def bgr_to_rgb(frame_path):
    return cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)


def write_video(filename, rgb_frames, fps):
    m,n,_ = rgb_frames[0].shape
    output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (n, m), isColor=True)
    for im in rgb_frames:
        bgr_im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        output.write(bgr_im)
    output.release()

def draw_focus_matrix(frame_path, threshold, points, metric_fun):
    img = cv2.imread(frame_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    modified_img = rgb_img.copy()
    green = (0 , 255, 0)
    red = (255, 0, 0)
    total_calc = group_metric(img, points, metric_fun, np.mean)
    for start_point, end_point in points:
        cv2.rectangle(modified_img, start_point, end_point, green if total_calc > threshold else red, thickness=2)
    return modified_img