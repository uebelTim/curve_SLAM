import cv2
from ultralytics import YOLO
import ultralytics as u
import numpy as np
import torch
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.optimize import fsolve
import time
import matplotlib.patches as mpatches


#get path of this file
HOME = os.path.abspath('')
print(HOME)
DEBUG = False

def otsuThresholding(img, thresh=0, maxval=255):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OTSU thresholding
    return cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


def get_line_endpoints(line):
    line =np.array(line,dtype=np.int32).squeeze()
    #sort line by x and y
    line = sorted(line,key=lambda x: (x[0],x[1]))
    #get endpoints
    s = line[0]
    e = line[-1]
    return s,e

def filter_outliers(points):
    x,y = points.T
    dist = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    mean = np.mean(dist)
    #remove points that are more than 3*mean
    idx = np.where(dist < 3*mean)
    return points[idx]

def split_into_lanelines(contour):
    points = contour.squeeze()
    non_straight_segments = []
    straight_segments = []
    for i in range(len(points) - 4):
        x1, y1 = points[i]
        x2, y2 = points[i+2]
        x3, y3 = points[i+4]
        # Calculate the slopes of the line segments
        slope1 = (y2 - y1) / (x2 - x1) if x2-x1 != 0 else 100_000
        slope2 = (y3 - y2) / (x3 - x2) if x3-x2 != 0 else 100_000
        # Check if the slopes are equal within a certain tolerance
        if abs(slope1 - slope2) > 0.05:
            non_straight_segments.append(points[i+2])
        else:
            straight_segments.append(points[i+2])

    #[ToDo]only remove straight segements if the straight segments are not the longest segments
    non_straight_segments = np.array(non_straight_segments,dtype=np.int32)
    straight_segments = np.array(straight_segments,dtype=np.int32)

    #calcualte distance between points
    x,y = non_straight_segments.T
    dist = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    #get points of the two biggest distances
    idx = np.argpartition(dist, -2)[-2:]
    for i in idx:
        #add i+1 to idx
        idx = np.append(idx,i+1)
    segmentendpoints = non_straight_segments[idx]
    #split into left and right as points between the two biggest distances
    left=[]
    right=[]
    #sort idx
    idx = np.sort(idx)
    right_ids = np.arange(idx[1],idx[2])
    all_ids = np.arange(len(non_straight_segments))
    #rightids is all ids except left ids
    left_ids = np.setdiff1d(all_ids,right_ids)
    left = non_straight_segments[left_ids]
    right = non_straight_segments[right_ids]
    left = filter_outliers(left)
    right = filter_outliers(right)
    
    if DEBUG:
        img = np.zeros((442,640,3),dtype=np.uint8)
        #draw contour before processing
        cv2.drawContours(img,[contour],-1,(255,255,255),1)
        #draw non straight segments and straight segments
        for point in non_straight_segments:
            cv2.circle(img,(point[0],point[1]),1,(255,0,0),2)
        for point in straight_segments:
            cv2.circle(img,(point[0],point[1]),1,(0,0,255),2)
        #draw endpoints
        for point in segmentendpoints:
            cv2.circle(img,(point),1,(0,255,0),15)
        plt.imshow(img)
        plt.title('straight and non straight segments')
        #make legend; red = non straight, blue = straight, green = endpoints
        red_patch = mpatches.Patch(color='red', label='non straight segments')
        blue_patch = mpatches.Patch(color='blue', label='straight segments')
        green_patch = mpatches.Patch(color='green', label='endpoints')
        plt.legend(handles=[red_patch,blue_patch,green_patch])
        plt.show()

        image = np.zeros((442,640,3),dtype=np.uint8)
        #draw contour before processing
        cv2.drawContours(image,[contour],-1,(255,255,255),1)
        for point in left:
            cv2.circle(image,(point[0],point[1]),1,(255,0,0),2)
        for point in right:
            #make cross
            cv2.drawMarker(image,(point[0],point[1]),(0,0,255),cv2.MARKER_CROSS,10,1)
            #cv2.cross(image,(point[0],point[1]),1,(0,0,255),2)
        plt.imshow(image)
        plt.title('left and right points')
        plt.show()

    return left,right

def get_lane_contour(img):
    #invert image
    img = cv2.bitwise_not(img)
    #find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    lane_point = [320, 400]
    for i in range(4):
        #check if lane point is inside contour
        inside = cv2.pointPolygonTest(contours[i],(lane_point),False)
        if inside == 1:
            lane_contour = contours[i]
            break

    return lane_contour
    
def transform_image(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    otsu = otsuThresholding(img,thresh=240)
    M =  get_birdview_matrix(otsu[1],CT=0.4,borderColor=[255])
    return M




def detectSegments(frame, model):
    #img = cv2.resize(frame,(640,640))
    results = model.predict(frame, show=False)
    masks = results[0].masks  # Masks object

    classes     = results[0].boxes.cls.cpu().numpy()
    conf        = results[0].boxes.conf.cpu().numpy()
    boxes_xyxy  = results[0].boxes.xyxy.cpu().numpy()

    segments = []
    if not masks == None:
        for i, segment in enumerate(masks.segments):
            segments.append([segment*[640,640],boxes_xyxy[i],classes[i], conf[i]])
    return segments

def get_normal_lanes(segments):
    segs =[]
    for segment in segments:
        if segment[2] == 2.0:
            segs.append(segment[0][:])
    return segs

def get_laneline_segments(img):
    model = YOLO(f'{HOME}/models/LaneSegmYolov8s/weights/best.pt')
    classes = model.names
    #print('classes: ',classes)
    segments = detectSegments(img, model)
    lanes = get_normal_lanes(segments)
    return lanes



#draw filled in lanes
def get_filled_lanes_img(img,lanes):
    h,w = img.shape[:2]
    empty  =np.zeros((h,w,1),dtype=np.uint8)
    for lane in lanes:
        cv2.fillPoly(empty, np.int32([lane]), (255))
    return empty

#Transform to birdeye view
def get_birdview_matrix(img, CT=0.34, CB=0.15, CH=0.31, CW=0.7):
    #Groesse und Punkte des Ausgangsbildes
    if len(img.shape) == 3:
        (hoeheOrg,breiteOrg,tiefeOrg) = img.shape
    else:
        (hoeheOrg,breiteOrg) = img.shape
    top_Left = [int(hoeheOrg*(CT)),0]
    top_Right = [int(hoeheOrg*(CT)),breiteOrg]
    bottom_Left = [int(hoeheOrg*(1-CB)),0]
    bottom_Right = [int(hoeheOrg*(1-CB)),breiteOrg]
    ptsArea = [top_Left,top_Right,bottom_Right,bottom_Left]
    ptsAreaFlipped = np.array([np.flip(ptsArea,1)],np.float32)
    #ptsArea = np.array([ptsArea],np.int32)
    #Groesse Ziels und Zielpunkte
    breiteZiel = breiteOrg
    hoeheZiel = int(round((1-CH)*hoeheOrg))
    goal_top_Left = [0,0]
    goal_top_Right = [0,breiteZiel]
    goal_bottom_Left = [hoeheZiel,int(breiteZiel*(CW/2))]
    goal_bottom_Right = [hoeheZiel,int(breiteZiel*(1-(CW/2)))]
    ptsGoal = [goal_top_Left,goal_top_Right,goal_bottom_Right,
    goal_bottom_Left]
    ptsGoalFlipped = np.array([np.flip(ptsGoal,1)],np.float32)
    #ptsGoal = np.array([ptsGoal],np.int32)
    #transformation
    M = cv2.getPerspectiveTransform(ptsAreaFlipped, ptsGoalFlipped)
    dsize = (breiteZiel,hoeheZiel)
    #imgTrans = cv2.warpPerspective(img, M, dsize,borderMode=cv2.BORDER_CONSTANT, borderValue=borderColor)

    return M, dsize

def transform_img_to_birdview(img,M,dsize,borderColor=[0,0,0]):
    img = img.squeeze()
    imgTrans = cv2.warpPerspective(img, M, dsize,borderMode=cv2.BORDER_CONSTANT, borderValue=borderColor)
    return imgTrans

def transform_line_to_birdview(line,M):
    line = line.squeeze().reshape(-1,1,2)
    line = line.astype(np.float32)
    line = cv2.perspectiveTransform(line,M)
    line = line.reshape(-1,2).astype(np.int32)
    return line

#get skeleton of lanes
def get_lanelines_centerline(img):
    #to binary image if not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #blur
    img = cv2.GaussianBlur(img,(5,5),5)
    #make bottom 2 lines of img black
    img[img.shape[0]-2:img.shape[0],0:img.shape[1]] = 0
    skeleton = cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    lanelines = []
    contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #sort contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i in range(2):
        c = contours[i]
        #aproxximate contour
        epsilon = 0.01 #contour_length // desired_points
        aprox = cv2.approxPolyDP(c,epsilon,True)
        lanelines.append(aprox)

    return lanelines


#get endpoints of a curved line
def get_endpoints(line):
    #from greyscale to rgb
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x,y,w,h = cv2.boundingRect(line)
    x_s1, x_s2 = x, x+w
    y_s1, y_s2 = y, y+h
    points_s =[[x_s1,y_s1],[x_s2,y_s1]]
    points_e = [[x_s1,y_s2],[x_s2,y_s2]]
    #check if one of the points lies on the line
    dist_s = []
    for i, point in enumerate(points_s):
        p = (point[0],point[1])
        dist = cv2.pointPolygonTest(line, p, True)
        #print('dist: ',dist)
        dist_s.append(dist)
    p_s = points_s[np.argmin(np.abs(dist_s))]
    dist_e = []
    for i, point in enumerate(points_e):
        p = (point[0],point[1])
        dist = cv2.pointPolygonTest(line, p, True)
        #print('dist: ',dist)
        dist_e.append(dist)
    p_e = points_e[np.argmin(np.abs(dist_e))]
    #change p_s and p_e if p_e[1] is smaller than p_s[1]
    if p_e[1] < p_s[1]:
        p_s, p_e = p_e, p_s

    return p_s, p_e

#calculate stepsizes to reduce a list perfectly to a desired length
def find_stepsizes(length, desired_length):
    diff = abs(length - desired_length)
    #print('diff: ',diff)
    stepsizes = []
    step =2
    while  diff > 0:
        r = length // step
        #check if it is a whole number
        if r %1 == 0:
            if diff - r >= 0:
                stepsizes.append(step)
                diff -= r
                length -= r
                step =2
        step += 1
    return stepsizes

def remove_nth_elements(array,n):
    #check if n is a list
    if isinstance(n, list):
        #print('n is a list')
        for i in n:
            #print('i: ',i)
            array = remove_nth_elements(array,i)
        return array
    arr=np.array(array)
    indices = np.arange(len(arr))
    mask = (indices + 1) % n == 0
    arr = arr[~mask]
    return arr

#make shure that the two lines have the same number of points
def equalize_pixel_lines(line1, line2,nr_points=100):
    #nr_points is minimum of length of line1 and line2 and the desired number of points
    nr_points = min(len(line1),len(line2),nr_points)
    steps1 = find_stepsizes(len(line1),nr_points)
    steps2 = find_stepsizes(len(line2),nr_points)
    line1 = remove_nth_elements(line1,steps1)
    line2 = remove_nth_elements(line2,steps2)
    return line1, line2

def remove_duplicates_from_line(line):
    line = np.array(line)
    line = np.unique(line,axis=0)
    return line


def get_contour_pixels(contour,img):
    image =np.zeros_like(img)
    #cv2.polylines(image, [contour], -1, (255, 255, 255), 1,lineType = cv2.LINE_4)
    cv2.polylines(image, [contour], False, (255, 255, 255), 1)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pixels = np.where(image == 255)
    #return pixels as [x,y]
    pixels = np.array([pixels[1],pixels[0]]).T
    return pixels

     
#get middle line between lanelines
def get_middle_line(lanelines,img):
    image = np.zeros_like(img)
    lines_pixels = []
    for line in lanelines:
        line_pixels = get_contour_pixels(line,img)
        lines_pixels.append(line_pixels)
    #equalize pixel lines
    line1, line2 = equalize_pixel_lines(lines_pixels[0],lines_pixels[1])
    #iterate over both lines and get the middle point
    middle_line = []
    #enumerate over all points of both lines
    for i in range(len(line1)):
        point1 = line1[i]
        point2 = line2[i]
        p1_x,p1_y = point1
        p2_x,p2_y = point2
        mp_x = int((p1_x + p2_x) / 2)
        mp_y = int((p1_y + p2_y) / 2)
        middle_point = [mp_x,mp_y]
        middle_point = np.array(middle_point, dtype=np.int32)
        middle_line.append(middle_point)
    middle_line = np.array(middle_line)
    middle_line = remove_duplicates_from_line(middle_line)
    #add dimension on axis 1
    middle_line = np.expand_dims(middle_line, axis=1)
    #draw middle line and lanelines
    # for line in lanelines:
    #     cv2.drawContours(image, [line], -1, (0,255,0), 1)
    # #aprox middle line with a line
    # cv2.polylines(image, [middle_line], False, (255,0,0), 1)
    #cv2.drawContours(image, [middle_line], -1, (255,0,0), 1)
    #get endpoints of middle line

    return middle_line

def draw_estimated_circle(image,radius,cp,s,e,mp):
    #draw middle_line
    image = np.zeros_like(final_img)
    cv2.polylines(image, [middle_line], False, (255,0,0), 1)
    #draw endpoints
    cv2.drawMarker(image, s, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    cv2.drawMarker(image, e, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #draw middle point
    cv2.drawMarker(image, mp, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #draw center point
    cv2.drawMarker(image, cp, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #draw circle around center point with radius = radius
    cv2.circle(image, cp, int(radius), (0, 255, 0), 1)
    #draw line between endpoints
    cv2.line(image, s, e, (0, 255, 0), 1)
    #draw line between middle point and line between endpoints
    #get middle point between endpoints
    m_x = (s[0]+e[0])/2
    m_y = (s[1]+e[1])/2
    m =np.array([m_x,m_y],dtype=np.int32)
    cv2.line(image, mp, m, (0, 255, 0), 1)
    plt.imshow(image)
    plt.title('curvature')
    plt.show()

def correct_centerpoint(cp,s,e,left_curve):
    #fsolve only gives one solution. but there are two solutions for the centerpoint
    #so we have to check which one is the correct one
    m = (s[1]-e[1])/(s[0]-e[0])
    #get y-intercept
    b = s[1] - m*s[0]
    #indicates if cp is left or right of line between endpoints
    point_left =False
    #check if cp is left or right of line between endpoints
    if cp[1] < m*cp[0]+b:
        point_left = True
    if left_curve == point_left:
        #correct cp
        return cp
    else:
        #wrong cp
        #move cp to other side of line between endpoints
        #get middle point between endpoints
        m_x = (s[0]+e[0])/2
        m_y = (s[1]+e[1])/2
        m =np.array([m_x,m_y],dtype=np.int32)
        #get vector from middle point to cp
        v = cp-m
        #get vector from middle point to other side of line between endpoints
        v2 = np.array([-v[0],-v[1]],dtype=np.int32)
        #get new cp
        cp = m+v2
        return cp

def calculate_centerpoint(radius,s,e):
    left_curve = False
    if radius < 0:
        left_curve = True
    # x, y = symbols("x y")
    # eq1 = Eq((x-e[0])**2 + (y-e[1])**2, radius**2)
    # eq2 = Eq((x-s[0])**2 + (y-s[1])**2, radius**2)
    # sol = solve((eq1, eq2), (x, y))
    # print('sol: ',sol)
    # cp=np.array(sol[1],dtype=np.int32)
    # print('cp: ',cp)

    def equations(p):
        x, y = p
        return [(x-e[0])**2 + (y-e[1])**2- radius**2, (x-s[0])**2 + (y-s[1])**2- radius**2 ]

    x, y =  fsolve(equations, (1, 1))
    cp = np.array([x,y],dtype=np.int32)
    cp = correct_centerpoint(cp,s,e,left_curve)
   
    return cp
    

def calculate_curvature(middle_line, endpoints):
    # 2. Ansatz für die Berechnung der Krümmung nach https://www.mathopenref.com/arcradius.html
    s,e = endpoints
    #get distance between endpoints
    d = np.sqrt((s[0]-e[0])**2 + (s[1]-e[1])**2)
    #get width of curve (distance between middle point and line between endpoints)
    #mp = point in middle of curve
    mp = middle_line[len(middle_line)//2].squeeze()
    #get line between endpoints
    #m = slope
    m = (s[1]-e[1])/(s[0]-e[0])
    #b = y-intercept
    b = s[1] - m*s[0]
    #calculate distance between middle point and line between endpoints
    h = (m*mp[0] - mp[1] + b)/np.sqrt(m**2 + 1)
    #calculate radius
    radius = (h/2)+ d**2/(8*h)
    #calculate centerpoint
    cp = calculate_centerpoint(radius,s,e)
    #draw_estimated_circle(final_img,abs(radius),cp,s,e,mp)

    return radius,cp


def show_estimated_circle(birdview_img,endpoints,middle_line,radius,cp):
    #draw circle
    image = np.zeros_like(birdview_img)
    cv2.polylines(image, [middle_line], False, (255,0,0), 1)
    #draw endpoints
    cv2.drawMarker(image, endpoints[0], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    cv2.drawMarker(image, endpoints[1], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #draw center point
    cv2.drawMarker(image, cp, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #draw circle around center point with radius = radius
    cv2.circle(image, cp, int(radius), (0, 255, 0), 1)
    plt.imshow(image)
    plt.title('estimated radius')
    plt.show()


def get_middleLine_YOLO(frame):
    laneline_segments = get_laneline_segments(frame)
    filled_lanes_img = get_filled_lanes_img(frame,laneline_segments) 
    M,dsize = get_birdview_matrix(filled_lanes_img)
    birdview_img =  transform_img_to_birdview(filled_lanes_img,M,dsize)
    lanelines = get_lanelines_centerline(birdview_img)
    middle_line = get_middle_line(lanelines, birdview_img) 
    if DEBUG:
         #show original frame
        #from cv2 to plt
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame)
        plt.title('Original Frame')
        plt.show()
        #show filled lanes
        plt.imshow(filled_lanes_img,cmap='gray')
        plt.title('Filled Lanes')
        plt.show()
        #show birdview
        plt.imshow(birdview_img,cmap='gray')
        plt.title('Birdview')
        plt.show()
        #draw lanelines and middle line on birdview
        birdview_img = cv2.cvtColor(birdview_img, cv2.COLOR_GRAY2RGB)
        for i in range(len(lanelines)):
            cv2.drawContours(birdview_img, [lanelines[i]], -1, (0,255,0), 2)
        cv2.polylines(birdview_img, [middle_line], False, (255,0,0), 2)
        #draw endpoints
        plt.imshow(birdview_img)
        plt.title('Birdview with lanelines and middle line')
        plt.show()

    return middle_line

def get_middleLine_Otsu(img):
    start_max = time.perf_counter()
    #blur image
    img = cv2.GaussianBlur(img,(5,5),0)
    otsu = otsuThresholding(img,thresh=240)
    #otsu_birdsview =  transform_to_birdview(otsu[1],CT=0.4,borderColor=[255,255,255])
    M,dsize =  get_birdview_matrix(otsu[1],CT=0.4)
    otsu_birdsview =  transform_img_to_birdview(otsu[1],M,dsize,borderColor=[255])
    lane_contour = get_lane_contour(otsu_birdsview)
    #lane_contour = transform_line_to_birdview(lane_contour,M)
    #split lane contour into left and right
    left,right = split_into_lanelines(lane_contour)
    middle_line = get_middle_line([left,right],otsu_birdsview)
    middle_line =middle_line.squeeze()
    if DEBUG:
        print('debug')
        end_max = time.perf_counter()
        print('time to get middle line',end_max-start_max)
        plt.imshow(otsu_birdsview,cmap='gray')
        plt.show()
        print('shape',otsu_birdsview.shape)
        #draw left and right lane and middle line
        img = cv2.cvtColor(otsu_birdsview, cv2.COLOR_GRAY2RGB)
        cv2.polylines(img, [left], False, (0, 255, 0), 2)
        cv2.polylines(img, [right], False, (0, 255, 0), 2)
        cv2.polylines(img, [middle_line], False, (0, 0, 255), 2)
        plt.imshow(img)
        plt.title('middle line')
        plt.show()

    return middle_line

#main function
def get_lane_radius(frame):
    global DEBUG
    DEBUG=False
    middle_line = get_middleLine_Otsu(frame)
    #middle_line = get_middleLine_YOLO(frame)
    endpoints = get_endpoints(middle_line)
    radius,cp = calculate_curvature(middle_line,endpoints)

    if DEBUG:
        M,dsize =  get_birdview_matrix(frame,CT=0.4)
        frame =  transform_img_to_birdview(frame,M,dsize,borderColor=[255,255,255])
        show_estimated_circle(frame,endpoints,middle_line,radius,cp)
    return radius,cp




img = cv2.imread('frame168.jpg')
plt.imshow(img)
plt.show()

radius,cp = get_lane_radius(img)
print('radius:',radius)
print('cp:',cp)
%timeit get_lane_radius(img)




# def lines_to_bv(left,right,M):
#     left = transform_line_into_birdview(left,M)
#     right = transform_line_into_birdview(right,M)
#     return left,right

# img = cv2.GaussianBlur(img,(5,5),0)
# otsu = otsuThresholding(img,thresh=240)
# otsu=otsu[1]
# print('shape otsu:',otsu.shape)
# plt.imshow(otsu,cmap='gray')
# plt.show()
# lane_contour = get_lane_contour(otsu)
# # #split lane contour into left and right
# left,right = split_into_lanelines(lane_contour)
# trans_otsu, M = transform_to_birdview(otsu)
# %timeit transform_line_into_birdview(lane_contour,M)
# lane_contour = transform_line_into_birdview(lane_contour,M)
# #draw lane contour
# otsu_trans = cv2.cvtColor(trans_otsu,cv2.COLOR_GRAY2RGB)
# cv2.drawContours(otsu_trans, [lane_contour], -1, (0,255,0), 1)
# plt.imshow(otsu_trans)
# plt.show()



# image = transform_image(img)
# print('time transform:')
# %timeit transform_image(img)
# lane_contour = get_lane_contour(image)
# print('time get_lane_contour:')
# %timeit get_lane_contour(image)
# #split lane contour into left and right
# left,right = split_into_lanelines(lane_contour)
# print('time split_into_lanelines:')
# %timeit split_into_lanelines(lane_contour)
# middle_line = get_middle_line([left,right],image)
# middle_line =middle_line.squeeze()
# print('time get_middle_line:')
# %timeit get_middle_line([left,right],image)