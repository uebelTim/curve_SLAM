
import cv2
print(cv2.__version__)
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

class CurvatureEstimator:
    def __init__(self, mode = 'otsu'):
        self.mode = mode
        if mode == 'yolo':
            self.model = YOLO(f'{HOME}/models/LaneSegmYolov8s/weights/best.pt')
        self.curvature = 0
        self.radius = 0
        self.cp = None
        self.lanelines = None
    
    #main function; call once per frame; then get required values with getters
    def process_frame(self,frame,debug=False):
        global DEBUG
        DEBUG=debug
        if self.mode == 'otsu':
            middle_line = self.get_middleLine_Otsu(frame)
        elif self.mode == 'yolo':
            middle_line = self.get_middleLine_Yolo(frame)
            
        endpoints = self.get_endpoints(middle_line)
        radius,cp = self.calculate_curvature(middle_line,endpoints)
        self.radius = radius
        self.cp = cp
        self.curvature = 1/radius
    
    def get_curvature(self):
        return self.curvature
    
    def get_radius(self):
        return self.radius
    
    def get_cp(self):
        return self.cp
    
    def get_lanelines(self,nr_points=50):
        left = self.get_contour_pixels(self.left)
        right = self.get_contour_pixels(self.right)
        left,right = self.equalize_pixel_lines(left,right,nr_points=nr_points)
        return [left,right]
        
        
    def get_middleLine_Otsu(self,img):
        #blur image
        img = cv2.GaussianBlur(img,(5,5),0)
        otsu = self.otsuThresholding(img,thresh=240)
        #otsu_birdsview =  transform_to_birdview(otsu[1],CT=0.4,borderColor=[255,255,255])
        M,dsize,borderlines =  self.get_birdview_matrix(otsu,CT=0.4)
        otsu_birdsview =  self.transform_img_to_birdview(otsu,M,dsize,borderColor=[255])
        lane_contour = self.get_lane_contour(otsu_birdsview)
        #lane_contour = transform_line_to_birdview(lane_contour,M)
        #split lane contour into left and right
        left,right = self.split_into_lanelines2(lane_contour,otsu_birdsview,borderlines)
        middle_line= self.get_middle_line([left,right],otsu_birdsview)
        middle_line = middle_line.squeeze()
        self.lanelines = [left,right]
        return middle_line
    
    def otsuThresholding(self,img, thresh=0, maxval=255):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # OTSU thresholding
        otsu = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if DEBUG:
            plt.imshow(otsu[1], cmap='gray')
            plt.title('Otsu thresholding')
            plt.show()
        return otsu[1]
    
    #Transform to birdeye view
    def get_birdview_matrix(self,img, CT=0.34, CB=0.2, CH=0.31, CW=0.7):
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
        #get line equation of birdview borders
        #left line
        l = [(goal_top_Left[1],goal_top_Left[0]),(goal_bottom_Left[1],goal_bottom_Left[0])]
        r = [(goal_top_Right[1],goal_top_Right[0]),(goal_bottom_Right[1],goal_bottom_Right[0])]
        return M, dsize,[l,r]
    
    def transform_img_to_birdview(self,img,M,dsize,borderColor=[0,0,0]):
        img = img.squeeze()
        imgTrans = cv2.warpPerspective(img, M, dsize,borderMode=cv2.BORDER_CONSTANT, borderValue=borderColor)
        if DEBUG:
            plt.imshow(imgTrans,cmap='gray')
            plt.title('birdview')
            plt.show()
        return imgTrans
    
    def get_lane_contour(self,img):
        #img = cv2.GaussianBlur(img,(7,7),0)
        #dilate image
        img = cv2.dilate(img,np.ones((5,5),np.uint8),iterations = 1)
        #open up white lines
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        #invert image
        img = cv2.bitwise_not(img)
        #median blur image
        #img = cv2.medianBlur(img,5)
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
        #approximate contour
        lane_contour = cv2.approxPolyDP(lane_contour,0.001*cv2.arcLength(lane_contour,True),True)
        if DEBUG:
            #draw contour on img
            #make img color
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            #draw all internal contours
            cv2.drawContours(img,[lane_contour],-1,(255,0,0),2)
            #draw lane contour
            plt.imshow(img)
            plt.title('lane contour')
            plt.show()
        return lane_contour
    
    def split_into_lanelines2(self,contour,img,borderlines):
        l, r = borderlines
        l1,l2 =l
        r1,r2 =r
        points = contour.squeeze()
        #remove points that are on the line between l1 and l2
        valid_points = []
        border=[]
        for point in points:
            #calculate distance to line
            distance_l = abs((l2[1]-l1[1])*point[0] - (l2[0]-l1[0])*point[1] + l2[0]*l1[1] - l2[1]*l1[0]) / np.sqrt((l2[1]-l1[1])**2 + (l2[0]-l1[0])**2)
            distance_r = abs((r2[1]-r1[1])*point[0] - (r2[0]-r1[0])*point[1] + r2[0]*r1[1] - r2[1]*r1[0]) / np.sqrt((r2[1]-r1[1])**2 + (r2[0]-r1[0])**2)
            if (distance_l > 5 and distance_r > 5):
                valid_points.append(point)
            else:
                border.append(point)
        valid_points = np.array(valid_points,dtype=np.int32)
        border = np.array(border,dtype=np.int32)
        #remove points at upper and lower border
        valid_points = valid_points[valid_points[:,1] > 2]
        valid_points = valid_points[valid_points[:,1] < img.shape[0]-2]
        #remove points that are in the lowest 20 pixels of the image and not more than 20 pixels away from the middle of the image
        for point in valid_points:
            if point[1] > img.shape[0]-20 and abs(point[0]-img.shape[1]/2) < 50:
                valid_points = np.delete(valid_points,np.where((valid_points==point).all(axis=1)),axis=0)
        #find the two biggest distances between points in valid_points 
        x = valid_points[:,0]
        y = valid_points[:,1]
        #find the biggest distance between adjacent points in valid_points
        dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        #add distance between last and first point
        dist = np.append(dist,np.sqrt((x[0]-x[-1])**2 + (y[0]-y[-1])**2))
        #get the points that are the endpoints of the biggest distance
        idx = np.argsort(dist)[-2:]
        gap_l1 =idx[0]
        gap_l2 =idx[1]
        gap_r1 =idx[0]+1 if idx[0] < len(valid_points)-1 else 0
        gap_r2 =idx[1]+1 if idx[1] < len(valid_points)-1 else 0
        left1 = valid_points[gap_l1]
        right1 = valid_points[gap_r1]
        left2 = valid_points[gap_l2]
        right2 = valid_points[gap_r2]
        left = []
        right = []
        #left are the points between left1 and left2
        #right are the points between right1 and right2
        if gap_l1 < gap_l2:
            left = valid_points[gap_l1:gap_l2]
        else:
            left = valid_points[gap_l2:gap_l1]
        if gap_r1 < gap_r2:
            right = valid_points[gap_r1:gap_r2]
        else:
            right = valid_points[gap_r2:gap_r1]
        left = np.array(left,dtype=np.int32)
        right = np.array(right,dtype=np.int32)
        left = self.filter_outliers(left)
        right = self.filter_outliers(right)
        #draw points in points2 and border in different colors
        if DEBUG:
            image = np.zeros((442,640,3),dtype=np.uint8)
            for point in points:
                cv2.circle(image,(point[0],point[1]),1,(0,255,0),3)
            for point in valid_points:
                cv2.circle(image,(point[0],point[1]),1,(255,0,0),3)
            for point in border:
                cv2.circle(image,(point[0],point[1]),1,(0,0,255),3)
            #draw left2 and right2 as circles
            cv2.circle(image,left1,1,(0,255,255),10)
            cv2.circle(image,right1,1,(0,255,255),10)
            cv2.circle(image,left2,1,(0,255,255),10)
            cv2.circle(image,right2,1,(0,255,255),10)
            #draw borderlines
            cv2.line(image,(l1[0],l1[1]),(l2[0],l2[1]),(0,255,0),1)
            cv2.line(image,(r1[0],r1[1]),(r2[0],r2[1]),(0,255,0),1)
            plt.imshow(image)
            plt.title('points2 and border')
            plt.show()
            image2 = np.zeros((442,640,3),dtype=np.uint8)
            #draw left and right as crosses
            for point in left:
                cv2.drawMarker(image2,(point[0],point[1]),(255,0,0),cv2.MARKER_CROSS,10,1)
            for point in right:
                cv2.drawMarker(image2,(point[0],point[1]),(0,0,255),cv2.MARKER_CROSS,10,1)
            plt.imshow(image2)
            plt.title('left and right')
            plt.show()
        return left, right
    
    #get middle line between lanelines
    def get_middle_line(self,lanelines,img):
        image = np.zeros_like(img)
        lines_pixels = []
        for line in lanelines:
            line = np.array(line)
            #sort line by x and y
            line = line[np.lexsort((line[:,1],line[:,0]))]
            line_pixels = self.get_contour_pixels(line,img)
            lines_pixels.append(line_pixels)
        #equalize pixel lines
        line1, line2 = self.equalize_pixel_lines(lines_pixels[0],lines_pixels[1])
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
        middle_line = self.remove_duplicates_from_line(middle_line)
        #add dimension on axis 1
        middle_line = np.expand_dims(middle_line, axis=1)
        if DEBUG:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for line in lanelines:
                cv2.polylines(img, [line], False, (255, 0, 0), 2)
            cv2.polylines(img, [middle_line], False, (0, 0, 255), 2)
            plt.imshow(img)
            plt.title('middle line')
            plt.show()
        return middle_line
    
    def get_middleLine_YOLO(self,frame):
        laneline_segments = self.get_laneline_segments(frame)
        filled_lanes_img = self.get_filled_lanes_img(frame,laneline_segments) 
        M,dsize = self.get_birdview_matrix(filled_lanes_img)
        birdview_img =  self.curvaturetransform_img_to_birdview(filled_lanes_img,M,dsize)
        lanelines = self.get_lanelines_centerline(birdview_img)
        middle_line = self.calculate_centerpointget_middle_line(lanelines, birdview_img) 
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
    
    def get_laneline_segments(self,img):
        classes = self.model.names
        #print('classes: ',classes)
        segments = self.detectSegments(img)
        lanes = self.get_normal_lanes(segments)
        return lanes
    
    def detectSegments(self,frame):
        #img = cv2.resize(frame,(640,640))
        results = self.model.predict(frame, show=False)
        masks = results[0].masks  # Masks object

        classes     = results[0].boxes.cls.cpu().numpy()
        conf        = results[0].boxes.conf.cpu().numpy()
        boxes_xyxy  = results[0].boxes.xyxy.cpu().numpy()

        segments = []
        if not masks == None:
            for i, segment in enumerate(masks.segments):
                segments.append([segment*[640,640],boxes_xyxy[i],classes[i], conf[i]])
        return segments
    
    def get_normal_lanes(self,segments):
        segs =[]
        for segment in segments:
            if segment[2] == 2.0:
                segs.append(segment[0][:])
        return segs
            
    
    def get_line_endpoints(self,line):
        line =np.array(line,dtype=np.int32).squeeze()
        #sort line by x and y
        line = sorted(line,key=lambda x: (x[0],x[1]))
        #get endpoints
        s = line[0]
        e = line[-1]
        return s,e

    def filter_outliers(self,points):
        x,y = points.T
        dist = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        mean = np.mean(dist)
        #remove points that are more than 3*mean
        idx = np.where(dist < 2*mean)
        return points[idx]


       
    def transform_image(self,img):
        img = cv2.GaussianBlur(img,(5,5),0)
        otsu = self.otsuThresholding(img,thresh=240)
        M =  self.get_birdview_matrix(otsu[1],CT=0.4,borderColor=[255])
        return M


    #draw filled in lanes
    def get_filled_lanes_img(self,img,lanes):
        h,w = img.shape[:2]
        empty  =np.zeros((h,w,1),dtype=np.uint8)
        for lane in lanes:
            cv2.fillPoly(empty, np.int32([lane]), (255))
        return empty

    

    def transform_line_to_birdview(self,line,M):
        line = line.squeeze().reshape(-1,1,2)
        line = line.astype(np.float32)
        line = cv2.perspectiveTransform(line,M)
        line = line.reshape(-1,2).astype(np.int32)
        return line

    #get skeleton of lanes
    def get_lanelines_centerline(self,img):
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
    def get_endpoints(self,line):
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
    def find_stepsizes(self,length, desired_length):
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

    def remove_nth_elements(self,array,n):
        #check if n is a list
        if isinstance(n, list):
            #print('n is a list')
            for i in n:
                #print('i: ',i)
                array = self.remove_nth_elements(array,i)
            return array
        arr=np.array(array)
        indices = np.arange(len(arr))
        mask = (indices + 1) % n == 0
        arr = arr[~mask]
        return arr

    #make shure that the two lines have the same number of points
    def equalize_pixel_lines(self,line1, line2,nr_points=100):
        #nr_points is minimum of length of line1 and line2 and the desired number of points
        nr_points = min(len(line1),len(line2),nr_points)
        steps1 = self.find_stepsizes(len(line1),nr_points)
        steps2 = self.find_stepsizes(len(line2),nr_points)
        line1 = self.remove_nth_elements(line1,steps1)
        line2 = self.remove_nth_elements(line2,steps2)
        return line1, line2

    def remove_duplicates_from_line(self,line):
        line = np.array(line)
        line = np.unique(line,axis=0)
        return line


    def get_contour_pixels(self,contour,img):
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

        
    def correct_centerpoint(self,cp,s,e,left_curve):
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

    def calculate_centerpoint(self,radius,s,e):
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
        cp = self.correct_centerpoint(cp,s,e,left_curve)
    
        return cp
        

    def calculate_curvature(self,middle_line, endpoints):
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
        cp = self.calculate_centerpoint(radius,s,e)
        #draw_estimated_circle(final_img,abs(radius),cp,s,e,mp)

        return radius,cp


    def show_estimated_circle(self,birdview_img,endpoints,middle_line,radius,cp):
        #draw circle
        image = birdview_img.copy()
        cv2.polylines(image, [middle_line], False, (255,0,0), 1)
        #draw endpoints
        cv2.drawMarker(image, endpoints[0], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.drawMarker(image, endpoints[1], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        #draw center point
        cv2.drawMarker(image, cp, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        #draw circle around center point with radius = radius
        print('radius: ',radius)
        print('cp: ',cp)
        #set cp to middle of image if out of bounds
        # if cp[0] < 0 or cp[0] > image.shape[1] or cp[1] < 0 or cp[1] > image.shape[0]:
        #     cp = np.array([image.shape[1]//2,image.shape[0]//2])
        cv2.circle(image, cp, int(abs(radius)), (0, 255, 0), 1)
        plt.imshow(image)
        plt.title('estimated radius')
        plt.show()

#main function
# def get_lane_radius_and_anchorpoints(frame,nr_anchorpoints=50):
#     global DEBUG
#     DEBUG=False
#     middle_line,left,right = get_middleLine_Otsu(frame,nr_points)
#     endpoints = get_endpoints(middle_line)
#     radius,cp = calculate_curvature(middle_line,endpoints)

#     if DEBUG:
#         M,dsize,borderlines =  get_birdview_matrix(frame,CT=0.4)
#         frame =  transform_img_to_birdview(frame,M,dsize,borderColor=[255,255,255])
#         show_estimated_circle(frame,endpoints,middle_line,radius,cp)
#     return radius,cp,left,right



if __name__ == '__main__':
    img = cv2.imread('frame168.jpg')
    plt.imshow(img)
    plt.show()

    # radius,cp = get_lane_radius_and_anchorpoints(img)
    # print('radius:',radius)
    # print('cp:',cp)

    # #%%
    # %timeit get_lane_radius(img)
