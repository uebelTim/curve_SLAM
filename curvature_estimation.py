
#%%
import cv2
print('cv2 version:', cv2.__version__,'\n')
import ultralytics as u
from ultralytics import YOLO

import numpy as np
import os
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy.optimize import fsolve
import time
import matplotlib.patches as mpatches
import os
from skimage.draw import polygon
import natsort 
from collections import deque
import math
from scipy.ndimage import gaussian_filter


#get path of this file
HOME = os.path.abspath('')
print(HOME)
DEBUG = False

class CurvatureEstimator:
    def __init__(self, mode = 'otsu'):
        self.mode = mode
        if mode == 'yolo':
            u.checks()
            self.model = YOLO(f'{HOME}/models/LaneSegmYolov8n/weights/best.pt')
        self.curvature = 0
        self.radius = 0
        self.lateral_offset = None
        self.heading_angle = None
        self.starting_line = False
        self._endpoints = None

    
    #main function; call once per frame; then get required values with getters
    def process_frame(self,frame,debug=False):
        global DEBUG
        DEBUG=debug
        if DEBUG:
            plt.imshow(frame)
            plt.title('frame')
            plt.show()
        self._frame = frame
        
        cv2.morphologyEx(frame, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        #otsu = self.otsuThresholding(img,thresh=240)
        frame = cv2.resize(frame,(640,640))
        M,dsize =  self.get_birdview_matrix(frame)
        birdview_frame =  self.transform_img_to_birdview(frame,M,dsize)
        self._birdview_img = birdview_frame
        middle_line, poly_coeffs = self.calculate_middle_line(birdview_frame)
        self._middle_line = middle_line
        radius = self.calculate_radius(middle_line,birdview_frame)
        self.radius = radius
        lateral_offset = self.calculate_lateral_offset(middle_line,birdview_frame)
        self.lateral_offset = lateral_offset
        heading_angle = self.calculate_heading_angle(middle_line, birdview_frame, poly_coeffs)
        self.heading_angle = heading_angle
        print('radius: ',radius)
        print('lateral_offset: ',lateral_offset)
        print('heading_angle: ',heading_angle)
        
        
 

        
        
    def get_curvature(self,debug=False):
        if debug and not DEBUG:
           self.show_estimated_circle(self._birdview_img, self._endpoints, self._middle_line, self.radius, self._cp, self._mp)
        radius = self.get_radius()
        curvature = 1/radius
        return curvature
    
    def get_radius(self):
        radius = self.radius
        
        #convert radius from pixels to meter
        #track width is 120 Pixel wich is 0.75m
        radius = radius /120 *0.75
        return radius
    
    def get_lateral_offset(self):
        offset = self.lateral_offset
        #convert offset from pixels to meter
        offset = -offset/120 *0.75
        return offset
    
    def get_heading_angle(self):
        return self.heading_angle
    
    def get_starting_line(self):
        return self.starting_line

    
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
    def get_birdview_matrix(self,img, CT=0.52, CB=0.0, CH=0.30, CW=0.85):
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
        # l = [(goal_top_Left[1],goal_top_Left[0]),(goal_bottom_Left[1],goal_bottom_Left[0])]
        # r = [(goal_top_Right[1],goal_top_Right[0]),(goal_bottom_Right[1],goal_bottom_Right[0])]
        return M, dsize

    def transform_img_to_birdview(self, img,M,dsize,borderColor=[0,0,0]):
        img = img.squeeze()
        imgTrans = cv2.warpPerspective(img, M, dsize,borderMode=cv2.BORDER_CONSTANT, borderValue=borderColor)
        # plt.imshow(imgTrans,cmap='gray')
        # plt.title('birdview')
        # plt.show()
        return imgTrans

    def transform_line_to_birdview(self,line,M):
        if len(line)<2:
            print('line is empty')
            return None
        line = np.array(line,dtype=np.int32)
        line = line.squeeze().reshape(-1,1,2)
        line = line.astype(np.float32)
        line = cv2.perspectiveTransform(line,M)
        line = line.reshape(-1,2).astype(np.int32)
        return line


    def calculate_middle_line(self,image):
        _, binary_image = cv2.threshold(image, 150, 1, cv2.THRESH_BINARY)
        lanelines,middle_line = self.find_lane_lines_fast(binary_image)
        if middle_line is None:
            return None,None
        height, width = image.shape[:2]
        middle_line_smooth, poly_coeffs = self.fit_polynomial(middle_line,degree=2,max_height=height)
        middle_line_smooth = np.array(middle_line_smooth,dtype=np.int32)
        if DEBUG:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            for line in lanelines:
                if line is not None and len(line) > 0:
                    cv2.polylines(img, [line], False, (255,0,0), 2)
            if middle_line is not None and len(middle_line) > 0:
                cv2.polylines(img, [middle_line], False, (0,0,255), 2)
            if middle_line_smooth is not None and len(middle_line_smooth) > 0:
                cv2.polylines(img, [middle_line_smooth], False, (0,255,0), 2)
            plt.imshow(img)
            plt.title('lines in birdview')
            plt.show()
        return middle_line_smooth, poly_coeffs
    
    def find_lane_lines_fast(self, binary_image):
        height, width = binary_image.shape
        if len(binary_image.shape) == 3:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        #blur
        binary_image = cv2.GaussianBlur(binary_image,(3,3),0)
        left = np.full((height, 2), -1)
        right = np.full((height, 2), -1)
        middle =np.full((height, 2), -1)
        left_diffs, right_diffs = np.array([]), np.array([])
        last_left_x, last_right_x = None, None
        left_active, right_active = True, True
        left_x,right_x = 0, width-1
        middle_x = width//2
        start_search_at = width//2
        found_startline = False
        self.starting_line = False
        deviation_threshold =4
        distances =[]
        pixel_distance_between_lanes = 120
        for y in range(height-1, -1, -1):
            #print('y: ',y)
            if y < int(height*3/4):
                print(' reached max height: ',y)
                break
            
            if binary_image[y, start_search_at] == 1:
                if last_left_x is not None and last_right_x is not None:
                    print('found start line',y)
                    found_startline = True
                    left_active, right_active = True, True
                    self.starting_line = True
                    continue
                if y > int(height*3/4+ height//8):
                    print('found start line',y)
                    found_startline = True
                    left_active, right_active = True, True
                    continue
                else:
                    print('found topend',y)
                    break
            row = binary_image[y]
            left_line_indices = np.where( row[:start_search_at])[0]
            right_line_indices = np.where(row[start_search_at:])[0]
            
            if len(left_line_indices) > 0 and  left_active:
                left_x = left_line_indices[-1]
                if last_left_x is None:
                    diff_left =0
                    avg_diff_left =0
                else:
                    diff_left = left_x - last_left_x#current x - previous x
                    #calculate slope of left line	
                    #diff_left = -(left_x - last_left_x)/(y - (y+1))
                    weights = np.arange(1,len(left_diffs)+1)**2
                    avg_diff_left =np.ma.average(left_diffs,weights=weights)
                    mean_diff_left = np.mean(left_diffs)
                    #print('diff_left: ',diff_left,'avg_diff_left: ',avg_diff_left,'mean_diff',mean_diff_left,'y: ',y)
                if abs(diff_left - avg_diff_left) <= deviation_threshold:
                    left[y] = [left_x, y]
                    last_left_x = left_x
                    last_left_y = y
                    left_diffs = np.append(left_diffs,diff_left)
                    if left_diffs.size > 10:
                        left_diffs = np.delete(left_diffs,0)
                    # print('diff_left: ',diff_left,'avg_diff_left: ',avg_diff_left)
                elif y > int(height*7/8):
                    print('deviation left in lower half of image, continue search',y)
                    last_left_x = left_x
                    continue
                else:
                    print('left deactivated',y)
                    print('left deviated; diff_left: ',diff_left,'avg_diff_left: ',avg_diff_left)
                    # print('diff_left: ',diff_left,'avg_diff_left: ',avg_diff_left)
                    # print('left_x: ',left_x,'start_search_at: ',start_search_at)
                    left_active = False
            #deactivate if found lane before and no lane found now
            elif len(left_line_indices) == 0 and last_left_x is not None and left_active and y < int(height*7/8):
                print('left line has ending')
                print('left deactivated',y)
                left_active = False
                
                
            if len(right_line_indices) > 0 and right_active:
                right_x = right_line_indices[0] + start_search_at
                if last_right_x is None:
                    diff_right =0
                    avg_diff_right =0
                else:
                    diff_right = right_x - last_right_x
                    #diff_right = (right_x - last_right_x)/(y - (y+1))
                    weights = np.arange(1,len(right_diffs)+1)**2
                    avg_diff_right = np.ma.average(right_diffs,weights=weights)
                if abs(diff_right - avg_diff_right) <= deviation_threshold:
                    right[y] = [right_x, y]
                    last_right_x = right_x
                    right_diffs = np.append(right_diffs,diff_right)
                    if right_diffs.size > 10:
                        right_diffs = np.delete(right_diffs,0)
                    #print('diff_right: ',diff_right,'avg_diff_right: ',avg_diff_right,'y: ',y)
                elif y > int(height*7/8):
                    print('deviation right in lower half of image, continue search',y)
                    last_right_x = right_x
                    continue
                    # right_diffs = np.append(right_diffs,diff_right)
                    # if right_diffs.size > 10:
                    #     right_diffs = np.delete(right_diffs,0)
                else:
                    print('right deactivated',y)
                    print('right deviated; diff_right: ',diff_right,'avg_diff_right: ',avg_diff_right)
                    right_active = False 
            elif len(right_line_indices) == 0 and last_right_x is not None and right_active:
                print('right deactivated',y)
                print('right line has ending')
                right_active = False
            
            #stop when both lanes end
            if left_active == False and right_active == False:
                break
            #if only one lane found, stop when it ends
            if y < int(height*7/8):
                if left_active == False and last_right_x is None:
                    print('left deactivated and no right',y)
                    break
                if right_active == False and last_left_x is None:
                    print('right deactivated and no left',y)
                    break

            #criteria for calculating middle line
            if len(left_line_indices) > 0 and len(right_line_indices) > 0 and left_active==True and right_active == True:
                #print('both lanes',y)
                distance = right_x - left_x
                if distance < 0.8*pixel_distance_between_lanes:
                    print('distance abnormal: ',distance,'y: ',y)
                    return [None,None],None
                distances.append(right_x - left_x)
                middle_x = left_x + (right_x - left_x)//2
                #print('both lanes',y,'left_x, right_x: ',left_x,right_x,'middle_x: ',middle_x)
            elif len(left_line_indices) > 0 and (len(right_line_indices) ==0  or right_active== False) and left_active==True:
                if len(distances) > 0:
                    middle_x = left_x + distances[-1]//2
                else:
                    middle_x = left_x + pixel_distance_between_lanes//2
                #print('left lane',y)
            elif len(right_line_indices) > 0 and (len(left_line_indices) ==0 or left_active ==False) and right_active==True:
                if len(distances) > 0:
                    middle_x = right_x - distances[-1]//2
                else:
                    middle_x = right_x - pixel_distance_between_lanes//2
                #print('right lane',y)
            else:
                #print('no lane',y)
                pass
            
            if len(left_line_indices) > 0 or len(right_line_indices) > 0:
                #print('middle_x: ',middle_x,'y: ',y)
                middle[y] = [middle_x, y]
                
            start_search_at = middle_x
            if start_search_at < 0:
                start_search_at = 0
            if start_search_at >= width:
                start_search_at = width-1
            if middle_x <= 0 or middle_x >= width:
                break
            if found_startline == True:
                left_active, right_active = True, True
            
        lanelines = [left[left[:, 0] != -1], right[right[:, 0] != -1]]
        middle_line = middle[middle[:, 0] != -1]
        middle_line = self.filter_outliers(middle_line,1.0)
        
        return lanelines, middle_line
        
        
    def calculate_lateral_offset(self, middle_line,image):
        if middle_line is None:
            return np.nan
        #get distamce between bottom point of middle line and center of image
        height, width = image.shape[:2]
        x,y = zip(*middle_line)
        #get x of max y
        bottom_x = x[np.argmax(y)]
        #left of center is positive
        offset = -(bottom_x - (width//2))
        return offset
    
    def calculate_heading_angle(self, middle_line ,image,poly_coeffs):
        if middle_line is None:
            return np.nan
        image =image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        height, width = image.shape[:2]
        x,y = zip(*middle_line)
        derivative_coefficients = np.polyder(poly_coeffs)
        bottom_row = max(y)-20
        slope = np.polyval(derivative_coefficients, bottom_row)
        # Calculate the y-coordinate at the point of tangency
        x_tangency = np.polyval(poly_coeffs, bottom_row)
        x_intercept = x_tangency - slope * bottom_row
        y_values = np.linspace(0, height - 1, height)
        x_tangent = slope * y_values + x_intercept
        tangent = list(zip(x_tangent.astype(int), y_values.astype(int)))
        angle =  math.atan(slope) * 180 / math.pi
        #orientation to the right is negative, to the left is positive
        angle = angle
        
        if DEBUG:
            cv2.polylines(image, [np.array(middle_line)], False, (0, 0, 255), 2)
            cv2.polylines(image, [np.array(tangent)], False, (255, 0, 0), 2)
            #draw parker at bottom row
            cv2.circle(image,(int(x_tangency),bottom_row), 5, (0,255,0), -1)
            plt.imshow(image,cmap='gray')
            plt.title('Heading Angle')
            plt.show()
            
        return angle
    
    def fit_polynomial(self, points, degree=2, num_points=100,max_height=640):
        # Unpack the points into two lists
        x_values, y_values = zip(*points)
        # Fit a polynomial of the specified degree to the points
        poly_coeffs = np.polyfit(y_values, x_values, degree)
        # Generate a set of y values for the new points
        new_y_values = np.linspace(min(y_values), max_height, num_points)
        #to int
        new_y_values = new_y_values.astype(int)
        # Generate the corresponding x values using the polynomial
        new_x_values = np.polyval(poly_coeffs, new_y_values)
        new_x_values = new_x_values.astype(int)
        # Combine the new y and x values into a list of points
        new_points = list(zip(new_x_values, new_y_values))

        return new_points, poly_coeffs
        
    def centered_moving_average(self,points,windowSize=10):
        half_window = windowSize // 2
        smoothed_points = []
        # Pad the points at the beginning and end with 'None'
        padded_points = [None]*half_window + points + [None]*half_window
        for i in range(half_window, len(padded_points) - half_window):
            window_points = [p for p in padded_points[i-half_window : i+half_window+1] if p is not None]
            if window_points:
                avg_x = np.mean(window_points)
                smoothed_points.append((avg_x))
        
        return smoothed_points    
    

    #get endpoints of a curved line
    def get_endpoints(self, line,image):
        line = np.array(line,dtype=np.int32)
        #from greyscale to rgb
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rect = cv2.minAreaRect(line)
        box_w,box_h = rect[1]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,0,255),2)
        endpoints = []
        dists = []
        for point in box:
            p = (int(point[0]),int(point[1]))
            dist = cv2.pointPolygonTest(line, p, True)
            dists.append(abs(dist))
        # print('dists: ',dists)
        # print('box: ',box)
        dists = np.array(dists)[np.argsort(box[:,1])]
        box = box[np.argsort(box[:,1])]
        # print('dists: ',dists)
        # print('box: ',box)
        box_bottom = box[:2]
        box_top = box[2:]
        #get cornerpoint with smallest distance to line of botom corners
        endpoints.append(box_bottom[np.argmin(dists[:2])])
        #get cornerpoint with smallest distance to line of top corners
        endpoints.append(box_top[np.argmin(dists[2:])])
        #print('endpoints: ',endpoints)
        if DEBUG:
            #draw rectangle
            cv2.polylines(image, [line], False, (0, 255, 0), 2)
            #draw endpoints as crosses
            cv2.drawMarker(image, tuple(endpoints[0]), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(image, tuple(endpoints[1]), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            plt.imshow(image)
            plt.title('box')
            plt.show()

        return endpoints,box_w,box_h

    def calculate_centerpoint(self, radius,s,e):
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

    def correct_centerpoint(self, cp,s,e,left_curve):
        #fsolve only gives one solution. but there are two solutions for the centerpoint
        #so we have to check which one is the correct one
        m = (s[1]-e[1])/(s[0]-e[0]) if s[0] != e[0] else 1_000_000
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

    def calculate_radius(self, middle_line, image):
        if middle_line is None:
            return np.nan
        endpoints,bb_w,bb_h = self.get_endpoints(middle_line,image)
        self._endpoints = endpoints
        #width is alway the distance between points and bigger than height
        (bb_w,bb_h) = (bb_w,bb_h) if bb_w > bb_h else (bb_h,bb_w )
        # 2. Ansatz für die Berechnung der Krümmung nach https://www.mathopenref.com/arcradius.html
        s,e = endpoints
        #get distance between endpoints
        #d = np.sqrt((s[0]-e[0])**2 + (s[1]-e[1])**2)
        d = bb_w
        #get width of curve (distance between middle point and line between endpoints)
        #mp = point in middle of curve
        mp = middle_line[len(middle_line)//2]
        self._mp = mp
        #get line between endpoints
        #m = slope
        m = (s[1]-e[1])/(s[0]-e[0]) if s[0] != e[0] else 1_000_000
        #b = y-intercept
        b = s[1] - m*s[0]
        #calculate distance between middle point and line between endpoints
        h = (m*mp[0] - mp[1] + b)/np.sqrt(m**2 + 1)
        # print('h: ',h)
        # print('d: ',d)
        # print('bb_w: ',bb_w)
        # print('bb_h: ',bb_h)
        #h= bb_h
        #calculate radius
        radius = (h/2)+ d**2/(8*h)
        if radius == np.inf:
            radius = 10_000
        elif radius == -np.inf:
            radius = 10_000
        #if h is very small, set radius to 10_000
        if abs(bb_h) < 3.5 and abs(bb_w) > 110:
            radius = 10_000 if h > 0 else -10_000
        print('radius: ',radius)
        #set max radius to +- 10_000
        radius = radius if abs(radius) < 10_000 else 10_000 if radius > 0 else -10_000
        #set min radius to +- 80 which is smallest possible radius for track (is 0.5m)
        radius = radius if abs(radius) > 80 else 80 if radius > 0 else -80
        #calculate centerpoint
        cp = self.calculate_centerpoint(radius,s,e)
        self._cp = cp
        #if radius is plus or minus infinite, set it to plus or minus 1_000_000
        
        
        if DEBUG:
            self.show_estimated_circle(self._birdview_img, endpoints, middle_line, radius, cp, mp)
        
        #left curve if radius is negative, right curve if radius is positive
        if cp[0] < mp[0]:
            print('left curve')
            radius = - abs(radius)
        else:
            print('right curve')
            radius = abs(radius)
        
       
        return radius


    def show_estimated_circle(self,birdview_img,endpoints,middle_line,radius,cp,mp):
        if radius is np.nan:
            return
        #draw circle
        image = birdview_img.copy()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        cv2.polylines(image, [middle_line], False, (255,0,0), 1)
        #draw endpoints
        cv2.drawMarker(image, endpoints[0], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.drawMarker(image, endpoints[1], (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        #draw center point
        cv2.drawMarker(image, cp, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.drawMarker(image, mp, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        #draw circle around center point with radius = radius
        #set cp to middle of image if out of bounds
        # if cp[0] < 0 or cp[0] > image.shape[1] or cp[1] < 0 or cp[1] > image.shape[0]:
        #     cp = np.array([image.shape[1]//2,image.shape[0]//2])
        cv2.circle(image, cp, int(abs(radius)), (0, 255, 0), 1)
        plt.imshow(image,cmap='gray')
        plt.title('estimated radius')
        plt.show()
        
#--------------------------------------------------------------------------------------------------------------------------
#------------------------------------------middle line detection with YOLO-------------------------------------------------
    def get_middleLine_YOLO(self,frame):
        frame = cv2.resize(frame,(640,640))
        laneline_segments = self.get_laneline_segments(frame)
        filled_lanes_img = self.get_filled_lanes_img(frame, laneline_segments) 
        M,dsize,borderlines = self.get_birdview_matrix(filled_lanes_img,CT=0.4)
        birdview_img =  self.transform_img_to_birdview(filled_lanes_img,M,dsize)
        self.birdview_img = birdview_img
        lanelines = self.get_lanelines_centerline(birdview_img)
        middle_line = self.get_middle_line(lanelines, birdview_img) 
       
        return middle_line
    
    def get_laneline_segments(self,img):
        classes = self.model.names
        #print('classes: ',classes)
        segments = self.detectSegments(img)
        lanes = self.get_normal_lanes(segments)
        return lanes
    
    def detectSegments(self,frame):
        results = self.model(frame, show=False, stream=True,conf=0.5, device=0)
        # masks = results[0].masks  # Masks object

        # classes     = results[0].boxes.cls.cpu().numpy()
        # conf        = results[0].boxes.conf.cpu().numpy()
        # boxes_xyxy  = results[0].boxes.xyxy.cpu().numpy()

        # segments = []
        
        # if not masks == None:
        #     for i, segment in enumerate(masks.segments):
        #         segments.append([segment*[640,640],boxes_xyxy[i],classes[i], conf[i]])
        # return segments
        segments = []
        for i,result in enumerate(results):
            classes = result.boxes.cls.to('cpu').numpy()
            conf = result.boxes.conf.to('cpu').numpy()
            boxes_xyxy = result.boxes.xyxy.to('cpu').numpy()
            masks = result.masks
            if masks is not None:
                for j, segment in enumerate(masks.xyn):
                    segments.append([segment*[640,640],boxes_xyxy[j],classes[j], conf[j]])
            break
        return segments
            
    
    def get_normal_lanes(self,segments):
        segs =[]
        for segment in segments:
            if segment[2] == 2.0:
                segs.append(segment[0][:])
        return segs
            
     #get middle line between lanelines
    def get_middle_line(self,lanelines,img):
        image = np.zeros_like(img)
        left_line,right_line = lanelines
        
        if left_line is not None and right_line is not None:
            print('2 lanelines')
            # lines_pixels = []
            # for line in lanelines:
            #     line = np.array(line)
            #     line = line.reshape((-1,2))
            #     #sort line by x and y
            #     line = line[np.lexsort((line[:,1],line[:,0]))]
            #     line_pixels = self.get_contour_pixels(line,img)
            #     lines_pixels.append(line_pixels)
            #equalize pixel lines
            lines = self.equalize_pixel_lines(lanelines)
            #iterate over both lines and get the middle point
            middle_line = []
            line1,line2 = lines
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
                    
        elif left_line is not None or right_line is not None:
            print('only one line')
            #middle line is the not none line
            middle_line = left_line if left_line is not None else right_line

        else:
            print('no lines')
            middle_line = []            
        
        middle_line = self.moving_average(middle_line) 
        print('middle line',middle_line)
        middle_line.sort(key=lambda point: (point[0], point[1]))
        print('middle line',middle_line)
        middle_line = np.array(middle_line, dtype=np.int32)
        middle_line = self.remove_duplicates_from_line(middle_line)
        
        
        #add dimension on axis 1
        middle_line = np.reshape(middle_line,(-1,1,2))
        
        if DEBUG:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for line in lanelines:
                if line is not None:
                    line = np.array(line, dtype=np.int32)
                    cv2.polylines(img, [line], False, (0, 255, 0), 2)
            cv2.polylines(img, [middle_line], False, (0, 0, 255), 2)
            plt.imshow(img)
            plt.title('middle line')
            plt.show()
        return middle_line
    
    def get_line_endpoints(self,line):
        line =np.array(line,dtype=np.int32).squeeze()
        #sort line by x and y
        line = sorted(line,key=lambda x: (x[0],x[1]))
        #get endpoints
        s = line[0]
        e = line[-1]
        return s,e

    def filter_outliers(self,points,deviation_threshold=2):
        x,y = points.T
        dist = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        mean = np.mean(dist)
        #remove points that are more than 3*mean
        idx = np.where(dist < deviation_threshold*mean)
        return points[idx]


       
    def transform_image(self,img):
        img = cv2.GaussianBlur(img,(5,5),0)
        otsu = self.otsuThresholding(img,thresh=240)
        M =  self.get_birdview_matrix(otsu[1],CT=0.4,borderColor=[255])
        return M


    #draw filled in lanes
    def get_filled_lanes_img(self,img,lanes):
        h,w = img.shape[:2]
        empty = np.zeros((640,640,1),dtype=np.uint8)
        for lane in lanes:
            #lane as array with type int32
            lane = np.array(lane,dtype=np.int32).squeeze()
            #lane = np.array(np.int32(lane))
            #print('lane: ',lane)
            #draw all points in lane
            # for point in lane:
            #     print('point: ',point)
            #     cv2.circle(empty,(point[0],point[1]),1,(255,255,255),-1)
            cv2.drawContours(empty, [lane], -1, (255,0,0), 1)
            rr, cc = polygon(lane[:, 1], lane[:, 0], img.shape)
            #empty[rr, cc,1] = 255
            cv2.fillPoly(empty, [lane], (255))
        if DEBUG:
            plt.imshow(empty,cmap='gray')
            plt.title('Filled Lanes')
            plt.show()
            
        return empty

    


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
        for i in range(len(contours)):
            c = contours[i]
            #aproxximate contour
            epsilon = 0.01 #contour_length // desired_points
            aprox = cv2.approxPolyDP(c,epsilon,True)
            lanelines.append(aprox)
        if DEBUG:
            plt.imshow(skeleton,cmap='gray')
            plt.title('Skeleton')
            plt.show()
            image =np.zeros_like(img)
            for line in lanelines:
                cv2.drawContours(image, [line], -1, (255,0,0), 1)
            plt.imshow(image,cmap='gray')
            plt.title('Lanes')
            plt.show()
        
        return lanelines


    

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
    def equalize_pixel_lines(self,lines,nr_points=100):
        print('equalize_pixel_lines')
        #nr_points is minimum of length of line1 and line2 and the desired number of points
        lens = [len(x) for x in lines] if len(lines) > 1 else [len(lines[0])]
        print('lens: ',lens)
        nr_points = min(*lens,nr_points)
        equalized_lines = []
        for line in lines:
            steps = self.find_stepsizes(len(line),nr_points)
            line = self.remove_nth_elements(line,steps)
            equalized_lines.append(line)
            
        return equalized_lines

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



if __name__ == '__main__':
    import pandas as pd
    estimator = CurvatureEstimator(mode='otsu')
    dir = '../Aufnahmen/data/debug'
    #list files in directory
    files = os.listdir(dir)
    files = natsort.natsorted(files)
    print('len files: ',len(files))
    
    curvatures = []
    radii = []
    lateral_offsets = []
    heading_angles = []
    start_line = []
    found_start = False
    for i,file in enumerate(files):
        print('i: ',i)
        # if i < 120:
        #     continue
        img = cv2.imread(os.path.join(dir,file),0)
        estimator.process_frame(img,debug=False)
        
        starting_line = estimator.get_starting_line()
        start_line.append(starting_line)
        curvature = estimator.get_curvature(debug=False)
        curvatures.append(curvature)
        radius = estimator.get_radius()
        radii.append(radius)
        lateral_offset = estimator.get_lateral_offset()
        lateral_offsets.append(lateral_offset)
        heading_angle = estimator.get_heading_angle()
        heading_angles.append(heading_angle)
        
        #input('press enter to continue')
        # if i > 400:
        #     break
        
    nans_curvatures = np.where(np.isnan(curvatures))[0]
    print('nans_curvatures: ',nans_curvatures)
    nans_offsets = np.where(np.isnan(lateral_offsets))[0]
    nans_angles = np.where(np.isnan(heading_angles))[0]
    for i in nans_curvatures:
        next_not_nan = np.where(~np.isnan(curvatures[i:]))[0][0]
        curvatures[i] = (curvatures[i-1]+next_not_nan)/2
    for i in nans_offsets:
        next_not_nan = np.where(~np.isnan(lateral_offsets[i:]))[0][0]
        lateral_offsets[i] = (lateral_offsets[i-1]+next_not_nan)/2
    for i in nans_angles:
        next_not_nan = np.where(~np.isnan(heading_angles[i:]))[0][0]
        heading_angles[i] = (heading_angles[i-1]+next_not_nan)/2
    nans_curvatures = np.where(np.isnan(curvatures))[0]
    print('nans_curvatures: ',nans_curvatures)
    
    df = pd.DataFrame({'curvature':curvatures,'lateral_offset':lateral_offsets,'heading_angle':heading_angles})
    dates =pd.read_csv('../Aufnahmen/data/debug_timestamps.csv')
    df['datetime'] = dates['datetime']
    df = df[['datetime','curvature','lateral_offset','heading_angle']]
    df=df[:len(dates)]
    df.to_csv('../Aufnahmen/data/frame_measurements.csv',index=False)
    print('saved dataframe to csv')

# # # # df_curvatures = pd.DataFrame(curvatures,columns=['curvature'])
# # # # df_curvatures.to_csv('../Aufnahmen/data/curvatures.csv',index=False)
# # # # print('saved curvatures to csv')
# # #%%

# # plt.plot(curvatures)
# # plt.title('curvature')
# # current_cmap = matplotlib.cm.get_cmap()
# # current_cmap.set_bad(color='red')
# # plt.show()
# # smooth_curvatures = gaussian_filter(curvatures, 1)
# # plt.plot(smooth_curvatures)
# # plt.title('smooth curvature')
# # plt.show()

# # df_curvatures = pd.Series(curvatures)
# # m_a_curvatures = df_curvatures.rolling(5).mean().to_numpy()
# # plt.plot(m_a_curvatures)
# # plt.title('moving average curvature')
# # plt.show()

# # plt.plot(radii,color='blue')
# # plt.title('radius')
# # plt.show()

# # radii_smooth = gaussian_filter(radii, 1)
# # plt.plot(radii_smooth,color='red')
# # plt.title('smooth radius')
# # plt.show()

# #%%
# plt.plot(lateral_offsets)
# plt.title('lateral offset')
# plt.show()
# smooth_offsets = gaussian_filter(lateral_offsets, 2)
# plt.plot(smooth_offsets)
# plt.title('smoothed lateral offset')
# plt.show()



# plt.plot(heading_angles)
# plt.title('heading angle')
# plt.show()

# smooth_heading_angles = gaussian_filter(heading_angles, 1)
# plt.plot(smooth_heading_angles)
# plt.title('smooth heading angle')
# plt.show()

# #make dataframe from curvatures
# df = pd.DataFrame(lateral_offsets,columns=['lateral_offset'])
# df['curvature'] = curvatures
# df['heading_angle'] = heading_angles
# dates =pd.read_csv('../Aufnahmen/data/debug_timestamps.csv')
# df['datetime'] = dates['datetime']
# df['datetime'] = pd.to_datetime(df['datetime'])
# #to index
# df.set_index('datetime',inplace=True)
# print(df.head())
# #save dataframe to csv

# #%%
# %timeit get_lane_radius(img)



# %%
