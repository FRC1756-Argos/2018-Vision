import libjevois as jevois
import cv2
import numpy as np

# @description Program to detect Vision Targets for FRC 2018
# 

class PythonSandbox:
    
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        
    # ###################################################################################################
    ## Function to group detected targets
    def find_if_close(self, target1, target2):
        row1,row2 = target1.shape[0], target2.shape[0]
        for i in range(row1):
            for j in range(row2):
                dist = np.linalg.norm(target1[i]-target2[j])
                #TODO: need to determine this threshold more accurately
                if abs(dist) < 150 :
                    return True
                elif i==row1-1 and j==row2-1:
                    return False

    # ###################################################################################################
    ## Function to group detected targets
    def check_if_target(self, target):
        # Based on the area of vision target from the maximum possible distance of the camera on the field
        AREA_THRESHOLD = 100 
        MIN_TARGET_ASPECT_RATIO_THRESHOLD = 5
        global area 
        area = cv2.contourArea(target)
         #skip targets if smaller than expected
        if area < AREA_THRESHOLD:
            return False
        #skip targets with unwanted aspect ratio
        x,y,w,h = cv2.boundingRect(target)
        aspect_ratio = int(h)/int(w)
        #if aspect_ratio < MIN_TARGET_ASPECT_RATIO_THRESHOLD:
            #return False
        return True
                
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
    
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        
        self.timer.start()
        
        # Convert to HSV color space
        HSV = cv2.cvtColor(inimg, cv2.COLOR_BGR2HSV)
        
        HUE_MIN = 25  #Reflective Green: 25 - ~80
        HUE_MAX = 90
    
        SAT_MIN = 100
        SAT_MAX = 255
    
        VAL_MIN = 150
        VAL_MAX = 255
           
        lowerBound = np.array([HUE_MIN,SAT_MIN,VAL_MIN])
        upperBound = np.array([HUE_MAX,SAT_MAX,VAL_MAX])
        
        # Filter color using HSV min and max values
        mask = cv2.inRange(HSV, lowerBound, upperBound)
        
        # filter noise based on area
        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1,1)))
        
        # close any holes of 3-pixel wide lo less
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, np.ones((3,3)))
        
        # quantify targets
        # NOTE: CHAIN_APPROX_SIMPLE will get corner points of a contour (target),
        #       use CHAIN_APPROX_NONE to get all points on a contour, not computationally efficient
        _, targets, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # show only the targets
        outimg = cv2.bitwise_and(inimg, inimg, mask= maskClose)
          
        
        for target in targets:                      
            is_good = self.check_if_target(target)
            if is_good == False:
                continue
            #get a visualization and info of the detected target
            M = cv2.moments(target)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.putText(outimg, str(area), (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            #x,y,w,h = cv2.boundingRect(target)
            #outimg = cv2.rectangle(outimg, (x,y), (x+w,y+h), (255,0,0), 2)
            
            bb = cv2.minAreaRect(target)
            bbox = cv2.boxPoints(bb)
            bbox = np.int0(bbox)
            outimg = cv2.drawContours(outimg, [bbox], 0, (255,0,0), 2)
            focal_length=(18.0*int(w))/2.0
            cv2.putText(outimg, str(focal_length), (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
        Total_Targets = len(targets)
        status = np.zeros((Total_Targets,1))

        # combine the targets to classify them left and right later
        for i,target1 in enumerate(targets):
            if self.check_if_target(target1) == False:
                continue   
            x = i    
            if i != Total_Targets-1:
                for j,target2 in enumerate(targets[i+1:]):
                    if self.check_if_target(target2) == False:
                        continue
                    x = x+1
                    dist = self.find_if_close(target1,target2)
                    if dist == True:
                        val = min(status[i],status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x]==status[i]:
                            status[x] = i+1

        # group targets and get visual feedback before getting coordinates info
        grouped_targets = []
        maximum = np.int0(status.max())+1
        for i in range(maximum):
            pos = np.where(status==i)[0]
            if pos.size != 0:
                targ = np.vstack(targets[i] for i in pos)
                rect = cv2.minAreaRect(targ)
                grouped_targets.append(rect)
        
        for group in grouped_targets:
            bgroup = cv2.boxPoints(group)
            bgroup = np.int0(bgroup)
            outimg = cv2.drawContours(outimg, [bgroup], -1, (0,0,255), 2)     
              
                
        # Write processing info on video
        fps = self.timer.stop()
        cv2.putText(outimg, "Argos Vision Module: at", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        cv2.putText(outimg, fps, (3, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        cv2.putText(outimg, "calculated focal length (inches):", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        

        height, width, channels = outimg.shape # if outimg is grayscale, change to: height, width = outimg.shape
        

        # Convert our BGR output image to video output format and send to host over USB. If your output image is not
        outframe.sendCvBGR(outimg)
