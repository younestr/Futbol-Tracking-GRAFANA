from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys 
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox
import cv2
import numpy as np

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self,frames):
        batch_size = 20 # to minimize the memory usage by limiting (20 frames by 20 frames)
        detections =[]
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1) # minimum conf is 10% // model.track for trackID
            detections += detections_batch 
            #break  # for testing only on the first batch to avoid the detection on all frames
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path): # checking track results from existing pickle file
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[], 
            "referees":[],
            "ball":[]
            # a goalkeeper class isn't here because we won't do any special stats or analysis on the GK ; will be treated as a player
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # {0:person, 1:goalkeeper}
            cls_names_inv = {v:k for k,v in cls_names.items()} # {person:0, goalkeeper:1}(this is more convenient)
 
            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalKeeper to player class // object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks: # looping over each track
                # 0 bbox , 1 mask , 2 conf , 3 cls id , 4 track id
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # no need for tracking on the ball , because there's 1 ball (unlike players and referees there's many of them)
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox} # tracks of players
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox} # tracks of refereees
            
            for frame_detection in detection_supervision: # for the ball ( detection without tracks )
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:   
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)   # dumping tracks in pickled file

        return tracks # dictionary of list of dictionaries

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # bottom of bounding box

        x_center , _ = get_center_of_bbox(bbox) # we only need x center
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)), # minor axis , major axis of ellipse
            angle = 0.0,
            startAngle=-45, # doesn't draw the whole ellipse looks more neat
            endAngle=235,
            color= color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # rectangle for track id under ellipse
        rectangle_width = 40
        rectangle_height= 20
        x1_rect = x_center - rectangle_width//2 # top left x-coordinate
        x2_rect = x_center + rectangle_width//2 # bottom-right x-coordinate
        y1_rect = y2 + 5  # Slight offset below the ellipse
        y2_rect = y1_rect + rectangle_height  # Bottom-right y-coordinate

        if track_id is not None:
            cv2.rectangle(frame, 
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            # adjustments
            font_scale = 0.4  # smaller font size
            font_thickness = 2  # thinner font
            text_size = cv2.getTextSize(str(track_id), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x_center - (text_size[0] // 2)  # Center text horizontally
            text_y = y1_rect + rectangle_height // 2 + text_size[1] // 2  # Center text vertically

            cv2.putText(
                        frame,
                        f"{track_id}",
                        (int(text_x),int(text_y)), # text position
                        cv2.FONT_HERSHEY_SIMPLEX, # font
                        font_scale, 
                        (0,0,0), # black colored
                        font_thickness 
                        )
        return frame
        
    def draw_triangle(self, frame, bbox, color):
        """
        Draws an inverted triangle on the frame based on the bounding box.
        
        Args:
            frame (ndarray): The image frame.
            bbox (list): Bounding box [x1, y1, x2, y2].
            color (tuple): Color of the triangle (B, G, R).
            size (int): Size of the triangle. Default is 20.
        Returns:
            ndarray: The frame with the triangle drawn.
        """
        # center coordinates of the ball
        y_center = int(bbox[1]) # y1
        x_center , _ = get_center_of_bbox(bbox)
        
        # vertices
        triangle_points = np.array([
            [x_center,y_center], # base lower head of inverted triang
            [x_center-10,y_center-20], # top left
            [x_center+10,y_center-20] # top right
        ])

        cv2.drawContours(frame,[triangle_points],0,color, cv2.FILLED) # filled inverted ball
        cv2.drawContours(frame,[triangle_points],0,(0,0,0), 2) # border

        return frame


    def draw_annotations(self,video_frames,tracks):
        output_video_frames=[]
        
        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy() # to not scratch & pollute on frames that are being fed

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # draw players
            for track_id , player in player_dict.items():
                color = player.get("team_color",(0,0,255)) # get team color , if not take red 
                frame = self.draw_ellipse(frame,player["bbox"], color, track_id) # red
            
            # draw referees
            for track_id , referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee["bbox"], (0,255,255)) # yellow

            # draw ball
            for track_id , ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball["bbox"], (0,255,0)) # green

            # draw id under ellipse of player
            output_video_frames.append(frame)
        
        return output_video_frames
