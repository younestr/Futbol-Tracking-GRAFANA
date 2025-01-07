from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys 
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox, get_foot_position
import cv2
import numpy as np
import pandas as pd 

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

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # semi-transparent rectangle
        overlay = frame.copy() # helps with transparency by drawing on the overlay
        cv2.rectangle(overlay,
                      (1350,850), # positions
                      (1900,970),
                      (255,255,255), # white
                      -1) # filled
        alpha = 0.4 # transparency factor
        cv2.addWeighted(overlay,
                        alpha,
                        frame,
                        1-alpha,
                        0,
                        frame)
        
        # calculating percetange 
        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # % of time each team had the ball
        team1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0] # numpy list for team 1
        team2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0] # numpy list for team 1

        team_1 = team1_num_frames / (team1_num_frames + team2_num_frames)
        team_2 = team2_num_frames / (team1_num_frames + team2_num_frames)

        cv2.putText(frame, f"Team 1 Possession :{team_1*100:.2f}%", (1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Possession :{team_2*100:.2f}%", (1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame 
    
    def draw_annotations(self,video_frames,tracks, team_ball_control):
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

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame,
                                               player["bbox"],
                                               (0,0,255)) # red

            # draw referees
            for track_id , referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee["bbox"], (0,255,255)) # yellow

            # draw ball
            for track_id , ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball["bbox"], (0,255,0)) # green
            
            # draw block for team possession
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # draw id under ellipse of player
            output_video_frames.append(frame)
        
        return output_video_frames

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions to fill gaps in the tracking data.

        Args:
            ball_positions (list): List of ball position dictionaries. Each dictionary
                                   corresponds to a frame and contains 'bbox' as the bounding box
                                   of the ball or an empty list if the ball is missing.

        Returns:
            list: List of interpolated ball positions with gaps filled.
        """
        # conversion to df
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions] # getting the 1st track id ( if not get empty dict) and bbox ( if not get empty list)
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        # interpolate missing values ( missing ball tracks )
        df_ball_positions = df_ball_positions.interpolate() # filling gaps lineraly
        df_ball_positions = df_ball_positions.bfill() # for first frames  (leading NaN values)

        # deconversion 
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position      