import cv2
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistanceEstimator():
    """
    A class for estimating and visualizing speed and distance covered by tracked objects 
    (e.g., players) in a sequence of frames. The calculations are based on transformed positions 
    (real-world coordinates) provided in the tracking data.

    Attributes:
        frame_window (int): Number of frames to group for speed and distance calculation.
        frame_rate (int): Frame rate of the video, used for time calculations.
    """
    def __init__(self):
        self.frame_window = 5 # number of frames used to calculate speed and distance.
        self.frame_rate = 24 # frame rate of the video in frames per second (fps).
    
    def add_speed_and_distance_to_tracks(self,tracks):
        """
        Adds speed (in km/h) and cumulative distance (in meters) to the tracking data.

        Args:
            tracks (dict): Tracking data for objects. Expected structure:
                           {
                               "players": [frame_0, frame_1, ...],
                               "ball": [frame_0, frame_1, ...],
                               "referees": [frame_0, frame_1, ...]
                           }
                           Each frame contains tracked objects with keys like 'position_transformed'.
        """
        total_distance= {} # dict to store cumulative distance for each object and track ID.

        # iterate over all tracked objects (e.g., players, ball, referees).
        for object, object_tracks in tracks.items():
            # skip "ball" and "referees" as speed and distance are not calculated for them.
            if object == "ball" or object == "referees":
                continue 

            number_of_frames = len(object_tracks) # total number of frames for this object.

            # process frames in batches of size `frame_window`.
            for frame_num in range(0,number_of_frames, self.frame_window):
                # determine the last frame in the current batch.
                last_frame = min(frame_num+self.frame_window,number_of_frames-1 )

                # iterate over all tracked IDs in the current frame.
                for track_id,_ in object_tracks[frame_num].items():
                     # skip if the object is not present in the last frame of the batch.
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # get the start & end positions for the object in the batch.        
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # skip if positions are invalid (e.g., outside the court area).
                    if start_position is None or end_position is None:
                        continue
                    
                    # calculate distance covered in the batch (in meters).
                    distance_covered = measure_distance(start_position,end_position)
                    # calculate time elapsed in the batch (in seconds).
                    time_elapsed = (last_frame-frame_num)/self.frame_rate
                    # Calculate speed in meters per second and convert to km/h.
                    speed_meteres_per_second = distance_covered/time_elapsed
                    speed_km_per_hour = speed_meteres_per_second*3.6

                    # initialize the total distance dictionary for this object and track ID.

                    if object not in total_distance:
                        total_distance[object]= {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    # update cumulative distance.
                    total_distance[object][track_id] += distance_covered

                    # add speed & distance to each frame in the current batch.
                    for frame_num_batch in range(frame_num,last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self,frames,tracks):
        """
        Overlays speed (in km/h) and cumulative distance (in meters) on video frames.

        Args:
            frames (list): List of video frames.
            tracks (dict): Tracking data with speed and distance annotations.

        Returns:
            list: Frames with annotated speed and distance.
        """
        output_frames = [] # list to store annotated frames

        # loop over all frames.
        for frame_num, frame in enumerate(frames):
            # loop over all tracked objects
            for object, object_tracks in tracks.items():
                # skipping since we're uninterested rn by ball & referee stats
                if object == "ball" or object == "referees":
                    continue 

                # iterating over objects in current frames
                for _, track_info in object_tracks[frame_num].items():
                   # skipping if there's missing speed & distance data
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       # getting bbox & footposition for text placement
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames