from utils import read_video, save_video
from tracking import Tracker
import cv2 
from team_assigner import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
import numpy as np
from camera_estimator import CameraMovementEstimator
from perspective_view_transformer import ViewTransformer
from speed_distance_estimator import SpeedAndDistanceEstimator

def main():
    # 1- read the input video
    input_path = 'input_vids/08fd33_4.mp4'
    video_frames = read_video(input_path)
    print(f"Loaded {len(video_frames)} frames from {input_path}")

    # 2- initialize Tracker class
    tracker = Tracker('models/best.pt')
    print("Tracker initialized with YOLO model.")

    # 3- generate or load object tracks
    stub_path = 'stubs/track_stubs.pkl'
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True, 
                                       stub_path=stub_path)
    print("Object tracking completed.")

    # 4- camera movement estimator

    tracker.add_position_to_tracks(tracks) # get object positions 
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    print("Adjusted Camera positions successfully! ")

    # 5- Perspective View transformer 
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    print("Perspective view transformer successfully added! ")

    # 6- interpolating missing ball tracks
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print("Missing ball tracks interpolated successfully !")

    # 7- speed & distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    print("Adding Speed & Distance of players on track info successfully !")

    # 8- assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num , player_track in enumerate(tracks['players']):
        for player_id , track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track["bbox"],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    print("Teams assigned successfully!")
    
    # 9- assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True # new parameter
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team']) # getting the team of who's player has the ball
        #else:
            #team_ball_control.append(team_ball_control[-1])  # Retain last possession
    team_ball_control = np.array(team_ball_control) # conversion to np array
    print("Ball in possession assigned succesfully!")

    # 10- draw annotations on video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    print("Annotations drawn on video frames.")

    # 11- drawing camera movement estimation
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    print("Camera movement annotations drawn successfully !")

    # 12- drawing speed & distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    print("Speed & Distance annotations drawn successfully!")

    # 12- save the annotated video
    output_path = 'output_videos/annoTracks_withColor&BallInterpo&plyrAcquiAssign&Poss&CameraMvmt&ViewTransformer&SpeedDistance.avi'
    save_video(output_video_frames, output_path)
    print(f"Annotated video saved to {output_path}")

if __name__ == '__main__':
    main()