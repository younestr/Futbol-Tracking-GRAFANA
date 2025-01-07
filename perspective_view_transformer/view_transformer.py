import numpy as np 
import cv2

class ViewTransformer():
    """
    A class for transforming positions from a trapezoid-shaped perspective view (caused by a tilted camera)
    into real-world measurements as if viewed from a perpendicular (top-down) perspective.

    The transformation maps pixel coordinates from the distorted camera view to real-world coordinates
    on a rectangular field using a perspective transformation matrix.

    Attributes:
        court_width (float): The width of the court in real-world units (e.g., meters).
        court_length (float): The length of the court in real-world units (e.g., meters).
        pixel_vertices (np.ndarray): The pixel coordinates of the four corners of the court in the image.
        target_vertices (np.ndarray): The real-world coordinates corresponding to the four corners of the court.
        perspective_transformer (np.ndarray): The matrix used for perspective transformation.
    """

    def __init__(self):
        # dimensions of the court in real-world units (meters) - FIFA reg- 
        court_width = 68 # 68 meters
        court_length = 23.32 # 105/2 = 52.5 :: 9 rectangles each half court :: each rectangle 5.83 meter long :: scourt separated into 3 ( it's 105 meters all long )
        
        # pixel coordinates of the court corners in the distorted (camera) view.
        # these define a trapezoid shape due to the tilted camera perspective.
        self.pixel_vertices = np.array([[110, 1035],  # bottom left
                               [265, 275], # top left
                               [910, 260], # top right
                               [1640, 915]]) # bottom right
        
        # real-world coordinates of the court corners in a top-down, rectangular view.
        # these define the real-world rectangle corresponding to the trapezoid.
        self.target_vertices = np.array([
            [0,court_width], # bottom left in the real-world
            [0, 0], # top left in the real-world
            [court_length, 0], # top right in the real-world
            [court_length, court_width] # bottom right in the real-world
        ])

        # ensure vertices as type float32 for implemetation of  openCV function
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        #  compute the perspective transformation matrix.
        # maps pixel coordinates from the distorted (trapezoid) view to real-world (rectangular) coordinates.
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self,point):
        """
        Transforms a single point from the distorted camera view to the real-world view.

        Args:
            point (np.ndarray): The pixel coordinates of the point in the camera view.

        Returns:
            np.ndarray or None: The transformed coordinates in the real-world view, or None if the point
                                lies outside the trapezoid (court area).
        """
        # ensure the point is represented as an integer tuple for checking inside the polygon.
        p = (int(point[0]),int(point[1]))

        # check if the point is inside the trapezoid defined by the pixel vertices.
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None

        # reshape the point for transformation.
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)

        # apply the perspective transformation.
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.persepctive_trasnformer)
        
        # reshaped to 2D point
        return tranform_point.reshape(-1,2)

    def add_transformed_position_to_tracks(self,tracks):
        """
        Adds the transformed (real-world) position to each tracked object in the tracks.

        Args:
            tracks (dict): The dictionary containing tracking data for objects.
                           Expected structure:
                           {
                               "players": [frame_0, frame_1, ...],
                               "ball": [frame_0, frame_1, ...]
                           }
                           Each frame contains tracked objects with 'position_adjusted' key.

        Modifies:
            The `tracks` dictionary is updated with a new key `'position_transformed'` for each object,
            containing the real-world coordinates of the object.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed