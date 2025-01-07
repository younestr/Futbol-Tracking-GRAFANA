import math

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox 
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]  # x2 - x1

def measure_distance(p1, p2): # distance between 2 points for ball assignment
    """
    Calculates the Euclidean distance between two points.

    Args:
        p1 (tuple): The first point as (x1, y1).
        p2 (tuple): The second point as (x2, y2).

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)