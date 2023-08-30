import face_recognition

from typing import Tuple
from numpy.typing import NDArray
from cv2.typing import MatLike


def crop_frame_around_face(
    frame:MatLike,
    crop_to_dimensions:Tuple[int,int],
    target_face_encodings:list[NDArray],
    face_match_tolerance:float = 0.6,
) -> Tuple[bool, NDArray, Tuple[int,int], Tuple[int,int]]:
    """
    \nArguments:
    \n- frame [MatLike] : the frame from which the target face is to be cropped from
    \n- crop_to_dimensions [Tuple[int,int]] : the dimensions to crop the frame to. ( row , column )
    \n- target_face_encodings [list[NDArray]] : a list of encodings which must be matched with a face within the frame, cropping will occur around said matching face.
    \n- face_match_tolerance [float] : the tolerance for a face to be considered a match. Lower is more strict.
    \n
    \nReturns: 
    \nTuple[ bool, NDArray, Tuple[int,int] ]
    \n- Tuple[0] bool : True if successfully cropped image around face, False if face could not be found in frame
    \n- Tuple[1] NDArray: The cropped frame, will be [] if could not successfully locate face.
    \n- Tuple[2] Tuple[int,int]: the position within the orignal frame that the frame was cropped to. Format: (left, top)
    \n- Tuple[3] Tuple[int,int]: the dimensions of the frame (will be same as passed argument). Format: (row, column)
    """


    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame,known_face_locations=face_locations)

    for i,encoding in enumerate(face_encodings):
        face_comparison_result = face_recognition.compare_faces(target_face_encodings,encoding,tolerance=face_match_tolerance)

        if all(face_comparison_result):
            #every known encoding matched the one located in the frame
            target_face_location_in_frame=face_locations[i]
            break
    else:
        #no face within frame was matched to known face encodings
        return (False,[],(-1,-1),(-1,-1))
    

    ## frame dimensions
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]


    ##face pos
    face_left = target_face_location_in_frame[3]
    face_right = target_face_location_in_frame[1]
    face_top = target_face_location_in_frame[0]
    face_bottom = target_face_location_in_frame[2]

    ## face dimensions
    face_width = face_right - face_left
    face_height = face_bottom - face_top

    ## target dimensions
    crop_to_width = crop_to_dimensions[0]
    crop_to_height = crop_to_dimensions[1]

    ## target position (default center around face)
    crop_to_left = int(face_left + (face_width - crop_to_width) / 2)
    crop_to_top = int(face_top + (face_height - crop_to_height) / 2)

    ## adjust crop position to account for bounds of frame

    if crop_to_left < 0:
        #hug left of frame
        crop_to_left = 0
    elif crop_to_left + crop_to_width > frame_width:
        #hug right of frame
        crop_to_left = frame_width - crop_to_width


    if crop_to_top < 0:
        #hug top of frame
        crop_to_top = 0
    elif crop_to_top + crop_to_height > frame_height:
        #hug bottom of frame
        crop_to_top = frame_height - crop_to_height

    
    cropped_frame = frame[ crop_to_top : crop_to_top + crop_to_height , crop_to_left: crop_to_left + crop_to_width]

    return (True,cropped_frame,(crop_to_left,crop_to_top),(crop_to_width,crop_to_height))



def crop_frame(
        frame:MatLike,
        crop_to_dimensions:Tuple[int,int],
        crop_position:Tuple[int,int]
    ) -> MatLike:
    """
    Arguments:
    \n- frame [MatLike] : the frame to crop from
    \n- crop_to_dimensions [Tuple[int,int]] : the dimensions to crop the frame to. ( row, column)
    \n- target_position [Tuple[int,int]] : the position to begin cropping from. ( left, top )
    \nReturns:
    \n- MatLike : the cropped frame.
    """
    left, top = crop_position
    width, height = crop_to_dimensions

    return frame[ top : top + height , left : left + width ]
