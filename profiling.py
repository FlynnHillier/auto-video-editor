"""Identification of individuals"""
from __future__ import annotations

import face_recognition
import json
import numpy
from pathlib import Path

from faces import get_all_faces
import cv2

from typing import Tuple
from numpy.typing import NDArray



## Utility

def _NDArray_list_to_list_list(NDArray_list:list[NDArray]) -> list[list]:
    """convert a list of NDArrays to a list of lists

    Args:
        NDArray_list (list[NDArray]): a list of numpy arrays to be converted to a list of lists

    Returns:
        list[list]: the converted list of lists
    """
    out = []
    for NDArray in NDArray_list:
        out.append(NDArray.tolist())

    return out

def _list_list_to_NDArray_list(list_list:list[list]) -> list[NDArray]:
        """convert a list of lists to a list of NDArrays

        Args:
            list_of_lists (list[list]): a list of lists to be converted to a list of numpy arrays

        Returns:
            list[NDArray]: the converted list of NDArrays
        """
        out = []
        for _list in list_list:
            out.append(numpy.array(_list))

        return out






class Profile:
    """Store data regarding an individual within forms of media, and identify at a later date."""
    id:str
    face_encodings:list[NDArray]
    add_face_encoding_default_tolerance:float


    def __init__(self,
        id:str,
        face_encodings:list[NDArray],
        add_face_encoding_default_tolerance: float = 0.0
    ) -> None:
        """
        Args:
            id (str): an id that can be used to identify who the data contained within the profile pertains to.
            face_encodings (list[NDArray]): an initial set of face encodings.
            add_face_encoding_default_tolerance (float, optional): The default tolerance used when attempting to add a new face encoding to the profile, if the new face does not match existing faces to a degree below this tolerance, they will be rejected unless this functionality is set to be ignored. Defaults to 0.0.
        """
        self.id = id
        self.face_encodings = face_encodings
        self.add_face_encoding_default_tolerance = add_face_encoding_default_tolerance

    

    def add_face_encoding(self,
        face_encoding:NDArray,
        tolerance:float | None = None,
        check_against_every_saved_encoding:bool = False,
        force_add=False,
    ) -> bool:
        """Add a face encoding to the existing pool. Encodings should be similar to those that are already saved.

        Args:
            face_encoding (NDArray): The face encoding to be added.
            tolerance (float | None): how similar the face must be to existing faces to be added (0 -> 1) . If None, use default tolerance defined in constructor . Defaults to None.
            check_against_every_saved_encoding (bool): If True every saved encoding's distance from the passed encoding must be within the provided tolerance level. If False, check against average. Defaults to False.
            force_add (bool, optional): if set to True will add face regardless of if it matches already saved faces . Defaults to False.

        Returns:
            bool: True if face was added, False if not (face encoding did not match saved encodings sufficiently).
        """

        if tolerance == None:
            #use default encoding tolerance
            tolerance = self.add_face_encoding_default_tolerance

        if not force_add:
            #check face encoding is sufficiently similar to existing face encodings
            avg_distance, individual_distances = self.face_encoding_distance_against_saved_encodings(face_encoding)

            if not check_against_every_saved_encoding:
                #check average
                if not (avg_distance > tolerance):
                    return False
            else:
                #check every
                for distance in individual_distances:
                    if distance > tolerance:
                        return False
                    
        self.face_encodings.append(face_encoding)
        return True
    


    def face_encoding_distance_against_saved_encodings(self,
        face_encoding_to_check:NDArray,
    ) -> Tuple[float,list[float]]:
        """Generate data pertaining to how similar the given face encoding is to saved face encodings.
        
        if no encodings are present, the score will default to 1.

        Args:
            face_encoding_to_check (NDArray): the face encoding to check againast saved face encodings. 

        Returns:
            Tuple[float,list[float]]: First index contains the average encodings distance. The second index is an array of each individual encoding distance.
        """

        if len(self.face_encodings) == 0:
            #return max score if no saved face encodings exist
            return (1,[])

        face_distances = face_recognition.face_distance(self.face_encodings,face_encoding_to_check)
        average_distance = sum(face_distances) / len(face_distances)

        return (average_distance,face_distances)
    


    def clear_face_encodings(self) -> None:
        """Clear saved face encodings"""
        self.face_encodings = []



    def save_to_file(self,
        directory:str,
        filename:str | None = None,
    ) -> str:
        """Write a copy of the profile to disk at the specified location.

        Args:
            directory (str): the target directory.
            filename (str | None, optional): The filename, if None defaults to the profile id. Defaults to None.

        Returns:
            str: the filepath of the saved profile
        """
        
        #create directory if not exists
        Path(directory).mkdir(exist_ok=True)
        
        if filename == None:
            #default to profile id name
            filename = self.id
        
        if not filename.endswith(".json"):
            #append correct file extension
            filename += ".json"

        target_filepath = str(Path(directory).joinpath(filename))

        _dict = {
            "id":self.id,
            "face_encoding_tolerance":self.add_face_encoding_default_tolerance,
            "face_encodings":_NDArray_list_to_list_list(self.face_encodings),
        }

        with open(target_filepath,"w+") as f:
            f.write(json.dumps(_dict,indent=2))
            f.close()

        return target_filepath



    def load_from_file(
        filepath:str,
    ) -> Profile:
        """load a saved profile from disk

        Args:
            filepath (str): the filepath of the target json containing the details of the profile to be loaded.
            raise_exception (bool, optional): if False, the function will return False prematurely instead of raising exception. Defaults to True.

        Raises:
            FileNotFoundError: If the target JSON cannot be found.
            Exception: invalid file extension (expects '.json')
            KeyError: an expected key within the JSON file is not present.
            TypeError: a key was provided within the JSON file, but was of the incorrect type.

        Returns:
            Profile: the loaded Profile instance.
        """

        if not Path(filepath).exists():
            #filepath does not exist
            raise FileNotFoundError(str(Path(filepath)))


        if Path(filepath).suffix != ".json":
            #file does not possess correct file extension
            raise Exception(f"invalid file extension {Path(filepath).suffix} , expected '.json'")
  

        ## read json

        with open(filepath,"r") as f:
            data = json.load(f)
            f.close()


        ## check json has all required keys

        expected_keys = [
            "id",
            "face_encodings",
            "add_face_encoding_default_tolerance"
        ]

        for expected_key in expected_keys:
            if expected_key not in data.keys():
                raise KeyError(f"json located at '{filepath}' is missing expected key: '{expected_key}'")
                

        ## check json values are of the expected type

        expected_types = {
            "id":str,
            "face_encodings":list,
            "add_face_encoding_default_tolerance":float,
        }

        for expected_type_key in expected_types.keys():
            expected_type = expected_types[expected_type_key]
            actual_type = type(data[expected_type_key])
            
            if actual_type != expected_type:
                raise TypeError(f"key '{expected_type_key}' expected to be of type '{expected_type}' , recieved type '{actual_type}'")
                
        
        ##assign the values

        return Profile(
            id=data["id"],
            add_face_encoding_default_tolerance=data["add_face_encoding_default_tolerance"],
            face_encodings=_list_list_to_NDArray_list(data["face_encodings"])
        )
    


def build_profiles_from_video(
    video_fp:str,
    profile_directory:str | None = None,
    save_profiles:bool = False,
    time_step: int = 1,
    prompt_on_existing_profile: bool = False,
    overwrite_profile_by_default: bool = False,
) -> list[Profile]:
    """build multiple profiles easily from an inputted video

    Args:
        video_fp (str): the video filepath which contains people from which the profiles are to be generated.
        profile_directory (str | None, optional): If provided, provides the ability to append details to existing profiles with the same id. Defaults to None.
        save_profiles (bool, optional): save the generated profiles to a directory. Will use profile_directory if provided, else will prompt user. Defaults to False.
        time_step (int, optional): the amount of time between face-scans when reading the video. Defaults to 1.
        prompt_on_existing_profile (bool, optional): if reading profiles from an existing profile directory, ask the user wether to overwrite/append to a profile when it already exists. Defaults to False.
        overwrite_profile_by_default (bool, optional): if prompts are disabled, describes the default behaviour when existing profile is found. Defaults to False.

    Raises:
        FileNotFoundError: The specified proifile_directory could not be found.

    Returns:
        list[Profile]: the list of generated profiles
    """    

    if profile_directory != None:
        if not Path(profile_directory).exists():
            #existing profile directory does not exist
            raise FileNotFoundError(str(Path(profile_directory)))


    print("fetching faces")
    face_encodings, face_images = get_all_faces(video_filepath=video_fp,time_step=time_step)
    print("faces fetched.")


    profiles : list[Profile] = []

    for i,face_encoding in enumerate(face_encodings):
        face_image = face_images[i]

        ##show user image and ask for id
        cv2.imshow(f"{i}",face_image)
        cv2.waitKey(1)
        id = input("who is this? (leave blank to skip)\n> ")
        cv2.destroyWindow(f"{i}")
        
        if id == "":
            #user skipped face
            continue
        
        instantiated = False
        
        if profile_directory != None:
            #search for existing profile in specified profile directory.

            target_profile_path = Path(profile_directory).joinpath(id + ".json")

            if target_profile_path.exists():
                #profile under this name exists.

                if not prompt_on_existing_profile:
                    #do not prompt

                    if not overwrite_profile_by_default:
                        #append to profile by default
                        profile = Profile.load_from_file(str(target_profile_path))
                        profile.add_face_encoding(face_encoding)
                        instantiated = True
                    else:
                        pass #default behaviour

                else:
                    ## prompt user to choose how to approach pre-existing profile 
                    valid_inputs = ["1","2"]
                    while True:
                        u_input = input(f"""A profile at the location '{str(target_profile_path)}' already exists.\n1. append to\n2. ignore (this means existing profile will be overwritten if saving)\n> """)

                        if u_input in valid_inputs:
                            #valid input so break out of input loop
                            break
                        
                        #clear the previous itteration of prints / inputs. (on invalid input)
                        for _ in range(4):
                            print("\033[1A",end="\x1b[2K")
                    
                    if u_input == "1":
                        #append to
                        profile = Profile.load_from_file(str(target_profile_path))
                        profile.add_face_encoding(face_encoding)
                        instantiated = True
                    elif u_input == "2":
                        #overwrite
                        pass #default behaviour

        if not instantiated:
            #instantiate profile
            profile = Profile(
                id=id,
                face_encodings=[face_encoding]
            )

        profiles.append(profile)
        

    if save_profiles == True:
        #save generated profiles

        #default save to dir to profile directory ( if provided )
        save_to_path = profile_directory

        if not save_to_path:
            #ask user where to save profiles to if profile directory is not specified
            while True:
                u_path = input(
                    "save profiles to:\n> ")
                
                if Path(u_path).is_dir():
                    #valid input
                    break
                
                #clear previous print itteration (on invalid input)
                for _ in range(4):
                    print("\033[1A",end="\x1b[2K")

            #create directory if does not exist
            Path(u_path).mkdir(parents=True,exist_ok=True)

            save_to_path = str(Path(u_path))

        for profile in profiles:
            profile.save_to_file(directory=save_to_path)
    
    return profiles