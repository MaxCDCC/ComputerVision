from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def parse_coco_keypoints(keypoints: Sequence[float]) -> List[Tuple[float, float, int]]:
    # convert COCO keypoints into a list of tuples (x, y, visibility)
    # [x1, y1, v1, x2, y2, v2, ...] -> [(x1, y1, v1), (x2, y2, v2), ...]
    return [
        (keypoints[i], keypoints[i + 1], int(keypoints[i + 2]))
        for i in range(0, len(keypoints), 3)
    ]


def build_frames(data: dict) -> Tuple[Dict[int, dict], Dict[int, List[dict]]]:
    # organize images and annotations for easier access

    # Create a dictionary to quickly access image information by image id
    image_info = {img['id']: img for img in data['images']}

    # Store all annotations belonging to the same image
    ann_by_image = defaultdict(list)   
    for ann in data['annotations']:
        ann_by_image[ann['image_id']].append(ann)
    return image_info, ann_by_image


def camera_view_from_filename(filename: str) -> Optional[str]:
    # extract the view prefix from the filename
    # out1_frame_10.jpg -> out1
    return filename.split('_frame_')[0]


def get_image_sizes_by_view(data: dict) -> Dict[str, Tuple[int, int]]:
    # Read the (width, height) of one image per camera view from the COCO file.
    # This is needed by cv2.calibrateCamera, which requires the image size to
    # build a sensible initial guess for the principal point (image center).
    sizes: Dict[str, Tuple[int, int]] = {}

    for img in data['images']:
        view = camera_view_from_filename(img['file_name'])
        if view is None or view in sizes:
            continue

        width = img.get('width')
        height = img.get('height')
        if width and height:
            sizes[view] = (int(width), int(height))

    return sizes


def group_annotations_by_frame(image_info: Dict[int, dict], ann_by_image: Dict[int, List[dict]]) -> Dict[int, Dict[int, List[Tuple[str, dict]]]]:
    # group annotations by frame and by player category
    # grouped[frame][player] = [(view, annotation), ...]
    grouped: Dict[int, Dict[int, List[Tuple[str, dict]]]] = defaultdict(lambda: defaultdict(list))

    # Loop through all images in the dataset
    for image_id, image in image_info.items():

        # Get the camera view (out1, out2, ...)
        view = camera_view_from_filename(image['file_name'])

        if view is None:
            continue

        # Extract the frame number from the filename
        # out1_frame_15.jpg -> 15
        frame_index = int(image['file_name'].split('_frame_')[1].split('_')[0])

        # Add every annotation found in this image
        for ann in ann_by_image.get(image_id, []):

            # category_id identifies a player/team class
            grouped[frame_index][ann['category_id']].append((view, ann))

    return grouped
