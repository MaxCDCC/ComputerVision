from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def parse_coco_keypoints(keypoints: Sequence[float]) -> List[Tuple[float, float, int]]:
    # convert COCO keypoints into a list of tuples (x, y, visibility)
    return [
        (keypoints[i], keypoints[i + 1], int(keypoints[i + 2]))
        for i in range(0, len(keypoints), 3)
    ]


def build_frames(data: dict) -> Tuple[Dict[int, dict], Dict[int, List[dict]]]:
    # organize images and annotations for easier access
    image_info = {img['id']: img for img in data['images']}
    ann_by_image = defaultdict(list)
    for ann in data['annotations']:
        ann_by_image[ann['image_id']].append(ann)
    return image_info, ann_by_image


def camera_view_from_filename(filename: str) -> Optional[str]:
    # extract the view prefix from the filename
    return filename.split('_frame_')[0]


def group_annotations_by_frame(image_info: Dict[int, dict], ann_by_image: Dict[int, List[dict]]) -> Dict[int, Dict[int, List[Tuple[str, dict]]]]:
    # group annotations by frame and by player category
    grouped: Dict[int, Dict[int, List[Tuple[str, dict]]]] = defaultdict(lambda: defaultdict(list))
    for image_id, image in image_info.items():
        view = camera_view_from_filename(image['file_name'])
        if view is None:
            continue
        frame_index = int(image['file_name'].split('_frame_')[1].split('_')[0])
        for ann in ann_by_image.get(image_id, []):
            grouped[frame_index][ann['category_id']].append((view, ann))
    return grouped
