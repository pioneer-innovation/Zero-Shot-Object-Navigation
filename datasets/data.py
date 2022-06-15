from .scene_util import get_scenes

from .constants import (
    KITCHEN_OBJECT_CLASS_LIST,
    LIVING_ROOM_OBJECT_CLASS_LIST,
    BEDROOM_OBJECT_CLASS_LIST,
    BATHROOM_OBJECT_CLASS_LIST,
    FULL_OBJECT_CLASS_LIST,
)
from .constants import (
    SEEN_FULL_OBJECT_18CLASS_LIST,
    SEEN_BATHROOM_OBJECT_18CLASS_LIST,
    SEEN_BEDROOM_OBJECT_18CLASS_LIST,
    SEEN_KITCHEN_OBJECT_18CLASS_LIST,
    SEEN_LIVING_ROOM_OBJECT_18CLASS_LIST
)
from .constants import (
    SEEN_FULL_OBJECT_14CLASS_LIST,
    SEEN_BATHROOM_OBJECT_14CLASS_LIST,
    SEEN_BEDROOM_OBJECT_14CLASS_LIST,
    SEEN_KITCHEN_OBJECT_14CLASS_LIST,
    SEEN_LIVING_ROOM_OBJECT_14CLASS_LIST
)
from .constants import (
    UNSEEN_FULL_OBJECT_4CLASS_LIST,
    UNSEEN_BATHROOM_OBJECT_4CLASS_LIST,
    UNSEEN_BEDROOM_OBJECT_4CLASS_LIST,
    UNSEEN_KITCHEN_OBJECT_4CLASS_LIST,
    UNSEEN_LIVING_ROOM_OBJECT_4CLASS_LIST
)
from .constants import (
    UNSEEN_FULL_OBJECT_8CLASS_LIST,
    UNSEEN_BATHROOM_OBJECT_8CLASS_LIST,
    UNSEEN_BEDROOM_OBJECT_8CLASS_LIST,
    UNSEEN_KITCHEN_OBJECT_8CLASS_LIST,
    UNSEEN_LIVING_ROOM_OBJECT_8CLASS_LIST
)
rooms = ['Kitchen', 'Living_Room', 'Bedroom', 'Bathroom']


def name_to_num(name):
    return ["kitchen", "living_room", "bedroom", "bathroom"].index(name)


def num_to_name(num):
    return ["kitchen", "", "living_room", "bedroom", "bathroom"][int(num / 100)]


def get_data(scene_types, scenes):

    mapping = ["kitchen", "living_room", "bedroom", "bathroom"]
    idx = []
    for j in range(len(scene_types)):
        idx.append(mapping.index(scene_types[j]))

    scenes = [
        get_scenes("[{}]+{}".format(num + int(num > 0), scenes)) for num in [0, 1, 2, 3]
    ]

    possible_targets = FULL_OBJECT_CLASS_LIST

    targets = [
        KITCHEN_OBJECT_CLASS_LIST,
        LIVING_ROOM_OBJECT_CLASS_LIST,
        BEDROOM_OBJECT_CLASS_LIST,
        BATHROOM_OBJECT_CLASS_LIST,
    ]

    return [scenes[i] for i in idx], possible_targets, [targets[i] for i in idx], [rooms[i] for i in idx]

def get_seen_data(scene_types, scenes, split):

    mapping = ["kitchen", "living_room", "bedroom", "bathroom"]
    idx = []
    for j in range(len(scene_types)):
        idx.append(mapping.index(scene_types[j]))

    scenes = [
        get_scenes("[{}]+{}".format(num + int(num > 0), scenes)) for num in [0, 1, 2, 3]
    ]
    if split == "18/4":
        possible_targets = SEEN_FULL_OBJECT_18CLASS_LIST

        targets = [
            SEEN_KITCHEN_OBJECT_18CLASS_LIST,
            SEEN_LIVING_ROOM_OBJECT_18CLASS_LIST,
            SEEN_BEDROOM_OBJECT_18CLASS_LIST,
            SEEN_BATHROOM_OBJECT_18CLASS_LIST,
        ]
        print("Class Split:" + split + " Seen classes:" + str(len(possible_targets)))
    elif split == "14/8":
        possible_targets = SEEN_FULL_OBJECT_14CLASS_LIST

        targets = [
            SEEN_KITCHEN_OBJECT_14CLASS_LIST,
            SEEN_LIVING_ROOM_OBJECT_14CLASS_LIST,
            SEEN_BEDROOM_OBJECT_14CLASS_LIST,
            SEEN_BATHROOM_OBJECT_14CLASS_LIST,
        ]
        print("Class Split:" + split + " Seen classes:" + str(len(possible_targets)))

    return [scenes[i] for i in idx], possible_targets, [targets[i] for i in idx], [rooms[i] for i in idx]

def get_unseen_data(scene_types, scenes, split):

    mapping = ["kitchen", "living_room", "bedroom", "bathroom"]
    idx = []
    for j in range(len(scene_types)):
        idx.append(mapping.index(scene_types[j]))

    scenes = [
        get_scenes("[{}]+{}".format(num + int(num > 0), scenes)) for num in [0, 1, 2, 3]
    ]
    if split == "18/4":
        possible_targets = UNSEEN_FULL_OBJECT_4CLASS_LIST

        targets = [
            UNSEEN_KITCHEN_OBJECT_4CLASS_LIST,
            UNSEEN_LIVING_ROOM_OBJECT_4CLASS_LIST,
            UNSEEN_BEDROOM_OBJECT_4CLASS_LIST,
            UNSEEN_BATHROOM_OBJECT_4CLASS_LIST,
        ]
        print("Class Split:" + split + " Unseen classes:" + str(len(possible_targets)))

    elif split == "14/8":
        possible_targets = UNSEEN_FULL_OBJECT_8CLASS_LIST

        targets = [
            UNSEEN_KITCHEN_OBJECT_8CLASS_LIST,
            UNSEEN_LIVING_ROOM_OBJECT_8CLASS_LIST,
            UNSEEN_BEDROOM_OBJECT_8CLASS_LIST,
            UNSEEN_BATHROOM_OBJECT_8CLASS_LIST,
        ]
        print("Class Split:" + split + " Unseen classes:" + str(len(possible_targets)))

    return [scenes[i] for i in idx], possible_targets, [targets[i] for i in idx], [rooms[i] for i in idx]
