""" Contains the Episodes for Navigation. """
import random

import torch
from datasets.constants import UNSEEN_FULL_OBJECT_8CLASS_LIST,UNSEEN_FULL_OBJECT_4CLASS_LIST

from datasets.constants import GOAL_SUCCESS_REWARD, STEP_PENALTY
from datasets.constants import DONE
from datasets.environment import Environment
from datasets.glove import Glove
from utils.net_util import gpuify, toFloatTensor
from utils.action_util import get_actions
from utils.net_util import gpuify
from .episode import Episode
from utils import flag_parser

import json

c2p_prob = json.load(open("./data/c2p_prob.json"))
args = flag_parser.parse_arguments()

class BasicEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(BasicEpisode, self).__init__()

        self._env = None
        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.current_objs = None

        self.scene_states = []
        self.partial_reward = args.partial_reward
        self.seen_list = []
        if args.eval:
            random.seed(args.seed)
        self.room = None
        if args.zsd == True:
            if args.split == "18/4":
                n = 97
                self.unseen_objects = UNSEEN_FULL_OBJECT_4CLASS_LIST
            elif args.split == "14/8":
                n = 93
                self.unseen_objects = UNSEEN_FULL_OBJECT_8CLASS_LIST
        else:
            n = 101
            self.unseen_objects = []
        # glove embeddings for all the objs.
        self.objects = []
        with open("./data/gcn/objects.txt") as f:
            objects = f.readlines()
            for o in objects:
                o = o.strip()
                if args.zsd == True:
                    if o in self.unseen_objects:
                        continue
                    else:
                        self.objects.append(o)
                else:
                    self.objects.append(o)

        self.all_glove = torch.zeros(n, 300)
        glove = Glove(args.glove_file)
        for i in range(n):
            self.all_glove[i, :] = torch.Tensor(glove.glove_embeddings[self.objects[i]][:])

        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.CS_MAX = 0
        self.state_list = []

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def objstate_for_agent(self):
        return self.environment.current_objs

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int):

        action = self.actions_list[action_as_int]

        # if args.vis:
        #     print(action)

        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        reward, terminal, action_was_successful = self.judge(action)
        return reward, terminal, action_was_successful

    def judge(self, action):
        """ Judge the last event. """
        reward = STEP_PENALTY
        # Thresholding replaced with simple look up for efficiency.
        if self.environment.controller.state in self.scene_states:
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
                # added partial reward
                if self.partial_reward:
                    if args.model == "SelfAttention_test":
                        reward = self.get_partial_reward_SA()
                    else:
                        reward = self.get_partial_reward()
        else:
            self.scene_states.append(self.environment.controller.state)

        done = False

        if action["action"] == DONE:
            action_was_successful = False
            for id_ in self.task_data:
                if self.environment.object_is_visible(id_):
                    reward = GOAL_SUCCESS_REWARD
                    done = True
                    action_was_successful = True
                    if self.partial_reward:
                        self.seen_list = []
                        self.CS_MAX = 0
                        if args.model == "SelfAttention_test":
                            reward += self.get_partial_reward_SA()
                        else:
                            reward += self.get_partial_reward()
                    break
            self.seen_list = []
            self.CS_MAX = 0
            self.state_list = []
            # if args.vis:
                # print("Success:", action_was_successful)
        else:
            action_was_successful = self.environment.last_action_success

        return reward, done, action_was_successful

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(
            torch.LongTensor([target_object_index]), self.gpu_id
        )

    def get_partial_reward_SA(self):
        reward = STEP_PENALTY
        ind = self.objects.index(self.target_object)
        tgt_embedding = self.all_glove[ind, :]
        curren_objs = self.environment.current_objs
        for obj in curren_objs.keys():
            if obj in self.objects:
                ind = self.objects.index(obj)
                obj_embedding = self.all_glove[ind,:]
                CS = self.cos(obj_embedding,tgt_embedding)
                if CS>self.CS_MAX:
                    self.CS_MAX = CS
                    reward = self.CS_MAX*0.1
        return reward

    def get_partial_reward(self):
        reward = STEP_PENALTY
        reward_dict = {}
        if self.target_parents is not None:
            for parent_type in self.target_parents:
                parent_ids = self.environment.find_id(parent_type)
                for parent_id in parent_ids:
                    if self.environment.object_is_visible(parent_id) and parent_id not in self.seen_list:
                        reward_dict[parent_id] = self.target_parents[parent_type]
        if len(reward_dict) != 0:
            v = list(reward_dict.values())
            k = list(reward_dict.keys())
            reward = max(v)
            self.seen_list.append(k[v.index(reward)])
        return reward

    def _new_episode(
        self, args, scenes, possible_targets, targets=None, room = None, keep_obj=False, glove=None
    ):
        """ New navigation episode. """
        scene = random.choice(scenes)
        self.room = room
        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=0.25,
                images_file_name=args.images_file_name,
                local_executable_path=args.local_executable_path,
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        # Randomize the start location.
        start_state = self._env.randomize_agent_location()
        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]

        self.task_data = []

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        child_object = self.task_data[0].split("|")[0]
        # print('room is ', self.room)
        try:
            self.target_parents = c2p_prob[self.room][child_object]
        except:
            self.target_parents = None

        if args.verbose:
            print("Scene", scene, "Navigating towards:", goal_object_type)

        self.glove_embedding = None
        self.glove_embedding = toFloatTensor(
            glove.glove_embeddings[goal_object_type][:], self.gpu_id
        )

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        rooms=None,
        keep_obj=False,
        glove=None,
    ):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None
        self.current_objs = None
        self._new_episode(args, scenes, possible_targets, targets, rooms, keep_obj, glove)
