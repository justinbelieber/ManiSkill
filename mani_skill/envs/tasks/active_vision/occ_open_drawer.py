from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh


from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda, PandaWristCam, Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder



CABINET_COLLISION_BIT = 29


# TODO (stao): we need to cut the meshes of all the cabinets in this dataset for gpu sim, there may be some wierd physics
# that may happen although it seems okay for state based RL
@register_env(
    "occ_open_drawer",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=1000,
)
class OccOpenDrawerEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/OpenCabinetDrawer-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, PandaWristCam, Fetch]
    handle_types = ["prismatic"]

    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )

    min_open_frac = 0.75

    def __init__(
        self,
        *args,
        # robot_uids="panda",
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        cube_half_size=0.20,
        table_height=0.99,
        num_envs=1,
        **kwargs,
    ):
        self.table_height = table_height
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cube_half_size = cube_half_size
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        # self.all_model_ids = np.array(["1004", "1004"])
        if reconfiguration_freq is None:
            # if not user set, we pick a number
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=512,
                height=512,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-1.8, -1.3, 1.8], target=[-0.3, 0.5, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[2, 0, 0]))
        # 检查 agent 是否有 hand_camera
        if hasattr(self.agent, "cameras") and "hand_camera" in self.agent.cameras:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Setting hand_camera resolution to 512x512")
            cam = self.agent.cameras["hand_camera"]
            cam.width = 512   # 你想要的宽度
            cam.height = 512  # 你想要的高度

    def _load_scene(self, options: dict):
        # 添加桌子
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=0.02)
        self.table_scene.build()

        # self.obj = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=np.array([12, 42, 160, 255]) / 255,
        #     name="cube",
        #     body_type="dynamic",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )
        # 添加长方体
        self.obj = actors.build_box(
            self.scene,
            half_sizes=[0.05, self.cube_half_size, self.cube_half_size],
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.2, 0.2, self.cube_half_size]),
        )


        self._load_cabinets(self.handle_types)



    def _load_cabinets(self, joint_types: List[str]):
        # we sample random cabinet model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2**31)

        self._cabinets: List[Articulation] = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            cabinet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            cabinet_builder.set_scene_idxs(scene_idxs=[i])

            # 随机化 初始平面位置和姿态
            rand_xy = self._batched_episode_rng.uniform(
                low=[0, 0], high=[0, 0.5], size=(2,)
            )
            # print(f"rand_xy: {rand_xy}")

            # rand_yaw = self._batched_episode_rng.uniform(low=-np.pi, high=np.pi).item()
            rand_yaw = 0  # for debugging, we set yaw to 0
            q = [0.0, 0.0, np.sin(rand_yaw / 2), np.cos(rand_yaw / 2)]

            # print(f"rand_yaw: {rand_yaw}, q: {q}")
            # print(f"rand_xy: {rand_xy}, rand_yaw: {rand_yaw}")
            # print(f"rand_yaw: {rand_yaw}, q: {q}")
            cabinet_builder.initial_pose = sapien.Pose(
                p=[rand_xy[0][0], rand_xy[0][1], 0.0],
                q=q,
            )
            cabinet = cabinet_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(cabinet)
            # this disables self collisions by setting the group 2 bit at CABINET_COLLISION_BIT all the same
            # that bit is also used to disable collision with the ground plane
            for link in cabinet.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
                )
            self._cabinets.append(cabinet)
            handle_links.append([])
            handle_links_meshes.append([])

            # TODO (stao): At the moment code for selecting semantic parts of articulations
            # is not very simple. Will be improved in the future as we add in features that
            # support part and mesh-wise annotations in a standard querable format
            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in joint_types:
                    handle_links[-1].append(link)
                    # save the first mesh in the link object that correspond with a handle
                    handle_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "handle"
                            in render_shape.name,
                            mesh_name="handle",
                        )[0]
                    )

        # we can merge different articulations/links with different degrees of freedoms into a single view/object
        # allowing you to manage all of them under one object and retrieve data like qpos, pose, etc. all together
        # and with high performance. Note that some properties such as qpos and qlimits are now padded.
        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        self.handle_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            name="handle_link",
        )
        # store the position of the handle mesh itself relative to the link it is apart of
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            ),
            device=self.device,
        )

        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.0000001,
            color=[0, 0, 0, 0],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 1], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options):
        # To spawn cabinets in the right place, we need to change their z position such that
        # the bottom of the cabinet sits at z=0 (the floor). Luckily the partnet mobility dataset is made such that
        # the negative of the lower z-bound of the collision mesh bounding box is the right value

        # this code is in _after_reconfigure since retrieving collision meshes requires the GPU to be initialized
        # which occurs after the initial reconfigure call (after self._load_scene() is called)
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)

        # get the qmin qmax values of the joint corresponding to the selected links
        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        with torch.device(self.device):
            b = len(env_idx)
            # 初始化桌子
            self.table_scene.initialize(env_idx)
            # 初始化橱柜
            table_top_z = self.table_scene.table_height
            print("Table Height:", table_top_z)
            xy = torch.zeros((b, 3)) # [b, 3] tensor
            xy[:, 2] = self.cabinet_zs[env_idx] # + table_top_z # set z position of the cabinet
            print("self.cabinet_zs:", self.cabinet_zs[env_idx])
            xy[:, 0] = common.to_tensor(self._batched_episode_rng.uniform(
                low=0.3, high=0.3, size=(b,)
            )[0], device=self.device)
            xy[:, 1] = common.to_tensor(self._batched_episode_rng.uniform(
                low=0.3, high=0.3, size=(b,)
            )[0], device=self.device)
            # set the pose of the cabinet
            
            rand_yaw = self._batched_episode_rng.uniform(
                low=np.pi / 12.0 * 5, high=np.pi / 12.0 * 5, size=(b,)
                # low=- np.pi / 4.0, high = np.pi / 4.0, size=(b,)
            )[0]
            # rand_yaw = 0  # for debugging, we set yaw to 0

            q = torch.zeros((b, 4), device=self.device)
            q[:, 3] = common.to_tensor(
                np.sin(rand_yaw / 2), device=self.device
            )
            q[:, 0] = common.to_tensor(
                np.cos(rand_yaw / 2), device=self.device
            )

            print("env_idx:", env_idx)
            print("xy:", xy)
            print("q:", q)
            self.cabinet.set_pose(Pose.create_from_pq(p=xy, q=q))
            
            # 初始化物体
            # self.obj.set_pose(Pose.create_from_pq(p=[0.2, 0.2, 1.0]))

            # close all the cabinets. We know beforehand that lower qlimit means "closed" for these assets.
            qlimits = self.cabinet.get_qlimits()  # [b, self.cabinet.max_dof, 2])
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            # NOTE (stao): This is a temporary work around for the issue where the cabinet drawers/doors might open
            # themselves on the first step. It's unclear why this happens on GPU sim only atm.
            # moreover despite setting qpos/qvel to 0, the cabinets might still move on their own a little bit.
            # this may be due to oblong meshes.
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

            # 物体的位置放在 handle_link_positions 和 机械臂 之间
            panda_tcp_pos = self.agent.tcp.pose.p
            # 挡在 handle_link 和 机械臂 之间
            # obj_pos = (self.handle_link_positions(env_idx) + panda_tcp_pos) / 2

            # 放在很远的位置
            obj_pos = self.handle_link_positions(env_idx) + torch.tensor([10.0, 10.0, 0.0], device=self.device)
            self.obj.set_pose(Pose.create_from_pq(p=obj_pos))


    def _after_control_step(self):
        # after each control step, we update the goal position of the handle link
        # for GPU sim we need to update the kinematics data to get latest pose information for up to date link poses
        # and fetch it, followed by an apply call to ensure the GPU sim is up to date
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.handle_link_goal.set_pose(
            Pose.create_from_pq(p=self.handle_link_positions())
        )
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        # even though self.handle_link is a different link across different articulations
        # we can still fetch a joint that represents the parent joint of all those links
        # and easily get the qpos value.
        open_enough = self.handle_link.joint.qpos >= self.target_qpos
        handle_link_pos = self.handle_link_positions()

        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
        return {
            "success": open_enough & link_is_static,
            "handle_link_pos": handle_link_pos,
            "open_enough": open_enough,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                target_handle_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
        amount_to_open_left = torch.div(
            self.target_qpos - self.handle_link.joint.qpos, self.target_qpos
        )
        open_reward = 2 * (1 - amount_to_open_left)
        reaching_reward[
            amount_to_open_left < 0.999
        ] = 2  # if joint opens even a tiny bit, we don't need reach reward anymore
        # print(open_reward.shape)
        open_reward[info["open_enough"]] = 3  # give max reward here
        reward = reaching_reward + open_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
