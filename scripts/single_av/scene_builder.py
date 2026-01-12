import sapien.core as sapien
import numpy as np


def create_scene():
    # 初始化 SAPIEN 引擎和渲染器
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    # 创建物理场景
    scene_config = sapien.SceneConfig()
    scene_config.gravity = [0, 0, -9.81]
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240)

    # 添加灯光
    scene.add_directional_light(np.array([1, -1, -1]), color=[1, 1, 1])
    scene.add_point_light(position=[1, 1, 1], color=[1, 1, 1])

    # 添加地面
    ground_material = scene.create_physical_material(0.8, 0.8, 0.01)
    scene.add_ground(altitude=0, render=True, material=ground_material)

    # 加载 Panda 机械臂
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    panda = loader.load(
        "maniskill_assets/descriptions/panda/panda.urdf"
    )
    panda.set_pose(sapien.Pose([0, 0, 0]))

    # 加载橱柜（以 OpenCabinetDrawer 任务中使用的为例）
    cabinet = loader.load(
        "maniskill_assets/descriptions/open_cabinet/39467/mobility.urdf"
    )
    cabinet.set_pose(sapien.Pose([0.5, 0.0, 0.0]))

    # 添加几个红色的方块
    builder = scene.create_actor_builder()
    box_half_size = [0.02, 0.02, 0.02]
    builder.add_box_collision(half_size=box_half_size)
    builder.add_box_visual(half_size=box_half_size, color=[1, 0, 0])
    builder.set_mass(0.1)

    for i in range(3):
        box = builder.build(name=f"cube_{i}")
        box.set_pose(sapien.Pose([0.4 + 0.05 * i, -0.1, 0.3]))

    # 创建查看器并渲染
    viewer = sapien.SapienViewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(1.2, 0.0, 0.6)
    viewer.set_camera_rpy(0, 0, -np.pi / 2)

    print("启动查看器...")
    while not viewer.closed:
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == "__main__":
    create_scene()
