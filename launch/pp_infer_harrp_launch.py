import os
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():

    # コンポーネントの定義
    pointpillars_component = ComposableNode(
        package='pp_infer',
        plugin='pp_infer::PointPillarsHarrpNode', # 登録したクラス名
        name='pp_infer_harrp',
        parameters=[{
            'nms_iou_thresh': 0.01,
            'pre_nms_top_n': 4096,
            'class_names': ['Pedestrian'],
            # 提示されたパスをそのまま適用
            'model_path': '/home/demulab-kohei/colcon_ws/src/ros2_tao_pointpillars/include/harrp_epoch400.onnx', 
            # 注意: ユーザー名が demulab-kohei ではなく nvidia になっていますが、提示された通りにしています
            'engine_path': '/home/nvidia/Projects/PointPillars/trt.fp16.engine',
            'data_type': 'fp32',
            'intensity_scale': 255.0,
        }],
        remappings=[
            ('/point_cloud', '/livox/lidar/no_ground')
        ],
        # コンポーネント単体で動かす場合も、将来的な通信効率化のために設定しておくと良いです
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # コンテナの定義
    # コンポーネントは「実行ファイル」ではなく、この「コンテナ」の中で動きます
    container = ComposableNodeContainer(
        name='pointpillars_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            pointpillars_component
        ],
        output='screen',
    )

    return LaunchDescription([
        container
    ])
