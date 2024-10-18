from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt

# NuScenes 데이터셋 로드
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/kim/nas/nuscenes_datasets/nuscenes_datasets/nuscenes', verbose=True)

def load_point_cloud(file_path, point_type):
    if point_type == 'lidar':
        pc = LidarPointCloud.from_file(file_path)
    elif point_type == 'radar':
        pc = RadarPointCloud.from_file(file_path)
    return pc.points

def transform_points(points, calibrated_sensor, ego_pose):
    # Sensor-to-ego 변환
    cs_rotation = Quaternion(calibrated_sensor['rotation'])
    cs_translation = np.array(calibrated_sensor['translation'])
    cs_transform = np.eye(4)
    cs_transform[:3, :3] = cs_rotation.rotation_matrix
    cs_transform[:3, 3] = cs_translation
    points = np.dot(cs_transform, np.vstack((points[:3, :], np.ones(points.shape[1]))))

    # Ego-to-global 변환
    ego_rotation = Quaternion(ego_pose['rotation'])
    ego_translation = np.array(ego_pose['translation'])
    ego_transform = np.eye(4)
    ego_transform[:3, :3] = ego_rotation.rotation_matrix
    ego_transform[:3, 3] = ego_translation
    points = np.dot(ego_transform, points)

    return points[:3, :]

def plot_bev_view(nusc, sample, lidar_points, radar_points_list, vehicle_anns):
    plt.figure(figsize=(15, 15))

    # GT 박스 시각화
    for ann in vehicle_anns:
        box = nusc.get_box(ann['token'])
        corners = box.corners()[:2, :]  # x, y 좌표만 추출
        plt.plot(np.append(corners[0], corners[0, 0]), np.append(corners[1], corners[1, 0]), 'g-', linewidth=2, label='GT Box')

    # 라이다 포인트 시각화
    plt.scatter(lidar_points[0, :], lidar_points[1, :], c='blue', s=0.1, label='Lidar Points')

    # 각 레이더 포인트 시각화
    radar_colors = ['red', 'orange', 'purple', 'brown', 'magenta']
    radar_labels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
    for i, radar_points in enumerate(radar_points_list):
        plt.scatter(radar_points[0, :], radar_points[1, :], c=radar_colors[i], s=10, label=radar_labels[i])

    plt.title('BEV Visualization with Lidar, 5-direction Radar, and GT Boxes')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 비율을 동일하게 설정하여 BEV 뷰 유지
    plt.show()

# 첫 번째 Scene 선택 및 첫 샘플 불러오기
scene = nusc.scene[0]
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

# 라이다 및 5방향 레이더 토큰 로드
lidar_token = sample['data']['LIDAR_TOP']
radar_channels = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']

# 라이다 및 레이더 포인트 로드
lidar_file = nusc.get_sample_data_path(lidar_token)
lidar_points = load_point_cloud(lidar_file, 'lidar')

# 레이더 포인트 로드 및 좌표 변환
radar_points_list = []
for radar_channel in radar_channels:
    radar_token = sample['data'][radar_channel]
    radar_file = nusc.get_sample_data_path(radar_token)
    radar_points = load_point_cloud(radar_file, 'radar')

    radar_cs = nusc.get('calibrated_sensor', nusc.get('sample_data', radar_token)['calibrated_sensor_token'])
    radar_ego = nusc.get('ego_pose', nusc.get('sample_data', radar_token)['ego_pose_token'])

    radar_points_transformed = transform_points(radar_points, radar_cs, radar_ego)
    radar_points_list.append(radar_points_transformed)

# 센서와 에고 포즈 정보 가져오기
lidar_cs = nusc.get('calibrated_sensor', nusc.get('sample_data', lidar_token)['calibrated_sensor_token'])
lidar_ego = nusc.get('ego_pose', nusc.get('sample_data', lidar_token)['ego_pose_token'])

# 좌표 변환
lidar_points_transformed = transform_points(lidar_points, lidar_cs, lidar_ego)

# 차량 어노테이션 불러오기 및 필터링
anns = [nusc.get('sample_annotation', ann_token) for ann_token in sample['anns']]
vehicle_anns = [ann for ann in anns if 'vehicle' in ann['category_name']]

# 시각화
plot_bev_view(nusc, sample, lidar_points_transformed, radar_points_list, vehicle_anns)


# from nuscenes.nuscenes import NuScenes
# import matplotlib.pyplot as plt

# # NuScenes 데이터셋 경로와 버전 설정
# nusc = NuScenes(version='v1.0-trainval', dataroot='/home/kim/nas/nuscenes_datasets/nuscenes_datasets/nuscenes', verbose=True)

# # 첫 번째 Scene 선택 및 첫 샘플 불러오기
# scene = nusc.scene[0]  # 첫 번째 Scene 선택
# first_sample_token = scene['first_sample_token']  # 첫 샘플 토큰 가져오기
# sample = nusc.get('sample', first_sample_token)  # 샘플 불러오기

# # 라이다 데이터 시각화 (LIDAR_TOP 데이터 사용)
# sensor_channel = 'LIDAR_TOP'
# nusc.render_sample_data(sample['data'][sensor_channel], with_anns=True)

# # BEV(Bird's Eye View)로 샘플 시각화
# nusc.render_sample(sample['token'], render_anns=True, box_vis_level='lidar')

# # 샘플의 모든 어노테이션 불러오기
# anns = [nusc.get('sample_annotation', ann_token) for ann_token in sample['anns']]

# # 차량만 필터링하여 시각화
# vehicle_anns = [ann for ann in anns if 'vehicle' in ann['category_name']]
# for ann in vehicle_anns:
#     nusc.render_annotation(ann['token'])

# plt.show()
