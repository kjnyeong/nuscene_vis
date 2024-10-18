import os
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
import cv2

# Initialize NuScenes with the 'trainval' version
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/kim/nas/nuscenes_datasets/nuscenes_datasets/nuscenes', verbose=True)

# 레이더 및 라이다 파일명을 기반으로 샘플 토큰 찾기
def find_sample_token_by_filename(radar_filename, nusc):
    filename_only = os.path.basename(radar_filename)
    for record in nusc.sample_data:
        if filename_only in record['filename']:
            return record['sample_token']
    raise ValueError(f"파일명을 기반으로 샘플 토큰을 찾을 수 없습니다: {filename_only}")

# 레이더 포인트 클라우드를 로드하고 변환하는 함수
def load_and_transform_radar_pointcloud(nusc, radar_token, ego_position):
    pointsensor = nusc.get('sample_data', radar_token)
    pc = RadarPointCloud.from_file(os.path.join(nusc.dataroot, pointsensor['filename']))

    # Calibrated sensor transformation
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Ego pose translation only (회전 없이 평행 이동만 적용)
    ego_record = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.translate(np.array(ego_record['translation']))

    # 자차 위치를 기준으로 상대 좌표계로 변환
    pc.translate(-ego_position)

    # 90도 회전 적용 (Z축 기준 회전)
    rotation_90_degrees = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pc.rotate(rotation_90_degrees)

    return pc

# 시각화 및 저장 함수
def visualize_and_save_radar_with_projection(nusc, sample_token, map_image_path, radar_point_size, save_path):
    sample_record = nusc.get('sample', sample_token)

    ego_pose = nusc.get('ego_pose', sample_record['data']['LIDAR_TOP'])
    ego_position = np.array(ego_pose['translation'])

    radar_tokens = {
        'front': sample_record['data']['RADAR_FRONT'],
        'front_left': sample_record['data']['RADAR_FRONT_LEFT'],
        'front_right': sample_record['data']['RADAR_FRONT_RIGHT'],
        'back_left': sample_record['data']['RADAR_BACK_LEFT'],
        'back_right': sample_record['data']['RADAR_BACK_RIGHT']
    }

    all_radar_points = []
    for key, radar_token in radar_tokens.items():
        radar_pc = load_and_transform_radar_pointcloud(nusc, radar_token, ego_position)
        all_radar_points.append(radar_pc.points[:3, :])

    radar_points_combined = np.concatenate(all_radar_points, axis=1)

    fig, ax = plt.subplots(figsize=(10, 10))

    map_img = cv2.imread(map_image_path, cv2.IMREAD_UNCHANGED)
    if map_img is None:
        print(f"Error: Could not load image from {map_image_path}")
        return

    if map_img.shape[-1] == 4:
        alpha_channel = map_img[:, :, 3]
        rgb_img = cv2.cvtColor(map_img, cv2.COLOR_BGRA2BGR)
        background = np.ones_like(rgb_img, dtype=np.uint8) * 255
        alpha_factor = alpha_channel[:, :, np.newaxis] / 255.0
        map_img = rgb_img * alpha_factor + background * (1 - alpha_factor)

    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)

    ax.set_position([0, 0, 1, 1])
    ax.set_aspect('equal', 'box')

    extent = [-50, 50, -50, 50]
    ax.imshow(map_img, extent=extent, aspect='auto')

    # 라이다 시각화는 하지 않음
    ax.scatter(radar_points_combined[0, :], radar_points_combined[1, :], s=radar_point_size, c='r', label='Radar Points')

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_axis_off()

    # 이미지 저장
    plt.savefig(save_path)
    plt.close()

# 여러 Scene 폴더를 처리하는 함수
def process_all_scenes(nusc, base_image_path, radar_base_path, save_base_path, radar_point_size=2):
    os.makedirs(save_base_path, exist_ok=True)  # 저장 폴더 생성 (한 번만)

    for scene_idx in range(40):
        scene_folder = f"scene {scene_idx}"
        image_folder = os.path.join(base_image_path, scene_folder)
        radar_folder = os.path.join(radar_base_path, scene_folder)

        # 이미지 파일 목록 동적으로 확인
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        for idx, image_filename in enumerate(image_files):
            bev_image_path = os.path.join(image_folder, image_filename)

            # 레이더 파일 예시에서 RADAR_FRONT 파일을 찾음
            radar_filename = [f for f in os.listdir(radar_folder) if 'RADAR_FRONT' in f][0]
            radar_full_path = os.path.join(radar_folder, radar_filename)

            try:
                sample_token = find_sample_token_by_filename(radar_full_path, nusc)
                # 결과 파일을 하나의 폴더에 저장
                save_path = os.path.join(save_base_path, f"result_scene_{scene_idx:04d}_img_{idx:04d}.png")
                visualize_and_save_radar_with_projection(nusc, sample_token, bev_image_path, radar_point_size, save_path)
                print(f"Processed and saved scene {scene_idx}, image {idx}")
            except Exception as e:
                print(f"Error processing scene {scene_idx}, image {idx}: {e}")

# 실행
process_all_scenes(nusc, "/home/kim/nuscenes_pratice/dataprocess/scene-0004/image", "/home/kim/nuscenes_pratice/dataprocess/scene-0004/radar", "/home/kim/nuscenes_pratice/dataprocess/scene-0004/results_radar00", radar_point_size=2)
