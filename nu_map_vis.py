
import matplotlib.pyplot as plt
import tqdm
import numpy

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

from nuscenes.nuscenes import NuScenes


def run():
    print("run main & finished")

def save_figure(layer_list, file_path, mode ='all'):

    if mode == 'all':
        fig, ax = nusc_map.render_map_patch(my_patch, layer_list, figsize=(10, 10), bitmap=bitmap)

        filename = f"{file_path}nuscenesmap_layer_all.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")

    elif mode =='all_split':
        for layer in layer_list:
            fig, ax = nusc_map.render_map_patch(my_patch, [layer], figsize=(10, 10), bitmap=bitmap)

            filename = f"{file_path}nuscenesmap_layer_{layer}.png"
            plt.savefig(filename)
            print(f"Saved: {filename}")

    elif mode == 'pdcms':
        fig, ax = nusc_map.render_map_patch(my_patch, layer_list, figsize=(10, 10), bitmap=bitmap)

        filename = f"{file_path}nuscenesmap_layer_pdcms_layer123.png"
        plt.savefig(filename)



if __name__=='__main__':
    nusc_map = NuScenesMap(dataroot='/home/kim/nas/nuscenes', map_name='singapore-onenorth')

    # map_name: singapore-onenorth, singepore-hollandvillage, singapore-queenstown, boston-seaport
    bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
    # my_patch = (0, 1100, 200, 1300)
    my_patch = (400, 1000, 600, 1200)
    my_layers = nusc_map.non_geometric_layers
    pdcms_layers = ['drivable_area', 'ped_crossing', 'walkway','carpark_area']
    save_figure(pdcms_layers, './', mode='pdcms')

    run()
