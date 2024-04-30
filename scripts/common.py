import numpy as np

YCB_data = {
    1: "01_master_chef_can", 2: "02_cracker_box", 3: "03_sugar_box", 4: "04_tomato_soup_can", 5: "05_mustard_bottle",
    6: "06_tuna_fish_can", 7: "07_pudding_box", 8: "08_gelatin_box", 9: "09_potted_meat_can", 10: "10_banana",
    11: "11_pitcher_base", 12: "12_bleach_cleanser", 13: "13_bowl", 14: "14_mug", 15: "15_power_drill",
    16: "16_wood_block", 17: "17_scissors", 18: "18_large_marker", 19: "19_large_clamp", 20: "20_extra_large_clamp",
    21: "21_foam_brick"
}

# reverse the dictionary
YCB_data = {v: k for k, v in YCB_data.items()}
YCB_data = {k: v for k, v in sorted(YCB_data.items(), key=lambda item: item[1])}


def load_intrinsics(transoform_json):
    fl_x = transoform_json['fl_x']
    fl_y = transoform_json['fl_y']
    k1 = transoform_json['k1']
    k2 = transoform_json['k2']
    k3 = transoform_json['k3']
    p1 = transoform_json['p1']
    p2 = transoform_json['p2']
    cx = transoform_json['cx']
    cy = transoform_json['cy']
    w = transoform_json['w']
    h = transoform_json['h']

    K = np.array([[fl_x, 0, cx],
                    [0, fl_y, cy],
                    [0, 0, 1]])
    return K, w, h


flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


