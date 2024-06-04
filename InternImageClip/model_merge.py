import os
import argparse
import cv2
import numpy as np

pred_weights = [[0.56028428, 0.45633311, 0.30659414, 0.78656681, 0.20652187
                    , 0.39475866, 0.31585305, 0.53421764, 0.74223141],
                [0.52408918, 0.32324631, 0.26253174, 0.83887334, 0.17266217
                    , 0.53008834, 0.41071099, 0.53685896, 0.68691321],
                [0.50805455, 0.3748674, 0.30024447, 0.80033509, 0.18874887
                    , 0.55144693, 0.50638191, 0.3992809, 0.68362116],
                [0.56407799, 0.39486307, 0.22769779, 0.82681815, 0.21469049
                    , 0.52657907, 0.38212802, 0.58088675, 0.68311093],
                [0.53722111, 0.38366476, 0.30381604, 0.83195799, 0.19572608,
                 0.53412688, 0.58182947, 0.56599924, 0.68431602],
                [0.45981711, 0.36783135, 0.31139273, 0.78748517, 0.18271955
                    , 0.55082975, 0.49648015, 0.38642368, 0.70966688]]
pred_weights = np.array(pred_weights).T
pred_sum = np.sum(pred_weights, axis=1)[..., None]
pred_weights = pred_weights / pred_sum
pred_weights = np.ones_like(pred_weights)
colors_map = {
    0: (0, 0, 0),  # Background
    9: (36, 179, 83),  # 9-tree
    2: (250, 250, 55),  # 2-structure
    5: (112, 128, 0),  # 5-stone
    6: (52, 209, 183),  # 6-terrain-vegetation
    1: (8, 100, 193),  # 1-building
    7: (255, 0, 43),  # 7-terrain-other
    3: (204, 155, 59),  # 3-road
    4: (49, 179, 245),  # 4-sky
    8: (255, 0, 204),  # 8--snow
}
def parse_args():
    parser = argparse.ArgumentParser(description='merge the segmentation results from six models')
    parser.add_argument('rgb_path',help='WeatherProof test datset path')
    parser.add_argument('label_folder', help='the folder containing the segmentation results to be merged')
    parser.add_argument('out_label_folder', help='the folder to save the final merge result')
    parser.add_argument('--simplify', action='store_true', help='simplify the merge process by only using the first image in each subfolder')

    return parser.parse_args()

def convert_mask_to_color_map(mask_path, img_path):
    # Load the image and segmentation mask
    # image_path = os.path.splitext(mask_path)[0] + '.jpg'
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask + 1  # submit

    vis_colored = np.zeros_like(image)
    for label, color in colors_map.items():
        vis_colored[mask == label] = color
    vis_colored = cv2.cvtColor(vis_colored, cv2.COLOR_RGB2BGR)
    return vis_colored, mask


def one_hot_encode(matrix, num_classes=9):
    shape = matrix.shape

    one_hot_matrix = np.zeros((shape[0], shape[1], num_classes))

    one_hot_matrix[
        np.arange(shape[0])[:, None], np.arange(shape[1]), matrix.squeeze()] = 1

    return one_hot_matrix


def merge_segmentation_predictions(preds):
    assert len(
        preds) == 6, "Input should contain 6 prediction images."
    height, width = preds[0].shape
    preds_np = np.array(preds).reshape((6, -1))[..., None] - 1
    one_hot_pred = one_hot_encode(preds_np)
    weighed_pred = (one_hot_pred * pred_weights.T[:, None, :]).transpose(
        (1, 2, 0))
    weighed_pred = np.sum(weighed_pred, axis=-1)
    try:
        weighed_pred = np.argmax(weighed_pred, axis=-1).reshape((height, width))
    except:
        print("error")

    return weighed_pred


def main():
    args = parse_args()
    label_folder = args.label_folder
    out_label_dir = args.out_label_folder
    rgb_path = args.rgb_path
    simplify = args.simplify
    label_order = ['building', 'structure', 'road', 'sky', 'stone',
                   'terrain-vegetation', 'terrain-other', 'terrain-snow',
                   'tree']

    for subdir, _, files in os.walk(rgb_path):
        sub_name = os.path.basename(subdir)
        print('sub_name: ', sub_name)
        results = []
        if subdir != rgb_path:  # Skip the root directory
            print('subdir: ', subdir)

            for filename in files:
                masks = []
                if simplify:
                    if filename != 'frame_000000.png':
                        continue
                img_path = os.path.join(subdir, filename)
                for labelroot, maskdirs, _ in os.walk(label_folder):
                    for maskdir in maskdirs:
                        mask_folder = os.path.join(label_folder, maskdir,
                                                   sub_name)
                        for maskname in os.listdir(mask_folder):
                            mask_path = os.path.join(mask_folder, maskname)
                            if not os.path.exists(mask_path):
                                continue
                            result, mask = convert_mask_to_color_map(
                                mask_path, img_path)
                            masks.append(mask)
                            results.append(result)
                            break
                    break
                finalmask = merge_segmentation_predictions(masks)

                if not os.path.exists(
                        os.path.join(out_label_dir, sub_name)):
                    os.makedirs(os.path.join(out_label_dir, sub_name))
                cv2.imwrite(os.path.join(out_label_dir, sub_name, filename),
                            finalmask)

                new_name = filename
                new_path = os.path.join(out_label_dir, sub_name,
                                        new_name)
                cv2.imwrite(new_path, finalmask)
            if simplify:
                for frame in range(1,300):
                    new_name = 'frame_{:06d}.png'.format(frame)
                    new_path = os.path.join(out_label_dir, sub_name,
                                            new_name)
                    cv2.imwrite(new_path, finalmask)

if __name__ == '__main__':
    main()
