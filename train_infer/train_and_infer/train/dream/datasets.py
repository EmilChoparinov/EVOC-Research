# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from ast import Assert
from enum import IntEnum

import albumentations as albu
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms

import dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Debug mode:
# 0: no debug mode
# 1: light debug
# 2: heavy debug
class ManipulatorNDDSDatasetDebugLevels(IntEnum):
    # No debug information
    NONE = 0
    # Minor debug information, passing of extra info but not saving to disk
    LIGHT = 1
    # Heavy debug information, including saving data to disk
    HEAVY = 2
    # Interactive debug mode, not intended to be used for actual training
    INTERACTIVE = 3


class ManipulatorNDDSDataset(TorchDataset):
    def __init__(
        self,
        ndds_dataset,
        manipulator_name,
        keypoint_names,
        network_input_resolution,
        network_output_resolution,
        image_normalization,
        image_preprocessing,
        augment_data=False,
        include_ground_truth=True,
        include_belief_maps=False,
        debug_mode=ManipulatorNDDSDatasetDebugLevels["NONE"],
    ):
        # Read in the camera intrinsics
        self.ndds_dataset_data = ndds_dataset[0]
        self.ndds_dataset_config = ndds_dataset[1]
        self.manipulator_name = manipulator_name
        self.keypoint_names = keypoint_names
        self.network_input_resolution = network_input_resolution
        self.network_output_resolution = network_output_resolution
        self.augment_data = augment_data

        # If include_belief_maps is specified, include_ground_truth must also be
        # TBD: revisit better way of passing inputs, maybe to make one argument instead of two
        if include_belief_maps:
            assert (
                include_ground_truth
            ), 'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps

        self.debug_mode = debug_mode

        assert (
            isinstance(image_normalization, dict) or not image_normalization
        ), 'Expected image_normalization to be either a dict specifying "mean" and "stdev", or None or False to specify no normalization.'

        # Image normalization
        # Basic PIL -> tensor without normalization, used for visualizing the net input image
        self.tensor_from_image_no_norm_tform = TVTransforms.Compose(
            [TVTransforms.ToTensor()]
        )

        if image_normalization:
            assert (
                "mean" in image_normalization and len(image_normalization["mean"]) == 3
            ), 'When image_normalization is a dict, expected key "mean" specifying a 3-tuple to exist, but it does not.'
            assert (
                "stdev" in image_normalization
                and len(image_normalization["stdev"]) == 3
            ), 'When image_normalization is a dict, expected key "stdev" specifying a 3-tuple to exist, but it does not.'

            self.tensor_from_image_tform = TVTransforms.Compose(
                [
                    TVTransforms.ToTensor(),
                    TVTransforms.Normalize(
                        image_normalization["mean"], image_normalization["stdev"]
                    ),
                ]
            )
        else:
            # Use the PIL -> tensor tform without normalization if image_normalization isn't specified
            self.tensor_from_image_tform = self.tensor_from_image_no_norm_tform

        assert (
            image_preprocessing in dream.image_proc.KNOWN_IMAGE_PREPROC_TYPES
        ), 'Image preprocessing type "{}" is not recognized.'.format(
            image_preprocessing
        )
        self.image_preprocessing = image_preprocessing

    def __len__(self):
        return len(self.ndds_dataset_data)

    def __getitem__(self, index):

        # Parse this datum
        datum = self.ndds_dataset_data[index]
        image_rgb_path = datum["image_paths"]["rgb"]

        # Extract keypoints from the json file
        data_path = datum["data_path"]
        if self.include_ground_truth:
            keypoints_lists = dream.utilities.load_keypoints(
                data_path, self.manipulator_name, self.keypoint_names
            )
        else:
            # Generate an empty 'keypoints' dict
            keypoints_lists = dream.utilities.load_keypoints(
                data_path, self.manipulator_name, []
            )

        # Load image and transform to network input resolution -- pre augmentation
        image_rgb_raw = PILImage.open(image_rgb_path).convert("RGB")
        image_raw_resolution = image_rgb_raw.size

        # Do image preprocessing, including keypoint conversion
        image_rgb_before_aug = dream.image_proc.preprocess_image(
            image_rgb_raw, self.network_input_resolution, self.image_preprocessing
        )
        # This converts the keypoints to match the preproccesed image
        keypoints_lists["kp_projs_before_aug"] = []
        for projection_keypoints in keypoints_lists["projections"]:
            if projection_keypoints is None:
                kp_projs_before_aug = None
            else:
                kp_projs_before_aug = dream.image_proc.convert_keypoints_to_netin_from_raw(
                    projection_keypoints,
                    image_raw_resolution,
                    self.network_input_resolution,
                    self.image_preprocessing,
                )
            keypoints_lists["kp_projs_before_aug"].append(kp_projs_before_aug)
        # print("keypoint object looks like: {}".format(keypoints))

        # Handle data augmentation
        if self.augment_data:
            #数据增强
            augmentation = albu.Compose(
                [
                    albu.GaussNoise(),
                    #Adjusts brightness and contrast randomly to improve robustness to lighting changes.
                    albu.RandomBrightnessContrast(brightness_by_max=False),
                    #translate, scale and rotate the input.:
                    #albu.ShiftScaleRotate(rotate_limit=15),
                    #albu.SafeRotate(limit=15),
                    # albu.Crop,
                    # albu.CropAndPad,
                    # albu.Resize,
                ],
                p=1.0,
                keypoint_params=albu.KeypointParams(format="xy", remove_invisible=False),
            )
            keypoints_lists["kp_projs_net_input"] = []
            for keypoints_kp_projs_before_aug in keypoints_lists["kp_projs_before_aug"]:
                if keypoints_kp_projs_before_aug is None:
                    kp_projs_net_input = None
                else:
                    data_to_aug = {
                        "image": np.array(image_rgb_before_aug),
                        "keypoints": keypoints_kp_projs_before_aug,
                    }
                    augmented_data = augmentation(**data_to_aug)
                    image_rgb_net_input = PILImage.fromarray(augmented_data["image"])
                    kp_projs_net_input = augmented_data["keypoints"]
                keypoints_lists["kp_projs_net_input"].append(kp_projs_net_input)
        else:
            assert(False), "W.I.P"
            image_rgb_net_input = image_rgb_before_aug
            kp_projs_net_input = kp_projs_before_aug
        if 'image_rgb_net_input' not in locals():
            print("huh!!")
        assert (
            image_rgb_net_input.size == self.network_input_resolution
        ), "Expected resolution for image_rgb_net_input to be equal to specified network input resolution, but they are different."

        # Now convert keypoints at network input to network output for use as the trained label
        keypoints_lists["kp_projs_net_output"] = []
        for keypoints_kp_projs_net_input in keypoints_lists["kp_projs_net_input"]:
            if keypoints_kp_projs_net_input is None:
                kp_projs_net_output = None
            else:
                kp_projs_net_output = dream.image_proc.convert_keypoints_to_netout_from_netin(
                    keypoints_kp_projs_net_input,
                    self.network_input_resolution,
                    self.network_output_resolution,
                )
            keypoints_lists["kp_projs_net_output"].append(kp_projs_net_output)


        # Convert to tensor for output handling
        # This one goes through image normalization (used for inference)
        image_rgb_net_input_as_tensor = self.tensor_from_image_tform(
            image_rgb_net_input
        )

        # This one is not (used for net input overlay visualizations - hence "viz")
        image_rgb_net_input_viz_as_tensor = self.tensor_from_image_no_norm_tform(
            image_rgb_net_input
        )

        # Convert keypoint data to tensors - use float32 size
        keypoints_lists["keypoint_positions_wrt_cam_as_tensor"] = []
        for keypoints_positions_wrt_cam in keypoints_lists["positions_wrt_cam"]:
            if keypoints_positions_wrt_cam is None:
                keypoint_positions_wrt_cam_as_tensor = None
            else:
                keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                    np.array(keypoints_positions_wrt_cam)
                ).float()
            keypoints_lists["keypoint_positions_wrt_cam_as_tensor"].append(keypoint_positions_wrt_cam_as_tensor)

        keypoints_lists["kp_projs_net_output_as_tensor"] = []
        for keypoints_kp_projs_net_output in keypoints_lists["kp_projs_net_output"]:
            if keypoints_kp_projs_net_output is None:
                kp_projs_net_output_as_tensor = None
            else:
                kp_projs_net_output_as_tensor = torch.from_numpy(
                    np.array(keypoints_kp_projs_net_output)
                ).float()
            keypoints_lists["kp_projs_net_output_as_tensor"].append(kp_projs_net_output_as_tensor)


        # Construct output sample
        sample = {
            "image_rgb_input": image_rgb_net_input_as_tensor,
            # leaving these two out as they will lead to errors of different shapes
            #"keypoint_projections_output": kp_projs_net_output_as_tensor,
            #"keypoint_positions": keypoint_positions_wrt_cam_as_tensor,
            "config": datum,
        }

        # Generate the belief maps directly
        if self.include_belief_maps:
            image_width, image_height = self.network_output_resolution
            image_transpose_resolution = (image_height, image_width)

            belief_maps_joint_box = []
            for keypoints_kp_projs_net_output_as_tensor in keypoints_lists["kp_projs_net_output_as_tensor"]:
                if keypoints_kp_projs_net_output_as_tensor is None:
                    # there is no single box/joint: so we have an empty heat_map
                    # generate empty heatmap

                    belief_maps_joint_box.append(np.zeros(image_transpose_resolution))
                else:
                    # we land here: meaning that there is at least one box/joint
                    # so for each we generate an individual belief map.
                    # They will need to be merged into one belief map
                    to_merge_heat_map_list = []
                    for single_keypoint_kp_projection_net_output_as_tensor in keypoints_kp_projs_net_output_as_tensor:
                        belief_maps = dream.image_proc.create_individual_belief_map(
                            self.network_output_resolution, single_keypoint_kp_projection_net_output_as_tensor
                        )
                        to_merge_heat_map_list.append(belief_maps)
                    # merge belief_maps first
                    # if len(to_merge_heat_map_list) == 0:
                    #     print("huh!!, no heat maps to merge=================")
                    # merged_maps = to_merge_heat_map_list[0]
                    # for box_index in range(1, len(to_merge_heat_map_list)):
                    #     new_merge = dream.image_proc.combine_belief_maps(merged_maps, to_merge_heat_map_list[box_index])
                    #     merged_maps = new_merge
                    # belief_maps_joint_box.append(merged_maps)
                    if len(to_merge_heat_map_list) > 0:
                        merged_maps = to_merge_heat_map_list[0]
                        for box_index in range(1, len(to_merge_heat_map_list)):
                            new_merge = dream.image_proc.combine_belief_maps(merged_maps,
                                                                             to_merge_heat_map_list[box_index])
                            merged_maps = new_merge
                    else:
                        # Handle the case where there are no belief maps to merge
                        # For example, you could append an empty heatmap or take some other action
                        merged_maps = np.zeros(image_transpose_resolution)
                    belief_maps_joint_box.append(merged_maps)

            # now we should have a list of heat maps for all the parts in the train images. In my case box and joint
            # now merge these into a tensor of n_points x h x w with the belief maps
            complete_merge = np.zeros((len(belief_maps_joint_box), image_height, image_width))

            for n_index in range(0, len(belief_maps_joint_box)):
                complete_merge[n_index] = belief_maps_joint_box[n_index]

            belief_maps_as_tensor = torch.tensor(complete_merge).float()
            sample["belief_maps"] = belief_maps_as_tensor

        # if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["LIGHT"]:
        #     kp_projections_as_tensor = torch.from_numpy(
        #         np.array(keypoints["projections"])
        #     ).float()
        #     sample["keypoint_projections_raw"] = kp_projections_as_tensor
        #     kp_projections_input_as_tensor = torch.from_numpy(
        #         kp_projs_net_input
        #     ).float()
        #     sample["keypoint_projections_input"] = kp_projections_input_as_tensor
        #     image_raw_resolution_as_tensor = torch.tensor(image_raw_resolution).float()
        #     sample["image_resolution_raw"] = image_raw_resolution_as_tensor
        #     sample["image_rgb_input_viz"] = image_rgb_net_input_viz_as_tensor

        # TODO: same as LIGHT debug, but also saves to disk
        # if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["HEAVY"]:
        #     pass

        # Display to screen
        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["INTERACTIVE"]:
            # Ensure that the points are consistent with the image transformations
            # The overlaid points on both image should be consistent, despite image transformations
            for index, individual_projection in enumerate(keypoints_lists["projections"]):
                debug_image_raw = dream.image_proc.overlay_points_on_image(
                    image_rgb_raw, individual_projection#, self.keypoint_names
                )
                # doesn't work over ssh:
                # debug_image_raw.show()
                debug_image_raw.save(("raw_image_{}.jpg".format(str(index))))

            for index, individual_kp_projs_net_input in enumerate(keypoints_lists["kp_projs_net_input"]):
                debug_image = dream.image_proc.overlay_points_on_image(
                    image_rgb_net_input, individual_kp_projs_net_input#, self.keypoint_names
                )
                debug_image.save(("net_input_{}.jpg".format(str(index))))

            # Also show that the output resolution data are consistent
            image_rgb_net_output = image_rgb_net_input.resize(
                self.network_output_resolution, resample=PILImage.BILINEAR
            )
            for index_keypoint_type, output_keypoints in enumerate(keypoints_lists["kp_projs_net_output"]):
                debug_image_rgb_net_output = dream.image_proc.overlay_points_on_image(
                    image_rgb_net_output, output_keypoints#, self.keypoint_names
                )
                debug_image_rgb_net_output.save("debug_image_rgb_net_output_{}.png".format(index_keypoint_type))


            if self.include_belief_maps:
                for kp_idx in range(len(self.keypoint_names)):
                    belief_map_kp = dream.image_proc.image_from_belief_map(
                        belief_maps_as_tensor[kp_idx]
                    )
                    belief_map_kp.save("belief_map_of_{}.png".format(kp_idx))

            #         belief_map_kp_upscaled = belief_map_kp.resize(
            #             self.network_input_resolution, resample=PILImage.BILINEAR
            #         )
            #         image_rgb_net_output_belief_blend = PILImage.blend(
            #             image_rgb_net_input, belief_map_kp_upscaled, alpha=0.5
            #         )
            #         image_rgb_net_output_belief_blend_overlay = dream.image_proc.overlay_points_on_image(
            #             image_rgb_net_output_belief_blend,
            #             [kp_projs_net_input[kp_idx]],
            #             [self.keypoint_names[kp_idx]],
            #         )
            #         image_rgb_net_output_belief_blend_overlay.show()

            # This only works if the number of workers is zero
            # input("Press Enter to continue...")

        return sample

if __name__ == "__main__":
    from PIL import Image

    # beliefs = CreateBeliefMap((100,100),[(50,50),(-1,-1),(0,50),(50,0),(10,10)])
    # for i,b in enumerate(beliefs):
    #     print(b.shape)
    #     stack = np.stack([b,b,b],axis=0).transpose(2,1,0)
    #     im = Image.fromarray((stack*255).astype('uint8'))
    #     im.save('{}.png'.format(i))

    path = "/home/sbirchfield/data/FrankaSimpleHomeDR20k/"
    # path = '/home/sbirchfield/data/FrankaSimpleMPGammaDR105k/'

    keypoint_names = [
        "panda_link0",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link6",
        "panda_link7",
        "panda_hand",
    ]

    found_data = dream.utilities.find_ndds_data_in_dir(path)
    train_dataset = ManipulatorNDDSDataset(
        found_data,
        "panda",
        keypoint_names,
        (400, 400),
        (100, 100),
        include_belief_maps=True,
        augment_data=True,
    )
    trainingdata = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True
    )

    targets = iter(trainingdata).next()

    for i, b in enumerate(targets["belief_maps"][0]):
        print(b.shape)
        stack = np.stack([b, b, b], axis=0).transpose(2, 1, 0)
        im = Image.fromarray((stack * 255).astype("uint8"))
        im.save("{}.png".format(i))
