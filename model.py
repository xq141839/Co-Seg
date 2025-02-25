import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
import cv2
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
from sam2.utils.dcpp import DetectionCellPostProcessor, calculate_instances
from skimage.color import rgba2rgb
from skimage import io
from sam2.utils.metrics import get_fast_pq, remap_label, get_fast_aji
from typing import List, Tuple, Type, Optional
import os
from tqdm import tqdm
from functools import partial
from typing import List, Literal, Tuple, Union
from collections import OrderedDict
from torchmetrics.functional.classification import binary_jaccard_index
from sam2.utils.tools import remove_small_objects


def InsGenerator(mask, visual=False):

    mask = mask.cpu().detach().numpy()[0][0]
    # instance_id = 0
    mask_shape = mask.shape
    targets = np.zeros(mask_shape, dtype=np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8)*255)
    for i in range(1, num_labels):
        targets[labels == i] = i

    targets = remove_small_objects(targets, 25000)
    return targets

class CoSeg(nn.Module):
    def __init__(self, sam_model = SAM2Base, dim=24, img_size=1024):
        super(CoSeg, self).__init__()

        self.num_nuclei_classes = 2
        self.device = "cuda"
        self._transforms = SAM2Transforms(
            resolution=img_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.model = sam_model
        self._features = None
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=(1024, 1024) # fixed
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=(1024, 1024)
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_sem: Optional[torch.Tensor] = None,
        mask_ins: Optional[torch.Tensor] = None,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)


        sparse_embeddings_sem, dense_embeddings_sem = self.model.RP_encoder_ins(
            points=None,
            boxes=None,
            masks=mask_ins,
            image=self._features["image_embed"][img_idx].unsqueeze(0),
        )

        sparse_embeddings_ins, dense_embeddings_ins = self.model.RP_encoder_sem(
            points=None,
            boxes=None,
            masks=mask_sem,
            image=self._features["image_embed"][img_idx].unsqueeze(0),
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]

        low_res_masks_ins = self.model.MP_decoder_ins(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.RP_encoder_sem.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings_ins,
            dense_prompt_embeddings=dense_embeddings_ins,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        low_res_masks_sem = self.model.MP_decoder_sem(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.RP_encoder_ins.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings_sem,
            dense_prompt_embeddings=dense_embeddings_sem,
            multimask_output=False,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        mask_ins = F.interpolate(low_res_masks_ins, (1024, 1024), mode="bilinear", align_corners=False)
        mask_sem = F.interpolate(low_res_masks_sem, (1024, 1024), mode="bilinear", align_corners=False)
        return mask_ins, mask_sem
    
    def calculate_step_metric_validation(self, predictions: dict, gt: dict) -> dict:
        """Calculate the metrics for the training step

        Args:
            predictions (DataclassHVStorage): OrderedDict: Processed network output
            gt (DataclassHVStorage): Ground truth values
        Returns:
            dict: Dictionary with metrics. Keys:
                binary_dice_scores, binary_jaccard_scores, tissue_pred, tissue_gt
        """
        # predictions = predictions.get_dict()
        # gt = gt.get_dict()

        # Tissue Tpyes logits to probs and argmax to get class

        predictions["nuclei_binary_map"][predictions["nuclei_binary_map"] >= 0.5] = 1
        predictions["nuclei_binary_map"][predictions["nuclei_binary_map"] < 0.5] = 0
    
        predictions["instance_map"] = predictions["instance_map"].detach().cpu()
        predictions["instance_types_nuclei"] = (
            predictions["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )
        instance_maps_gt = gt["instance_map"].detach().cpu()[0]
        # gt["nuclei_binary_map"] = torch.argmax(gt["nuclei_binary_map"], dim=1).type(
        #     torch.uint8
        # )
        gt["instance_types_nuclei"] = (
            gt["instance_types_nuclei"].detach().cpu().numpy().astype("int32")
        )

        binary_dice_scores = []
        binary_jaccard_scores = []
        cell_type_pq_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        aji_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # binary dice score: Score for cell detection per image, without background
        pred_binary_map = predictions["nuclei_binary_map"]
        target_binary_map = gt["nuclei_binary_map"]

        # binary aji
        cell_jaccard = (
            binary_jaccard_index(
                preds=pred_binary_map,
                target=target_binary_map,
            )
            .detach()
            .cpu()
        )
        binary_jaccard_scores.append(float(cell_jaccard))
        # pq values
        
        remapped_instance_pred = remap_label(predictions["instance_map"])
        remapped_gt = remap_label(instance_maps_gt)
        # print(remapped_instance_pred.shape, remapped_gt.shape)
        [dq, sq, pq], _, precision, recall, f1 = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
        pq_scores.append(pq)
        dq_scores.append(dq)
        sq_scores.append(sq)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        aji = get_fast_aji(true=remapped_gt, pred=remapped_instance_pred)
        aji_scores.append(aji)

        batch_metrics = {
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "dq_scores": dq_scores,
            "sq_scores": sq_scores,
            "aji_scores": aji_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "f1_scores": f1_scores,
        }


        return batch_metrics
    
    def calculate_tissue_metric_validation(self, predictions, gt):


        pred_binary_map = InsGenerator(predictions)
        target_binary_map = InsGenerator(gt)

        pred_binary_map = remap_label(pred_binary_map)
        target_binary_map = remap_label(target_binary_map)


        binary_dice_scores = []
        binary_jaccard_scores = []
        cell_type_pq_scores = []
        pq_scores = []
        dq_scores = []
        sq_scores = []
        aji_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # binary dice score: Score for cell detection per image, without background
        # pred_binary_map = predictions
        # target_binary_map = gt

        # binary aji
        # cell_jaccard = (
        #     binary_jaccard_index(
        #         preds=pred_binary_map,
        #         target=target_binary_map,
        #     )
        #     .detach()
        #     .cpu()
        # )
        # binary_jaccard_scores.append(float(cell_jaccard))
        # pq values
        
        remapped_instance_pred = remap_label(pred_binary_map)
        remapped_gt = remap_label(target_binary_map)
        # print(remapped_instance_pred.shape, remapped_gt.shape)
        [dq, sq, pq], _, precision, recall, f1 = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
        pq_scores.append(pq)
        dq_scores.append(dq)
        sq_scores.append(sq)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # aji = get_fast_aji(true=remapped_gt, pred=remapped_instance_pred)
        # aji_scores.append(aji)

        batch_metrics = {
            "binary_jaccard_scores": binary_jaccard_scores,
            "pq_scores": pq_scores,
            "dq_scores": dq_scores,
            "sq_scores": sq_scores,
            "aji_scores": aji_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "f1_scores": f1_scores,
        }


        return batch_metrics, pred_binary_map
    
    def calculate_instance_map(
        self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Calculate Instance Map from network predictions (after Softmax output)

        Args:
            predictions (dict): Dictionary with the following required keys:
                * nuclei_binary_map: Binary Nucleus Predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Type prediction of nuclei. Shape: (B, self.num_nuclei_classes, H, W)
                * hv_map: Horizontal-Vertical nuclei mapping. Shape: (B, 2, H, W)
            magnification (Literal[20, 40], optional): Which magnification the data has. Defaults to 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * torch.Tensor: Instance map. Each Instance has own integer. Shape: (B, H, W)
                * List of dictionaries. Each List entry is one image. Each dict contains another dict for each detected nucleus.
                    For each nucleus, the following information are returned: "bbox", "centroid", "contour", "type_prob", "type"
        """
        # reshape to B, H, W, C
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(
            0, 2, 3, 1
        )
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            predictions_["nuclei_type_map"][i][predictions_["nuclei_type_map"][i] >= 0.5] = 1
            predictions_["nuclei_type_map"][i][predictions_["nuclei_type_map"][i] < 0.5] = 0
            predictions_["nuclei_binary_map"][i][predictions_["nuclei_binary_map"][i] >= 0.5] = 1
            predictions_["nuclei_binary_map"][i][predictions_["nuclei_binary_map"][i] < 0.5] = 0
            pred_map = np.concatenate(
                [
                    predictions_["nuclei_type_map"][i]
                    .detach()
                    .cpu(),
                    predictions_["nuclei_binary_map"][i]
                    .detach()
                    .cpu(),
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(
        self, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert instance map (binary) to nuclei type instance map

        Args:
            instance_maps (torch.Tensor): Binary instance map, each instance has own integer. Shape: (B, H, W)
            type_preds (List[dict]): List (len=B) of dictionary with instance type information (compare post_process_hovernet function for more details)

        Returns:
            torch.Tensor: Nuclei type instance map. Shape: (B, self.num_nuclei_classes, H, W)
        """

        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros(
            (batch_size, h, w, self.num_nuclei_classes)
        )
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][
                    instance_map == nuclei
                ] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)
    
    def plot_results(
        self,
        img: torch.Tensor,
        predictions: dict,
        ground_truth: dict,
        img_name: str,
        outdir: str,
        scores: List[float],
    ) -> None:
        """Plot MoNuSeg results

        Args:
            img (torch.Tensor): Image as torch.Tensor, with Shape (1, 3, 1024, 1024) or (1, 3, 512, 512)
            predictions (dict): Prediction dictionary. Necessary keys:
                * nuclei_binary_map: Shape (1, 2, 1024, 1024) or (1, 2, 512, 512)
                * instance_map: Shape (1, 1024, 1024) or (1, 512, 512)
                * instance_types: List[dict], but just one entry in list
            ground_truth (dict): Ground-Truth dictionary. Necessary keys:
                * nuclei_binary_map: (1, 1024, 1024) or or (1, 512, 512)
                * instance_map: (1, 1024, 1024) or or (1, 512, 512)
                * instance_types: List[dict], but just one entry in list
            img_name (str): Image name as string
            outdir (Path): Output directory for storing
            scores (List[float]): Scores as list [Dice, Jaccard, bPQ]
        """
        # print(predictions["nuclei_binary_map"].shape)
        # outdir = Path(outdir) / "plots"
        os.makedirs(outdir, exist_ok=True)
        # outdir.mkdir(exist_ok=True, parents=True)
        predictions["nuclei_binary_map"][predictions["nuclei_binary_map"] >= 0.5] = 1
        predictions["nuclei_binary_map"][predictions["nuclei_binary_map"] < 0.5] = 0
        predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
            0, 2, 3, 1
        )
        # print(torch.tensor(ground_truth["instance_types_nuclei"]).shape, torch.tensor(ground_truth["instance_map"])[0].shape)
        # print(torch.unique(ground_truth["instance_map"]))
        ground_truth["instance_types"] =  calculate_instances(
            torch.tensor(ground_truth["instance_types_nuclei"]), ground_truth["instance_map"][0]
        )
        ground_truth["instance_map"] = ground_truth["instance_map"][0]
        ground_truth["nuclei_binary_map"] = ground_truth["nuclei_binary_map"][0]
        # print(np.unique(ground_truth["instance_map"]))
        h = ground_truth["instance_map"].shape[1]
        w = ground_truth["instance_map"].shape[2]

        # process image and other maps
       
        sample_image = img.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        pred_sample_binary_map = (
            predictions["nuclei_binary_map"][:, :, :, 0].detach().cpu().numpy()
        )[0]
        pred_sample_instance_maps = (
            predictions["instance_map"].detach().cpu().numpy()[0]
        )

        gt_sample_binary_map = (
            ground_truth["nuclei_binary_map"].detach().cpu().numpy()[0]
        ).astype(np.float16)
        gt_sample_instance_map = ground_truth["instance_map"].detach().cpu().numpy()[0]

        binary_cmap = plt.get_cmap("Greys_r")
        instance_map = plt.get_cmap("viridis")

        sample_image = np.array(Image.open(f'/home/***/miccai2025/nuclei/datasets/puma/image_1024/{img_name}.png')).astype(np.uint8)
        post_tissue_image = np.array(Image.open(f'visual/puma/instances_6/{img_name}_tissue_post.png')).astype(np.uint8)
        gt_tissue_image = np.array(Image.open(f'visual/puma/instances_6/{img_name}_gt_tissue.png')).astype(np.uint8)
        # sample_image = io.imread(f'/home/***/miccai2025/nuclei/datasets/puma/image_1024/{img_name}.png')[:,:,:3]
        # sample_image = cv2.imread(f'/home/***/miccai2025/nuclei/datasets/puma/image_1024/{img_name}.png')
        # print(np.unique(sample_image))
        # invert the normalization of the sample images
        
        # mean = (123.675, 116.280, 103.530)
        # std = (58.395, 57.12, 57.375)

        # mean = (0.5, 0.5, 0.5)
        # std = (0.5, 0.5, 0.5)
        # inv_normalize = transforms.Normalize(
        #     mean=[-0.5 / mean[0], -0.5 / mean[1], -0.5 / mean[2]],
        #     std=[1 / std[0], 1 / std[1], 1 / std[2]],
        # )
        # inv_samples = inv_normalize(torch.tensor(sample_image).permute(0, 3, 1, 2))
        # sample_image = inv_samples.permute(0, 2, 3, 1).detach().cpu().numpy()[0]

        # print(h, w)
        # start overlaying on image
        placeholder = np.zeros((2 * h, 4 * w, 3))
        # orig image
        placeholder[:h, :w, :3] = sample_image
        placeholder[h : 2 * h, :w, :3] = sample_image
        # binary prediction
        placeholder[:h, w : 2 * w, :3] = rgba2rgb(
            binary_cmap(gt_sample_binary_map * 255)
        )
        placeholder[h : 2 * h, w : 2 * w, :3] = rgba2rgb(
            binary_cmap(pred_sample_binary_map * 255)
        )

        # instance_predictions
        placeholder[:h, 2 * w : 3 * w, :3] = rgba2rgb(
            instance_map(
                (gt_sample_instance_map - np.min(gt_sample_instance_map))
                / (
                    np.max(gt_sample_instance_map)
                    - np.min(gt_sample_instance_map + 1e-10)
                )
            )
        )

        placeholder[h : 2 * h, 2 * w : 3 * w, :3] = rgba2rgb(
            instance_map(
                (pred_sample_instance_maps - np.min(pred_sample_instance_maps))
                / (
                    np.max(pred_sample_instance_maps)
                    - np.min(pred_sample_instance_maps + 1e-10)
                )
            )
        )
        gt_contours_polygon = [
            v["contour"] for v in ground_truth["instance_types"][0].values()
        ]
        gt_contours_polygon = [
            list(zip(poly[:, 0], poly[:, 1])) for poly in gt_contours_polygon
        ]
        gt_contour_colors_polygon = ["#007FD6" for i in range(len(gt_contours_polygon))]

        # print(np.asarray(gt_contour_colors_polygon).shape)
        # contour_image = Image.fromarray((np.asarray(gt_contour_colors_polygon)).astype(np.uint8))
        # contour_image.save(f"{outdir}/boundary_{img_name}.png")

        gt_cell_image = Image.fromarray((sample_image).astype(np.uint8)).convert(
            "RGB"
        )
        gt_drawing = ImageDraw.Draw(gt_cell_image)
        add_patch = lambda poly, color: gt_drawing.polygon(poly, outline=color, width=4)
        [
            add_patch(poly, c)
            for poly, c in zip(gt_contours_polygon, gt_contour_colors_polygon)
        ]
        contour_image = np.asarray(gt_cell_image) / 255
        contour_image = Image.fromarray((contour_image * 255).astype(np.uint8))
        contour_image.save(f"{outdir}/gt_{img_name}.png")

        gt_fuse_image = Image.fromarray(
            (gt_tissue_image).astype(np.uint8)
        ).convert("RGB")
        gt_fuse_drawing = ImageDraw.Draw(gt_fuse_image)
        add_patch = lambda poly, color: gt_fuse_drawing.polygon(
            poly, outline=color, width=4
        )
        [
            add_patch(poly, c)
            for poly, c in zip(gt_contours_polygon, gt_contour_colors_polygon)
        ]
        # plotting
        gt_fuse = np.asarray(gt_fuse_image) / 255
        gt_fuse = Image.fromarray((gt_fuse * 255).astype(np.uint8))
        gt_fuse.save(f"{outdir}/fuse_gt_{img_name}.png")

        placeholder[:h, 3 * w : 4 * w, :3] = np.asarray(gt_cell_image) / 255
        # pred
        
        pred_contours_polygon = [
            v["contour"] for v in predictions["instance_types"][0].values()
        ]
        pred_contours_polygon = [
            list(zip(poly[:, 0], poly[:, 1])) for poly in pred_contours_polygon
        ]
        pred_contour_colors_polygon = [
            "#007FD6" for i in range(len(pred_contours_polygon))
        ]

        pred_fuse_image = Image.fromarray(
            (post_tissue_image).astype(np.uint8)
        ).convert("RGB")
        fuse_drawing = ImageDraw.Draw(pred_fuse_image)
        add_patch = lambda poly, color: fuse_drawing.polygon(
            poly, outline=color, width=4
        )
        [
            add_patch(poly, c)
            for poly, c in zip(pred_contours_polygon, pred_contour_colors_polygon)
        ]
        # plotting
        contour_fuse = np.asarray(pred_fuse_image) / 255
        contour_fuse = Image.fromarray((contour_fuse * 255).astype(np.uint8))
        contour_fuse.save(f"{outdir}/fuse_{img_name}.png")

        pred_cell_image = Image.fromarray(
            (sample_image).astype(np.uint8)
        ).convert("RGB")
        pred_drawing = ImageDraw.Draw(pred_cell_image)
        add_patch = lambda poly, color: pred_drawing.polygon(
            poly, outline=color, width=4
        )
        [
            add_patch(poly, c)
            for poly, c in zip(pred_contours_polygon, pred_contour_colors_polygon)
        ]

        pred_draw_image = np.asarray(pred_cell_image) / 255
        pred_draw_image = Image.fromarray((pred_draw_image * 255).astype(np.uint8))
        pred_draw_image.save(f"{outdir}/pred_nuc_{img_name}.png")

        placeholder[h : 2 * h, 3 * w : 4 * w, :3] = np.asarray(pred_cell_image) / 255
        # plotting
        test_image = Image.fromarray((placeholder * 255).astype(np.uint8))
        test_image.save(f"{outdir}/raw_{img_name}.png")

        fig, axs = plt.subplots(figsize=(3, 2), dpi=1200)
        
        axs.imshow((placeholder * 255).astype(np.uint8))
        axs.set_xticks(np.arange(w / 2, 4 * w, w))

        axs.set_xticklabels(
            [
                "Image",
                "Binary-Cells",
                "Instances",
                "Countours",
            ],
            fontsize=6,
        )
        axs.xaxis.tick_top()

        axs.set_yticks(np.arange(h / 2, 2 * h, h))
        axs.set_yticklabels(["GT", "Pred."], fontsize=6)
        axs.tick_params(axis="both", which="both", length=0)
        grid_x = np.arange(w, 3 * w, w)
        grid_y = np.arange(h, 2 * h, h)

        for x_seg in grid_x:
            axs.axvline(x_seg, color="black")
        for y_seg in grid_y:
            axs.axhline(y_seg, color="black")
     
        if scores is not None:
            axs.text(
                20,
                1.85 * h,
                f"Dice: {str(np.round(scores[0], 2))}\nAJI: {str(np.round(scores[1], 2))}\nbPQ: {str(np.round(scores[2], 2))}",
                bbox={"facecolor": "white", "pad": 2, "alpha": 0.5},
                fontsize=4,
            )
        fig.suptitle(f"Patch Predictions for {img_name}", fontsize=6)
        fig.tight_layout()
        fig.savefig(f"{outdir}/pred_{img_name}.png")
        plt.close()


    def forward(self, x, prob_ins=None, prob_sem=None, img_id=None):

        batch_size = x.shape[0]

        backbone_out = self.model.forward_image(x)

        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        # vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]} # image encoder end


        ######################################

        num_images = len(self._features["image_embed"])

        outputs_mask_ins = []
        outputs_mask_sem = []
        normalize_coords = True
        for img_idx in range(num_images):

            # Transform input prompts
            point_coords = None
            point_labels = None
            box = None
            mask_input = None
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )

            if (prob_sem != None) and (prob_ins != None):
                masks_ins, masks_sem = self._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    prob_sem[img_idx].unsqueeze(0),
                    prob_ins[img_idx].unsqueeze(0),
                    img_idx=img_idx,
                )
            else:
                masks_ins, masks_sem = self._predict(
                    unnorm_coords,
                    labels,
                    unnorm_box,
                    prob_sem,
                    prob_ins,
                    img_idx=img_idx,
                )

            outputs_mask_ins.append(masks_ins.squeeze(0))
            outputs_mask_sem.append(masks_sem.squeeze(0))

        return torch.stack(outputs_mask_ins, dim=0), torch.stack(outputs_mask_sem, dim=0)
        



class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                # print(lateral_features.shape, top_down_features.shape)
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


    



        
    

