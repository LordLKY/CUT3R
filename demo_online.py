#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--save_ply",
        type=int,
        default=0,
        help="Whether to save the ply file or not",
    )

    return parser.parse_args()


def prepare_online_input(
    img, idx
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    image = img
    # Only images are provided.
    view = {
        "img": image["img"],
        "ray_map": torch.full(
            (
                image["img"].shape[0],
                6,
                image["img"].shape[-2],
                image["img"].shape[-1],
            ),
            torch.nan,
        ),
        "true_shape": torch.from_numpy(image["true_shape"]),
        "idx": idx,
        "instance": str(idx),
        "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
            0
        ),
        "img_mask": torch.tensor(True).unsqueeze(0),
        "ray_mask": torch.tensor(False).unsqueeze(0),
        "update": torch.tensor(True).unsqueeze(0),
        "reset": torch.tensor(False).unsqueeze(0),
    }

    return [view]


def prepare_online_output(output, outdir, f_id, use_pose=True, save_ply=False):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf

    pts3ds_self = output["pred"]["pts3d_in_self_view"].cpu()
    pts3ds_other = output["pred"]["pts3d_in_other_view"].cpu()
    conf_self = output["pred"]["conf_self"].cpu()
    conf_other = output["pred"]["conf"].cpu()

    pr_pose = pose_encoding_to_camera(output["pred"]["camera_pose"].clone()).cpu()
    R_c2w = pr_pose[:, :3, :3]
    t_c2w = pr_pose[:, :3, 3]

    if use_pose:
        pts3ds_other = geotrf(pr_pose, pts3ds_self)
        conf_other = conf_self
    
     # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = 0.5 * (output["view"]["img"].permute(0, 2, 3, 1) + 1.0)

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    # B = 1
    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = pts3ds_other  # B, H, W, 3
    conf_self_tosave = conf_self  # B, H, W
    conf_other_tosave = conf_other  # B, H, W
    colors_tosave = 0.5 * (output["view"]["img"].permute(0, 2, 3, 1).cpu() + 1.0)  # [B, H, W, 3]
    cam2world_tosave = pr_pose  # B, 4, 4
    intrinsics_tosave = torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    outdir = os.path.join(outdir, f"{f_id:06d}")
    os.makedirs(outdir, exist_ok=True)

    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    depth = depths_tosave[0].cpu().numpy()
    conf = conf_self_tosave[0].cpu().numpy()
    color = colors_tosave[0].cpu().numpy()
    c2w = cam2world_tosave[0].cpu().numpy()
    intrins = intrinsics_tosave[0].cpu().numpy()
    np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
    np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
    iio.imwrite(
        os.path.join(outdir, "color", f"{f_id:06d}.png"),
        (color * 255).astype(np.uint8),
    )
    np.savez(
        os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
        pose=c2w,
        intrinsics=intrins,
    )
    
    if save_ply:
        from plyfile import PlyData, PlyElement
        ply_path = os.path.join(outdir, "points3D.ply")

        def storePly(path, xyz, rgb):
            # Define the dtype for the structured array
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            
            normals = np.zeros_like(xyz)

            elements = np.empty(xyz.shape[0], dtype=dtype)
            attributes = np.concatenate((xyz, normals, rgb), axis=1)
            elements[:] = list(map(tuple, attributes))

            # Create the PlyData object and write to file
            vertex_element = PlyElement.describe(elements, 'vertex')
            ply_data = PlyData([vertex_element])
            ply_data.write(path)
        
        colors_tosave_u1 = torch.clamp(colors_tosave * 255, 0, 255).to(torch.uint8)
        storePly(ply_path, pts3ds_other_tosave.reshape(-1, 3).cpu().numpy(), colors_tosave_u1.reshape(-1, 3).cpu().numpy())


def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_online_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_online
    from src.dust3r.model import ARCroco3DStereo

    from src.dust3r.utils.image import load_images

    # Prepare image file paths.
    img_paths, _ = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    # img_mask = [True] * len(img_paths)

    images = load_images(img_paths, size=args.size)

    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    # online inference
    for f_id in range(len(img_paths)):
        view = prepare_online_input(images[f_id], f_id)
        print(f">> Running inference for frame {f_id}...")
        start_time = time.perf_counter()
        outputs = inference_online(view, model, device, verbose=False, init_iter=(f_id == 0))
        total_time = time.perf_counter() - start_time
        print(f">> Inference completed in {total_time:.2f} seconds.")
        prepare_online_output(outputs, args.output_dir, f_id, True, (args.save_ply != 0))


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_online_inference(args)


if __name__ == "__main__":
    main()

# How to run the demo:
# python demo_online.py --size 512 --seq_path local_examples/005_online --vis_threshold 1.5 --output_dir output/demo_005_online --save_ply 1
    
# use gs
# python gs/train.py --eval --iterations 200 -s output/demo_003
# python gs/render.py -m output/demo_003/model --iteration 200
# python gs/metrics.py -m output/demo_003/model