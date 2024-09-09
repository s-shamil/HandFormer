import pytorch3d.transforms
import torch
import torch.nn.functional as F

FRAME_SMOOTHING_WINDOW = 30

def compute_windowed_mean_wrist_xf(
    wrist_xf: torch.Tensor, frame_smoothing_window: int = FRAME_SMOOTHING_WINDOW
) -> torch.Tensor:

    wrist_rot = pytorch3d.transforms.matrix_to_quaternion(wrist_xf[..., :3, :3])
    wrist_rot = torch.cat(
        [
            wrist_rot[:, 0].unsqueeze(1).repeat(1, frame_smoothing_window - 1, 1, 1),
            wrist_rot,
        ],
        dim=1,
    )
    windowed_wrist_rot = wrist_rot.unfold(1, frame_smoothing_window, 1)
    windowed_wrist_rot_selfprod = (
        windowed_wrist_rot @ windowed_wrist_rot.transpose(-2, -1)
    ) / frame_smoothing_window
    windowed_mean_wrist_rot = pytorch3d.transforms.quaternion_to_matrix(
        windowed_wrist_rot_selfprod.svd().U[..., 0]
    )

    windowed_mean_wrist_xf = torch.zeros(
        (
            windowed_mean_wrist_rot.shape[0],
            windowed_mean_wrist_rot.shape[1],
            windowed_mean_wrist_rot.shape[2],
            4,
            4,
        ),
        device=windowed_mean_wrist_rot.device,
    )
    windowed_mean_wrist_xf[..., :3, :3] = windowed_mean_wrist_rot
    windowed_mean_wrist_xf[..., -1, -1] = 1

    wrist_translation = wrist_xf[:, :, :, :3, 3]
    wrist_translation = torch.cat(
        [
            wrist_translation[:, 0]
            .unsqueeze(1)
            .repeat(1, frame_smoothing_window - 1, 1, 1),
            wrist_translation,
        ],
        dim=1,
    )
    windowed_wrist_translation = wrist_translation.unfold(1, frame_smoothing_window, 1)
    windowed_mean_wrist_translation = windowed_wrist_translation.mean(-1)
    windowed_mean_wrist_xf[..., :3, 3] = windowed_mean_wrist_translation

    return windowed_mean_wrist_xf

def inverse_xf(xf: torch.Tensor) -> torch.Tensor:
    inv_xf = xf.clone()
    inv_xf[..., :3, :3] = xf[..., :3, :3].transpose(-2, -1)
    inv_xf[..., :3, 3] = (-inv_xf[..., :3, :3] @ xf[..., :3, 3].unsqueeze(-1)).squeeze(
        -1
    )
    return inv_xf

def get_xf_transformed_data(landmarks, xf_transforms):
    B, T, N, H, _ = landmarks.shape
    reference = "WindowedLeftWrist"

    # Overwrite xf matrices for both hands from the selected one, if defined by reference. 
    if reference == "LeftWrist":
        xf_transforms = xf_transforms[:, :, 0, ...].unsqueeze(2).repeat(1, 1, N, 1, 1)
    elif reference == "RightWrist":
        xf_transforms = xf_transforms[:, :, 1, ...].unsqueeze(2).repeat(1, 1, N, 1, 1)
    elif reference == "WindowedLeftWrist":
        xf_transforms = xf_transforms[:, :, 0, ...].unsqueeze(2).repeat(1, 1, N, 1, 1)
        xf_transforms = compute_windowed_mean_wrist_xf(xf_transforms)
    elif reference == "WindowedRightWrist":
        xf_transforms = xf_transforms[:, :, 1, ...].unsqueeze(2).repeat(1, 1, N, 1, 1)
        xf_transforms = compute_windowed_mean_wrist_xf(xf_transforms)
    elif reference == "WindowedWrist":
        xf_transforms = compute_windowed_mean_wrist_xf(xf_transforms)

    # Flattens, pads 1 after x,y,z for each point to prepare for matrix multiplication
    # Makes the shape from (..., 4) to (..., 4,1) with unsqueeze
    x_flat = F.pad(landmarks.reshape(-1, 3), (0, 1), value=1.0).unsqueeze(-1)    
    ref_to_world_xf = inverse_xf(xf_transforms)
    ref_to_world_xf = ref_to_world_xf.unsqueeze(3).repeat(1, 1, 1, H, 1, 1) # repeat the pair for all 21 joints
    # flatten ref_to_world_xf
    ref_transf_flat = ref_to_world_xf.view(-1, 4, 4)
    # apply transformation to landmarks
    x_ref = ref_transf_flat @ x_flat
    x_ref = x_ref[..., :3, 0].view(B, T, N, H, 3)

    transformed_landmarks = x_ref.permute(0, 4, 1, 3, 2) # 1,C,T,V,M shape at the end. V=H (#vertices), M=N (#entities)
    return transformed_landmarks
