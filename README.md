# ElastoPINN

## Setup

### Requirement

1. Only Ubuntu 22.04 is currently supported.
2. CUDA-available environment.
3. [PyTorch3D](https://github.com/facebookresearch/pytorch3d)-supported environment.

### Train & Test Environment

1. Clone this repository.
2. Run `setup_env.sh` file.
3. Download [PAC-NeRF](https://sites.google.com/view/PAC-NeRF) dataset and extract to `dataset` directory.
    ```tree
    ElastoPINN
        └ dataset
            └ pac-nerf
                ├ data
                │   ├ bird
                │   │   ├ all_data.json
                │   │   └ data
                │   │       ├ r_0_-1.png
                │   │       ├ r_0_0.png
                │   │       │   ⋮
                │   │       └ r_10_13.png
                │   ├ cat
                │   │   ├ all_data.json
                │   │   └ data
                │   │       ├ r_0_-1.png
                │   │       ├ r_0_0.png
                │   │       │   ⋮
                │   │       └ r_10_13.png
                │   │   ⋮
                │   ├ elastic
                │   │   ├ 0
                │   │   │   ├ all_data.json
                │   │   │   └ data
                │   │   │       ├ r_0_-1.png
                │   │   │       ├ r_0_0.png
                │   │   │       │   ⋮
                │   │   │       └ r_10_13.png
                │   │   ├ 1
                │   │   │   └ ⋯
                │   │   │   ⋮
                │   │   └ 9
                │   │   │   └ ⋯
                │   │   ⋮
                │   └ trophy
                │   │   ├ all_data.json
                │   │   └ data
                │   │       ├ r_0_-1.png
                │   │       ├ r_0_0.png
                │   │       │   ⋮
                │   │       └ r_10_13.png
                └ simulation_data
                    ├ bird
                    │   ├ 0.ply
                    │   ├ 1.ply
                    │   │   ⋮
                    │   └ 20.ply
                    ├ cat
                    │   ├ 0.ply
                    │   ├ 1.ply
                    │   │   ⋮
                    │   └ 20.ply
                    │   ⋮
                    ├ elastic
                    │   ├ 0
                    │   │   ├ 0.ply
                    │   │   ├ 1.ply
                    │   │   │   ⋮
                    │   │   └ 15.ply
                    │   ├ 1
                    │   │   └ ⋯
                    │   │   ⋮
                    │   ├ 9
                    │   │   └ ⋯
                    │   ⋮
                    └ trophy
                        ├ 0.ply
                        ├ 1.ply
                        │   ⋮
                        └ 20.ply
    ```
