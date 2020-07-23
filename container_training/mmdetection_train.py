# # Check Pytorch installation
# import torch, torchvision
# print(f"Torch version: {torch.__version__}, is CUDA on? {torch.cuda.is_available()}")

# # Check MMDetection installation
# import mmdet
# print(f"MMDET version {mmdet.__version__}")

# # Check mmcv installation
# from mmcv.ops import get_compiling_cuda_version, get_compiler_version
# print(f"MMVC CUDA version {get_compiling_cuda_version()}")
# print(f"MMVC compiler version {get_compiler_version()} ")


from argparse import ArgumentParser
import os
from mmcv import Config



def get_training_world():

    """
    Calculates number of devices in Sagemaker distributed cluster
    """

    # Get params of Sagemaker distributed cluster from predefined env variables
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_cpus = int(os.environ["SM_NUM_CPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Define PyTorch training world
    world = {}
    world["number_of_processes"] = num_gpus if num_gpus > 0 else num_cpus
    world["number_of_machines"] = len(hosts)
    world["size"] = world["number_of_processes"] * world["number_of_machines"]
    world["machine_rank"] = hosts.index(current_host)
    world["master_addr"] = hosts[0]
    world["master_port"] = "55555" # port is defined by Sagemaker

    return world

def training_configurator(args):
    """
    Configure training process
    """
    
    # updating path to config file inside SM container
    print(f"Will use config from file {os.path.abspath(args.config_file)}")
    cfg = Config.fromfile(os.path.abspath(args.config_file))
    
    if args.dataset.lower() == "coco":
        # TODO: need to pass proper data folder via envrionmental variable
        cfg.data_root = os.path.join("/opt/ml/input/data/")
        cfg.data.train.ann_file = "annotations/instances_train2017.json"
        cfg.data.train.img_prefix = "train2017"
        cfg.data.val.ann_file = "annotations/instances_val2017.json"
        cfg.data.val.img_prefix = "val2017"
        cfg.data.test.ann_file = "annotations/instances_test2017.json"
        cfg.data.test.img_prefix = "test2017"
        
        updated_config = os.path.join(os.getcwd(), "updated_config.py")
        cfg.dump(updated_config)
        print("Following config will be used for training:
              {cfg.pretty_text}")
        
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented.\
                                    Currently only COCO-style datasets are available.")
              
    return updated_config


if __name__ == "__main__":
    
    # Get initial configuration to select appropriate HuggingFace task and its configuration
    print('Starting training...')
    parser = ArgumentParser()
    parser.add_argument('--config-file', type=str, default=None, metavar="FILE", 
                        help="Only default MMDetection configs are supported now. \
                        See for details: https://github.com/open-mmlab/mmdetection/tree/master/configs/")
    parser.add_argument('--dataset', type=str, default="coco", help="Define which dataset to use.")
    
    sm_args, mmdetection_args = parser.parse_known_args()

    # Get task script and its cofiguration
    config_file = training_configurator(sm_args)

    # Derive parameters of distributed training cluster in Sagemaker
    world = get_training_world()

    # Creates launch configuration according to PyTorch Distributed Launch utility requirements: 
    # https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    launch_config = ["--nnodes", str(world['number_of_machines']), "--node_rank", str(world['machine_rank']),
                     "--nproc_per_node", str(world['number_of_processes']), "--master_addr", world['master_addr'], 
                     "--master_port", world['master_port']]
    
    # TODO: implement running inline with this sample: https://github.com/vdabravolski/mxnet-distributed-sample/blob/master/container_training/hvd_launcher.py#L177-L198
#     % env MKL_THREADING_LAYER=GNU
#     ! python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 tools/train.py ./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py --launcher pytorch

