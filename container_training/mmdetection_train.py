from argparse import ArgumentParser
import os
from mmcv import Config
import json
import subprocess


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
    abs_config_path = os.path.join("/opt/ml/code/mmdetection", args.config_file)
    print(f"Will use config from file {abs_config_path}")
    cfg = Config.fromfile(abs_config_path)
    
    if args.dataset.lower() == "coco":
        
        cfg.data_root = os.environ["SM_CHANNEL_TRAINING"] # By default, data will be download to /opt/ml/input/data/training
        cfg.data.train.ann_file = os.path.join(cfg.data_root, "annotations/instances_train2017.json")
        cfg.data.train.img_prefix = os.path.join(cfg.data_root, "train2017")
        cfg.data.val.ann_file = os.path.join(cfg.data_root, "annotations/instances_val2017.json")
        cfg.data.val.img_prefix = os.path.join(cfg.data_root, "val2017")
        cfg.data.test.ann_file = os.path.join(cfg.data_root, "annotations/instances_val2017.json")
        cfg.data.test.img_prefix = os.path.join(cfg.data_root, "val2017")
        
        # TODO: delete it
        print("DEBUG \n data root", os.listdir(cfg.data_root)) 
        
        updated_config = os.path.join(os.getcwd(), "updated_config.py")
        cfg.dump(updated_config)
        print(f"Following config will be used for training:{cfg.pretty_text}")
        
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
    
              
    # Train script config
    launch_config = [ "python -m torch.distributed.launch", 
                     "--nnodes", str(world['number_of_machines']), "--node_rank", str(world['machine_rank']),
                     "--nproc_per_node", str(world['number_of_processes']), "--master_addr", world['master_addr'], 
                     "--master_port", world['master_port']]
 
    train_config = [os.path.join(os.environ["MMDETECTION"], "tools/train.py"), 
                    config_file, 
                    "--launcher", "pytorch", 
                    # TODO: add ability to pass MMD arguments via https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py#L386-L415
                    #                     mmdetection_args if mmdetection_args!="" 
                    
                   ]
    
    # Concat MPI run configuration and training script and its parameters
    joint_cmd = " ".join(str(x) for x in launch_config+train_config)
    print("Following command will be executed: \n", joint_cmd)
    
    process = subprocess.Popen(joint_cmd,  stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)
    
    while True:
        output = process.stdout.readline()
        
        if process.poll() is not None:
            break
        if output:
            print(output.decode("utf-8").strip())
    rc = process.poll()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=joint_cmd)
    
    sys.exit(process.returncode)