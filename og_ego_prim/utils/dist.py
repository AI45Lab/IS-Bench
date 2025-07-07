import os
import socket
import time

import torch
import torch.distributed as dist


def find_free_port():
    s = socket.socket()
    s.bind(('', 0))
    return s.getsockname()[1]


def init_distributed_mode():
    if 'SLURM_PROCID' in os.environ:
        global_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = global_rank % torch.cuda.device_count()

        job_id = os.environ['SLURM_JOBID']
        host_file = 'dist_url.' + job_id + '.txt'

        if global_rank == 0:
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            dist_url = 'tcp://{}:{}'.format(ip, port)
            with open(host_file, 'w') as f:
                f.write(dist_url)
        else:
            while not os.path.exists(host_file):
                time.sleep(1)
            with open(host_file, 'r') as f:
                dist_url = f.read()
    else:
        print('Not using distributed mode')
        return False

    torch.cuda.set_device(local_rank)
    print(f'| distributed init (rank {global_rank}): {dist_url}')
    dist.init_process_group(
        backend='nccl', 
        init_method=dist_url,
        world_size=world_size, 
        rank=global_rank
    )
    dist.barrier()

    if global_rank == 0:
        for host_file in os.listdir('.'):
            if host_file.startswith('dist_url.'):
                os.remove(host_file)


def get_dist_info():
    return dist.get_rank(), dist.get_world_size()
