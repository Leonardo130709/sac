import argparse
from sac import SAC, config
from tensorboard import program


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='cartpole_balance')
    parser.add_argument('--encoder', default='MLP')
    parser.add_argument('--logdir', default='logdir')
    args = parser.parse_args()
    return args

def update_configs(configs, args):
    for k,v in vars(args).items():
        setattr(configs, k, v)
    return configs

args = parse_args()
config = update_configs(config, args)
tb = program.TensorBoard()
tb.configure(argv=[None, f'--logdir={args.logdir}', '--port=6006'])
tb.launch()

sac = SAC(config)
sac.learn()

