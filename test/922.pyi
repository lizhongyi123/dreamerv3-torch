

python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk

Monitor results:

tensorboard --logdir ./logdir

Namespace(act='SiLU', action_repeat=2, actor={'layers': 2, 'dist': 'normal', 'entropy': 0.0003, 'unimix_ratio': 0.01, 'std': 'learned', 'min_std': 0.1, 'max_std': 1.0,
'temp': 0.1, 'lr': 3e-05, 'eps': 1e-05, 'grad_clip': 100.0, 'outscale': 1.0}, batch_length=64, batch_size=16, compile=True,
cont_head={'layers': 2, 'loss_scale': 1.0, 'outscale': 1.0}, critic={'layers': 2, 'dist': 'symlog_disc', 'slow_target': True, 'slow_target_update': 1,
'slow_target_fraction': 0.02, 'lr': 3e-05, 'eps': 1e-05, 'grad_clip': 100.0, 'outscale': 0.0},
 dataset_size=1000000, debug=False, decoder={'mlp_keys': '$^', 'cnn_keys': 'image',
 'act': 'SiLU', 'norm': True, 'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5,
'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 'vector_dist': 'symlog_mse', 'outscale': 1.0},
 deterministic_run=False, device='cuda:0', disag_action_cond=False, disag_layers=4, disag_log=True,
disag_models=10, disag_offset=1, disag_target='stoch', disag_units=400, discount=0.997,
discount_lambda=0.95, dyn_deter=512, dyn_discrete=32, dyn_hidden=512, dyn_mean_act='none',
dyn_min_std=0.1, dyn_rec_depth=1, dyn_scale=0.5, dyn_std_act='sigmoid2', dyn_stoch=32,
encoder={'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 'cnn_depth': 32,
 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 'mlp_units': 1024, 'symlog_inputs': True},
 envs=4, eval_episode_num=10, eval_every=10000.0, eval_state_mean=False, evaldir=None, expl_behavior='greedy',
 expl_extr_scale=0.0, expl_intr_scale=1.0, expl_until=0, grad_clip=1000, grad_heads=('decoder', 'reward', 'cont'),
grayscale=False, imag_gradient='dynamics', imag_gradient_mix=0.0, imag_horizon=15, initial='learned', kl_free=1.0,
log_every=10000.0, logdir='./logdir/dmc_walker_walk', model_lr=0.0001, norm=True, offline_evaldir='', offline_traindir='',
opt='adam', opt_eps=1e-08, parallel=False, precision=32, prefill=2500, pretrain=100, rep_scale=0.1, reset_every=0, reward_EMA=True,
reward_head={'layers': 2, 'dist': 'symlog_disc', 'loss_scale': 1.0, 'outscale': 0.0}, seed=0, size=(64, 64), steps=1000000.0,
task='dmc_walker_walk', time_limit=1000, train_ratio=512, traindir=None, unimix_ratio=0.01, units=512, video_pred_log=True, weight_decay=0.0)

