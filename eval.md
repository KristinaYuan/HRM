OMP_NUM_THREADS=8 torchrun --nproc-per-node 1 evaluate.py \
  checkpoint=/data/yuanjiayi/HRM/checkpoints/maze-single/step_145831 \
  'save_outputs=[]'

step_20833
{'all': {'accuracy': np.float32(0.9639512), 'exact_accuracy': np.float32(0.143), 'lm_loss': np.float32(0.11452479), 'q_halt_accuracy': np.float32(0.802), 'q_halt_loss': np.float32(0.41494024), 'steps': np.float32(16.0)}}

step_41666
{'all': {'accuracy': np.float32(0.97620666), 'exact_accuracy': np.float32(0.155), 'lm_loss': np.float32(0.07049543), 'q_halt_accuracy': np.float32(0.908), 'q_halt_loss': np.float32(0.2801358), 'steps': np.float32(16.0)}}

step_62499
{'all': {'accuracy': np.float32(0.98436886), 'exact_accuracy': np.float32(0.28), 'lm_loss': np.float32(0.062407), 'q_halt_accuracy': np.float32(0.777), 'q_halt_loss': np.float32(0.6862378), 'steps': np.float32(16.0)}}

step_83332
{'all': {'accuracy': np.float32(0.98713994), 'exact_accuracy': np.float32(0.405), 'lm_loss': np.float32(0.051501498), 'q_halt_accuracy': np.float32(0.866), 'q_halt_loss': np.float32(0.60384524), 'steps': np.float32(16.0)}}

step_104165
{'all': {'accuracy': np.float32(0.9898078), 'exact_accuracy': np.float32(0.66), 'lm_loss': np.float32(0.043160457), 'q_halt_accuracy': np.float32(0.769), 'q_halt_loss': np.float32(0.90710956), 'steps': np.float32(16.0)}}

step_124998
{'all': {'accuracy': np.float32(0.9851934), 'exact_accuracy': np.float32(0.574), 'lm_loss': np.float32(0.061722193), 'q_halt_accuracy': np.float32(0.817), 'q_halt_loss': np.float32(0.6826471), 'steps': np.float32(16.0)}}

step_145831
{'all': {'accuracy': np.float32(0.99101555), 'exact_accuracy': np.float32(0.725), 'lm_loss': np.float32(0.050011426), 'q_halt_accuracy': np.float32(0.807), 'q_halt_loss': np.float32(0.9854983), 'steps': np.float32(16.0)}}

step_166664
{'all': {'accuracy': np.float32(0.9762755), 'exact_accuracy': np.float32(0.028), 'lm_loss': np.float32(0.103314705), 'q_halt_accuracy': np.float32(0.546), 'q_halt_loss': np.float32(0.73082274), 'steps': np.float32(16.0)}}

step_187497
{'all': {'accuracy': np.float32(0.98847437), 'exact_accuracy': np.float32(0.668), 'lm_loss': np.float32(0.06312762), 'q_halt_accuracy': np.float32(0.741), 'q_halt_loss': np.float32(1.2119917), 'steps': np.float32(16.0)}}

step_208330
{'all': {'accuracy': np.float32(0.98638105), 'exact_accuracy': np.float32(0.386), 'lm_loss': np.float32(0.06666817), 'q_halt_accuracy': np.float32(0.663), 'q_halt_loss': np.float32(1.0210209), 'steps': np.float32(16.0)}}

Best checkpoint: step_145831
Best exact accuracy: 0.725
