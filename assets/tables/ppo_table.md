| Run | Settings changed or used | Test return | Mastery | Dropout | Sessions |
|---|---|---|---|---|---|
| ppo_01 | lr=0.0003; gamma=0.99; lmbda=0.95; entropy=0.01; critic=0.5; clip=0.2; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 18.12 | 0.99 | 0.00 | 40.6 |
| ppo_02 | lr=0.0001; gamma=0.99; lmbda=0.95; entropy=0.01; critic=0.5; clip=0.2; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 2.4 | 0.08 | 0.06 | 57.1 |
| ppo_03 | lr=0.001; gamma=0.99; lmbda=0.95; entropy=0.01; critic=0.5; clip=0.2; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 18.81 | 0.99 | 0.00 | 37.6 |
| ppo_04 | lr=0.0003; gamma=0.95; lmbda=0.95; entropy=0.01; critic=0.5; clip=0.2; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 13.54 | 0.78 | 0.00 | 51.2 |
| ppo_05 | lr=0.0003; gamma=0.999; lmbda=0.95; entropy=0.01; critic=0.5; clip=0.2; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 18.1 | 0.98 | 0.00 | 40.4 |
| ppo_06 | lr=0.0003; gamma=0.99; lmbda=0.95; entropy=0.0; critic=0.5; clip=0.2; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 16.6 | 0.92 | 0.01 | 43.9 |
| ppo_07 | lr=0.0003; gamma=0.99; lmbda=0.95; entropy=0.05; critic=0.5; clip=0.2; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 17.89 | 0.94 | 0.00 | 39.7 |
| ppo_08 | lr=0.0003; gamma=0.99; lmbda=0.95; entropy=0.01; critic=0.5; clip=0.1; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 17.37 | 0.93 | 0.00 | 41.8 |
| ppo_09 | lr=0.0003; gamma=0.99; lmbda=0.95; entropy=0.01; critic=0.5; clip=0.3; rollout=1000; epochs=8; minibatch=100; total_frames=100000 | 17.59 | 0.96 | 0.00 | 42.3 |
| ppo_10 | lr=0.0003; gamma=0.99; lmbda=0.95; entropy=0.01; critic=0.5; clip=0.2; rollout=400; epochs=4; minibatch=100; total_frames=100000 | 6.4 | 0.27 | 0.00 | 58.4 |
