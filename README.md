### This is a README dedicated to running the experiments for the paper "Efficient Algorithms for Logistic Slate Contextual Bandits with Bandit Feedback"

To generate theta_star (the optimal parameter), use the following command:
```python3 generate_theta_star.py --length [] --seed []```


To run the files, use the following command:

`python3 main.py --alg_name [] --warmup --reward_type []--num_contexts [] --seed [] --theta_star [] --normalize_theta_star --horizon []--failure_level [] --arm_dim [] --slot_count [] --item_count []`