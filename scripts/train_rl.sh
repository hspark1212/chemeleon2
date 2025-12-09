# DNG - MP-20
python src/train_rl.py experiment=mp_20/rl_dng_po
python src/train_rl.py experiment=mp_20/rl_dng
# ablation studies on reward components
python src/train_rl.py experiment=mp_20/rl_dng +rl_module.reward_fn.weight_r_structure_diversity=0.0 +rl_module.reward_fn.weight_r_composition_diversity=0.0
python src/train_rl.py experiment=mp_20/rl_dng +rl_module.reward_fn.weight_r_structure_diversity=1.0 +rl_module.reward_fn.weight_r_composition_diversity=1.0

# DNG - Alex-MP-20
python src/train_rl.py experiment=alex_mp_20/rl_dng

# Bandgap - MP-20
python src/train_rl.py experiment=mp_20/rl_bandgap

# Bandgap - Alex-MP-20
python src/train_rl.py experiment=alex_mp_20_bandgap/rl_bandgap

# Custom Reward Example
python src/train_rl.py experiment=rl_custom_reward