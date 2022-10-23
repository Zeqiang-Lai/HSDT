python -m hsirun.test -a hsdt.hsdt -r checkpoints/hsdt_m/gaussian/model_best.pth
python -m hsirun.test -a hsdt.hsdt_8 -r checkpoints/hsdt_s/gaussian/model_best.pth
python -m hsirun.test -a hsdt.hsdt_deep -r checkpoints/hsdt_l/gaussian/model_best.pth


python -m hsirun.test -a hsdt.hsdt_ffn -r checkpoints/ablation/sm_ffn/ffn/hsdt_ffn.pth
python -m hsirun.test -a hsdt.hsdt_gdfn -r checkpoints/ablation/sm_ffn/gdfn/model_best.pth
python -m hsirun.test -a hsdt.hsdt_smffn1 -r checkpoints/ablation/sm_ffn/sm_ffn1/model_best.pth

python -m hsirun.test -a hsdt.hsdt_conv3d -r checkpoints/ablation/s3conv/conv3d/hsdt_conv3d.pth
python -m hsirun.test -a hsdt.hsdt_s3conv_sep -r checkpoints/ablation/s3conv/s3conv_sep/model_best.pth
python -m hsirun.test -a hsdt.hsdt_s3conv_seq -r checkpoints/ablation/s3conv/s3conv_seq/model_best.pth
python -m hsirun.test -a hsdt.hsdt_s3conv1 -r checkpoints/ablation/s3conv/s3conv1/model_best.pth

python -m hsirun.test -a hsdt.baseline_gssa -r checkpoints/ablation/break_down/baseline_gssa/model_best.pth
python -m hsirun.test -a hsdt.baseline_ssa -r checkpoints/ablation/break_down/baseline_ssa/model_best.pth
python -m hsirun.test -a hsdt.baseline_conv3d -r checkpoints/ablation/break_down/baseline/model_best.pth
