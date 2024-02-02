#!/usr/bin/bash
denoiser test \
--config=logs/5/fit/version_0/config.yaml \
--ckpt_path=logs/5/fit/version_0/checkpoints/epoch=9-step=16889.ckpt \
--trainer.logger.name=5/test/6 \
--data.subset=6

denoiser test \
--config=logs/5/fit/version_0/config.yaml \
--ckpt_path=logs/5/fit/version_0/checkpoints/epoch=9-step=16889.ckpt \
--trainer.logger.name=5/test/5 \
--data.subset=5

denoiser test \
--config=logs/5/fit/version_0/config.yaml \
--ckpt_path=logs/5/fit/version_0/checkpoints/epoch=9-step=16889.ckpt \
--trainer.logger.name=5/test/4 \
--data.subset=4

denoiser test \
--config=logs/4/fit/version_0/config.yaml \
--ckpt_path=logs/4/fit/version_0/checkpoints/epoch=9-step=18554.ckpt \
--trainer.logger.name=4/test/6 \
--data.subset=6

denoiser test \
--config=logs/4/fit/version_0/config.yaml \
--ckpt_path=logs/4/fit/version_0/checkpoints/epoch=9-step=18554.ckpt \
--trainer.logger.name=4/test/5 \
--data.subset=5

denoiser test \
--config=logs/4/fit/version_0/config.yaml \
--ckpt_path=logs/4/fit/version_0/checkpoints/epoch=9-step=18554.ckpt \
--trainer.logger.name=4/test/4 \
--data.subset=4

denoiser test \
--config=logs/3/fit/version_0/config.yaml \
--ckpt_path=logs/3/fit/version_0/checkpoints/epoch=8-step=15773.ckpt \
--trainer.logger.name=3/test/6 \
--data.subset=6

denoiser test \
--config=logs/3/fit/version_0/config.yaml \
--ckpt_path=logs/3/fit/version_0/checkpoints/epoch=8-step=15773.ckpt \
--trainer.logger.name=3/test/5 \
--data.subset=5

denoiser test \
--config=logs/3/fit/version_0/config.yaml \
--ckpt_path=logs/3/fit/version_0/checkpoints/epoch=8-step=15773.ckpt \
--trainer.logger.name=3/test/4 \
--data.subset=4

denoiser test \
--config=logs/2/fit/version_0/config.yaml \
--ckpt_path=logs/2/fit/version_0/checkpoints/epoch=9-step=18184.ckpt \
--trainer.logger.name=2/test/6 \
--data.subset=6

denoiser test \
--config=logs/2/fit/version_0/config.yaml \
--ckpt_path=logs/2/fit/version_0/checkpoints/epoch=9-step=18184.ckpt \
--trainer.logger.name=2/test/5 \
--data.subset=5

denoiser test \
--config=logs/2/fit/version_0/config.yaml \
--ckpt_path=logs/2/fit/version_0/checkpoints/epoch=9-step=18184.ckpt \
--trainer.logger.name=2/test/4 \
--data.subset=4

denoiser test \
--config=logs/1/fit/version_0/config.yaml \
--ckpt_path=logs/1/fit/version_0/checkpoints/epoch=8-step=16328.ckpt \
--trainer.logger.name=1/test/6 \
--data.subset=6

denoiser test \
--config=logs/1/fit/version_0/config.yaml \
--ckpt_path=logs/1/fit/version_0/checkpoints/epoch=8-step=16328.ckpt \
--trainer.logger.name=1/test/5 \
--data.subset=5

denoiser test \
--config=logs/1/fit/version_0/config.yaml \
--ckpt_path=logs/1/fit/version_0/checkpoints/epoch=8-step=16328.ckpt \
--trainer.logger.name=1/test/4 \
--data.subset=4

denoiser test \
--config=logs/0/fit/version_0/config.yaml \
--ckpt_path=logs/0/fit/version_0/checkpoints/epoch=8-step=16143.ckpt \
--trainer.logger.name=0/test/6 \
--data.subset=6

denoiser test \
--config=logs/0/fit/version_0/config.yaml \
--ckpt_path=logs/0/fit/version_0/checkpoints/epoch=8-step=16143.ckpt \
--trainer.logger.name=0/test/5 \
--data.subset=5

denoiser test \
--config=logs/0/fit/version_0/config.yaml \
--ckpt_path=logs/0/fit/version_0/checkpoints/epoch=8-step=16143.ckpt \
--trainer.logger.name=0/test/4 \
--data.subset=4
