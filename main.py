import os
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
import logging

from src.utils.helpers import set_seed, setup_logging, save_config, count_parameters

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """主训练函数"""
    try:
        # 设置随机种子
        set_seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

        # 打印配置
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # 保存配置
        save_config(cfg, ".")

        # 初始化数据模块
        logger.info("Initializing data module...")
        datamodule = hydra.utils.instantiate(cfg.data)

        # 设置数据以获取类别数量
        datamodule.setup("fit")

        # 更新模型配置
        if hasattr(datamodule, 'num_classes'):
            cfg.model.num_classes = datamodule.num_classes
            cfg.model.student.vit.num_classes = datamodule.num_classes
            logger.info(f"Number of classes: {datamodule.num_classes}")

        # 初始化模型
        logger.info("Initializing model...")
        model = hydra.utils.instantiate(cfg.model)

        # 打印模型信息
        teacher_params, teacher_trainable = count_parameters(model.teacher)
        student_params, student_trainable = count_parameters(model.student)

        logger.info(f"Teacher model parameters: {teacher_params:,} (trainable: {teacher_trainable:,})")
        logger.info(f"Student model parameters: {student_params:,} (trainable: {student_trainable:,})")
        logger.info(f"Compression ratio: {teacher_params / student_params:.2f}x")

        # 初始化日志记录器
        logger.info("Initializing logger...")
        exp_logger = hydra.utils.instantiate(cfg.logger)

        if isinstance(exp_logger, WandbLogger):
            # 设置实验名称
            if cfg.experiment_name:
                exp_logger._name = cfg.experiment_name

            # 更新wandb配置
            exp_logger.experiment.config.update({
                **OmegaConf.to_container(cfg, resolve=True),
                "teacher_params": teacher_params,
                "student_params": student_params,
                "compression_ratio": teacher_params / student_params
            })

        # 初始化训练器
        logger.info("Initializing trainer...")
        trainer = hydra.utils.instantiate(cfg.trainer, logger=exp_logger)

        # 开始训练
        logger.info("Starting training...")
        trainer.fit(model, datamodule)

        # 测试
        logger.info("Starting testing...")
        test_result = trainer.test(model, datamodule, ckpt_path="best")

        # 输出最终结果
        if test_result:
            test_acc = test_result[0].get('test/accuracy', 0.0)
            logger.info(f"Final test accuracy: {test_acc:.4f}")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()