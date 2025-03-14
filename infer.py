import argparse
import functools
import numpy as np

from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments


def wav_classify(predict_dir):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs', str, 'configs/cam++.yml', '配置文件')
    add_arg('use_gpu', bool, True, '是否使用GPU预测')
    add_arg('audio_path', str, predict_dir, '音频路径')
    add_arg('model_path', str, 'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
    args = parser.parse_args()
    # 获取识别器
    predictor = MAClsPredictor(configs=args.configs,
                               model_path=args.model_path,
                               use_gpu=args.use_gpu)

    label, score = predictor.predict(audio_data=args.audio_path)
    print('============================================================')
    print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')
    print('============================================================')

if __name__ == '__main__':
    wav_classify(r'G:\dataset\zhenlie\output\2025-03-12_17-03-46.wav')