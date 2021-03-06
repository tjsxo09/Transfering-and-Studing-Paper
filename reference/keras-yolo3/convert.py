#! /usr/bin/env python
"""
Reads Darknet config and weights and creates Keras model with TF backend.

"""

import argparse
import configparser
import io
import os
import pydot
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, Input, ZeroPadding2D, Add,
                          UpSampling2D, MaxPooling2D, Concatenate)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot


parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-w',
    '--weights_only',
    help='Save as Keras weights file instead of model file.',
    action='store_true')

def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int) # 디폴트값이 int인 딕셔너리를 생성한다.
    output_stream = io.StringIO() # 문자열 데이터를 파일로 저장한 다음 처리함 다시 쓰이지 않을 때 유용
    with open(config_file) as fin: # yolov3.cfg에 의해서
        for line in fin: # config 파일을 한줄 씩 읽어들인다.
            if line.startswith('['): # [로 시작하는 줄이면
                section = line.strip().strip('[]') # 양쪽 끝에 있는 공백과 줄바꿈 문자를 지우고 []도 지운다.
                _section = section + '_' + str(section_counters[section]) # 키 section에 대한 값 을 반환
                section_counters[section] += 1
                line = line.replace(section, _section) # line을 section에서 _section으로 변경한다.
            output_stream.write(line) # 변경한 line을 out_stream에 작성한다.
    output_stream.seek(0)
    return output_stream #

# %%
def _main(args):
    config_path = os.path.expanduser(args.config_path) 
    # config 파일의 경로로 터미널에서 입력받은 config_path로  절대경로를 알아냄
    weights_path = os.path.expanduser(args.weights_path)
    # config 파일과 같은 방법으로 weight 파일의 절대 경로를 알아냄
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path) # .cfg로 끝나지 않는 경로라면 .cfg가 아니라는 오류 메세지를 보내고 종료
    assert weights_path.endswith(
        '.weights'), '{} is not a .weights file'.format(weights_path)
    # config 파일과 마찬가지

    output_path = os.path.expanduser(args.output_path)
    # 출력 경로에 대한 절대 경로를 터미널을 이용하여 받은 이름을 가지고서 생성
    assert output_path.endswith(
        '.h5'), 'output path {} is not a .h5 file'.format(output_path)
    #.h5라는 문자열이 아니면 오류를 발생시키고 종료 시킨다.
    output_root = os.path.splitext(output_path)[0] # 입력받은 경로를 확장자와 그 외로 나누어서
    # 그 외의 부분만(경로 부분만) output_root라는 변수에 입력한다.

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    # weights_path에서 파일을 열어서 바이너리 파일을 열어서 가지고 온다.
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12)) # 12 문자만 읽어 들인다.
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.') # 다크넷으로 된 설정 파일을 변경함
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file) #cfg_parser에 연결함

    print('Creating Keras model.')
    input_layer = Input(shape=(None, None, 3)) # 이미지 크기에 상관없이 3 채널이면 모두 받아들임
    prev_layer = input_layer
    all_layers = []

    # 네트워크 감소
    weight_decay = float(cfg_parser['net_0']['decay'] # net_0의 decay = 0.0005
                         ) if 'net_0' in cfg_parser.sections() else 5e-4 # net_0가 없으면 0.0005로 맞추게 해둔다.
    count = 0 # 계산량을 세준다.
    out_index = []
    for section in cfg_parser.sections(): # 모든 다크넷 config section에서
        print('Parsing section {}'.format(section)) # 섹션 이름 출력
        if section.startswith('convolutional'): # convolutional이라고 시작하면
            filters = int(cfg_parser[section]['filters']) # 필터의 수를 cfg_parser의 filters에서 가지고 온다.
            size = int(cfg_parser[section]['size']) # 필터의 크기를 filters와 마찬가지로 가지고 온다.
            stride = int(cfg_parser[section]['stride']) # 보폭을 설정 
            pad = int(cfg_parser[section]['pad']) # 패딩 크기를 설정
            activation = cfg_parser[section]['activation'] # 활성화 함수를 설정
            batch_normalize = 'batch_normalize' in cfg_parser[section] # 배치 정규화 항목이 있으면 1 없으면 0

            padding = 'same' if pad == 1 and stride == 1 else 'valid' 
            # 만약 패딩이 1이고 보폭이 1이라면 padding을 'same'으로 바꾼다.
            # 아니라면 valid로 변경 -> 패딩을 설정하지 않음

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer) # None, None, 3
            # 인트형의 텐서 shape를 전달함 prev_layer는 전 레이어에서 가져옴

            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size) # 다크넷 모양은 필터갯수, 입력 채널 크기, 필터의 크기 이다.
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn'
                  if batch_normalize else '  ', activation, weights_shape) # 배치 정규화 항목이 있으면 'bn' 없으면 공백 출력

            conv_bias = np.ndarray( # bias 생성 
                shape=(filters, ), # 필터 개수 만큼 shape(filters, 1)
                dtype='float32', # 32비트 실수형으로
                buffer=weights_file.read(filters * 4)) # weight 파일을 읽어옴
            count += filters

            if batch_normalize: # bn이 True라면
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0]) # darknet의 가중치 형태는 tensorflow와 다르기 때문에 위치 변경
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]# batch_norm이 True라면 웨이트만 아니라면 바이어스도 같이 웨이트로 설정함

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # Create Conv2D layer
            if stride>1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer) # 패딩을 left top을 해주기 때문에 zeroPadding을 써서 직접 만든다.
            conv_layer = (Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding))(prev_layer) # convolution 실행 bn이 없으면 bias를 사용

            if batch_normalize: # bn이 있으면 배치 정규화 실행
                conv_layer = (BatchNormalization(
                    weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear': # 활성화 함수 추가 위에서는 제대로 된 활성화 함수인지만 확인
                all_layers.append(prev_layer) # 전체 레이어에 레이어를 추가함
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'): # route 레이어이면
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')] # ,를 기준으로 분리
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1: # layeys가 1개 초과면
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers) # 레이어가 여러개라면 연결 경로를 만들어 준다.
                all_layers.append(concatenate_layer) # 연결한 레이어를 추가한다.
                prev_layer = concatenate_layer
            else: # 1 이하이면
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer) # 그 래이어를 추가
                prev_layer = skip_layer

        elif section.startswith('maxpool'): # maxpool 레이어이면 yolov3 컨픽 파일에는 maxpool레이어는 없음
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'): # shortcut 레이어인 경우
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            out_index.append(len(all_layers)-1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    if len(out_index)==0: out_index.append(len(all_layers)-1)
    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    print(model.summary())
    if args.weights_only:
        model.save_weights('{}'.format(output_path))
        print('Saved Keras weights to {}'.format(output_path))
    else:
        model.save('{}'.format(output_path))
        print('Saved Keras model to {}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count +
                                                       remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))



if __name__ == '__main__':
    _main(parser.parse_args())
