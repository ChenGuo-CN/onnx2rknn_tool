import argparse
import os
import yaml
from rknn.api import RKNN

# 它接受一个参数 yaml_config_path，这个参数是一个字符串，表示YAML配置文件的路径。
# 函数的目的是读取这个YAML配置文件，解析其内容，并返回解析后的配置字典
def parse_model_config(yaml_config_path):
    with open(yaml_config_path,'r',encoding='utf-8') as f:
        yaml_config = f.read()
        print("=============模型转换配置================")
        print(yaml_config)
    model_configs = yaml.load(yaml_config,Loader=yaml.FullLoader)
    return model_configs

# 目的是将一个字符串（或布尔值）转换成布尔值（True 或 False），这个函数用于命令行参数解析
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    elif v.lower() in ('debug'):
        return 'debug'
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def convert_model(config_path, target_platform, output_dir, eval_perf_memory=False,
                  accuracy_analysis=False, verbose=False, device_id=None,
                  model_name=None,set_input_size_list_str=None):
    # 对没有默认值的参数进行检测
    if config_path is None or target_platform is None or output_dir is None:
        print("config_path/target_platform/output_dir are missing")
        return -1
    
    # 解析 config_path, 并检查配置文件是否存在
    yaml_config_path = config_path
    if not(os.path.exists(yaml_config_path)):
        print("model config %s not exists!" % yaml_config_path)
        return -1
    
    yaml_config_dir = os.path.abspath(os.path.dirname(yaml_config_path))

    # 打印日志信息
    # TODO 能否改变日志存储和展示方式
    print("=========================================")
    print("convert_model:")
    print("  config_path=%s" % yaml_config_path)
    print("  config_dir=%s" % str(yaml_config_dir))
    print("  out_path=%s" % output_dir)
    print("  target_platform=%s" % str(target_platform))
    print("=========================================")

    # 调用配置加载函数加载模型转换的详细配置
    model_configs = parse_model_config(yaml_config_path)
    print("=========================================")
    
    model = None
    model = model_configs['models']
    model['configs']['target_platform'] = target_platform   # TODO 这个覆盖模式有必要吗

    # 判断模型转换配置是否存在
    if model is None:
        print("Error: No valid model config in %s !\n" %yaml_config_path)
        return -1
    
    # 判断配置中必要项是否存在
    if model_name is None:
        if 'name' in model:
            model_name = model['name']
        else:
            print("Error: Please provide the model name !\n")

    # 构造RKNN实例
    rknn = RKNN(verbose=verbose)

    # 配置RKNN实例
    ## mean_values and std_values
    ### TODO 官方api中尝试从 channel_mean_value 中计算 mean_values 和 std_values, 为什么?
    if 'mean_values' in model['configs'].keys() and 'std_values' in model['configs'].keys():
        mean_values = model['configs']['mean_values']
        std_values = model['configs']['std_values']
    else:
        mean_std = model['configs']['channel_mean_value'].split(' ')
        mean_std = list(map(float, mean_std))
        mean_values = mean_std[0:-1]
        std_values = [mean_std[-1]] * len(mean_values)
        if mean_values == [-1]:
            mean_values = None
        if std_values == [-1]:
            std_values = None
    ## rgb2bgr
    rgb2bgr = False
    if 'quant_img_RGB2BGR' in model['configs']:
        rgb2bgr = str2bool(model['configs']['quant_img_RGB2BGR'])

    ## quantized_dtype 
    quantized_dtype = 'asymmetric_quantized-8'
    if 'quantized_dtype' in model['configs']:
        quantized_dtype = model['configs']['quantized_dtype']
    if quantized_dtype == 'asymmetric_quantized-u8':
        quantized_dtype = 'asymmetric_quantized-8'

    ## quantized_algorithm
    quantized_algorithm = 'normal'
    if 'quantized_algorithm' in model['configs']:
        quantized_algorithm = model['configs']['quantized_algorithm']

    ## quantized_method
    quantized_method = 'channel'
    if 'quantized_method' in model['configs']:
        quantized_method = model['configs']['quantized_method']
    
    ## optimization_level
    optimization_level = 3
    if 'optimization_level' in model['configs']:
        optimization_level = model['configs']['optimization_level']

    ## model_pruning    
    model_pruning = False
    if 'model_pruning' in model['configs']:
        model_pruning = str2bool(model['configs']['model_pruning'])
    
    ## quantize_weight
    quantize_weight = False
    if 'quantize_weight' in model['configs']:
        quantize_weight = str2bool(model['configs']['quantize_weight'])
    
    ## single_core_mode
    single_core_mode = False
    if 'single_core_mode' in model['configs']:
        single_core_mode = str2bool(model['configs']['single_core_mode'])
    
    ## sparse_infer
    sparse_infer=False
    if 'sparse_infer' in model['configs']:
        sparse_infer = str2bool(model['configs']['sparse_infer'])



    ## quantized_hybrid_level
    quantized_hybrid_level=0
    if 'quantized_hybrid_level' in model['configs']:
        quantized_hybrid_level = model['configs']['quantized_hybrid_level']

    ## float_dtype
    float_dtype='float16'
    if 'float_dtype' in model['configs']:
        float_dtype = model['configs']['float_dtype']

    ## custom_string
    custom_string=None
    if 'custom_string' in model['configs']:
        if model['configs']['custom_string'] == "None":  
            custom_string = None 
        else:
            custom_string = model['configs']['custom_string']

    ## remove_weight
    remove_weight=False
    if 'remove_weight' in model['configs']:
        remove_weight = str2bool(model['configs']['remove_weight'])

    ## compress_weight
    compress_weight=False
    if 'compress_weight' in model['configs']:
        compress_weight = str2bool(model['configs']['compress_weight'])

    ## inputs_yuv_fmt
    inputs_yuv_fmt=None
    if 'inputs_yuv_fmt' in model['configs']:
        if model['configs']['inputs_yuv_fmt'] == "None":  
            inputs_yuv_fmt = None 
        else:
            inputs_yuv_fmt = model['configs']['inputs_yuv_fmt']

    ## dynamic_input
    dynamic_input=None
    if 'dynamic_input' in model['configs']:
        if model['configs']['dynamic_input'] == "None":  
            dynamic_input = None 
        else:
            dynamic_input = model['configs']['dynamic_input']

    ## op_target
    op_target=None
    if 'op_target' in model['configs']:
        if model['configs']['op_target'] == "None":  
            op_target = None 
        else:
            op_target = model['configs']['op_target']

    ## remove_reshape
    remove_reshape=False
    if 'remove_reshape' in model['configs']:
        remove_reshape = str2bool(model['configs']['remove_reshape'])

    ## enable_flash_attention
    enable_flash_attention=False
    if 'enable_flash_attention' in model['configs']:
        enable_flash_attention = str2bool(model['configs']['enable_flash_attention'])
    
    print("mean_values:" + str(mean_values))
    print("std_values:" + str(std_values))
    print("quant_img_RGB2BGR: " + str(rgb2bgr))
    print("quantize: " + str(model['quantize']))
    print("quantized_dtype: " + str(quantized_dtype))
    print("quantized_algorithm: " + str(quantized_algorithm))
    print("target_platform: " + str(target_platform))
    print("quantized_method: " + str(quantized_method))
    print("optimization_level: " + str(optimization_level))

    print("quantized_hybrid_level: " + str(quantized_hybrid_level))
    print("float_dtype: " + str(float_dtype))
    print("custom_string: " + str(custom_string))
    print("remove_weight: " + str(remove_weight))
    print("compress_weight: " + str(compress_weight))
    print("inputs_yuv_fmt: " + str(inputs_yuv_fmt))
    print("dynamic_input: " + str(dynamic_input))
    print("op_target: " + str(op_target))
    print("remove_reshape: " + str(remove_reshape))
    print("enable_flash_attention: " + str(enable_flash_attention))

    rknn.config(mean_values=mean_values,
                std_values=std_values,
                quant_img_RGB2BGR=rgb2bgr,
                quantized_dtype=quantized_dtype,
                quantized_algorithm=quantized_algorithm,
                target_platform=target_platform,
                quantized_method=quantized_method,
                optimization_level=optimization_level,
                model_pruning=model_pruning,
                quantize_weight=quantize_weight,
                single_core_mode=single_core_mode,
                sparse_infer=sparse_infer,

                quantized_hybrid_level=quantized_hybrid_level,
                float_dtype=float_dtype,
                custom_string=custom_string,
                remove_weight=remove_weight,
                compress_weight=compress_weight,
                inputs_yuv_fmt=inputs_yuv_fmt,
                dynamic_input=dynamic_input,
                op_target=op_target,
                remove_reshape=remove_reshape,
                enable_flash_attention=enable_flash_attention
                )

    # 加载原始模型
    input_size_list = None
    inputs = None
    outputs = None
    if 'subgraphs' in model:
        input_size_list_str = None
        if set_input_size_list_str is not None:
            input_size_list_str = set_input_size_list_str
        elif 'input_size_list' in model['subgraphs']:
            input_size_list_str = model['subgraphs']['input_size_list']
        if input_size_list_str:
            input_size_list = []
            for input_size_str in input_size_list_str:
                input_size = list(map(int, input_size_str.split(',')))
                if len(input_size) == 3:
                    input_size.insert(0, 1)
                input_size_list.append(input_size)

        if 'inputs' in model['subgraphs']:
            inputs = model['subgraphs']['inputs']

        if 'outputs' in model['subgraphs']:
            outputs = model['subgraphs']['outputs']

        model_file_path = ''
        if 'pt_file_path' in model.keys():
            model_file_path = os.path.join(yaml_config_dir, model['pt_file_path'])
        elif 'model_file_path' in model.keys():
            model_file_path = os.path.join(yaml_config_dir, model['model_file_path'])
        if model['platform'] == 'onnx':
            rknn.load_onnx(model=model_file_path, inputs=inputs, outputs=outputs, input_size_list=input_size_list)
        else:
            print("Error: Only support ONNX to RKNN !\n")
            return -1
    
    # 加载量化数据集的相关配置
    if 'dataset' in model:
        dataset_path = os.path.join(yaml_config_dir, model['dataset'])
    else:
        dataset_path = None
    # 加载模型输出路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if model_name[-5:] == '.rknn':
        model_name = model_name[:-5]
    export_rknn_model_path = "%s.rknn" % (os.path.join(output_dir, model_name))
    
    # 构建RKNN模型并执行量化
    ret = rknn.build(do_quantization=model['quantize'], dataset=dataset_path)
    if ret != 0:
        print("rknn build fail " + str(ret))
        return -1
    
    # 导出RKNN模型
    ret = rknn.export_rknn(export_path=export_rknn_model_path)
    if ret != 0:
        print("rknn build fail " + str(ret))
        return -1
    print("output rknn path: " + export_rknn_model_path)

    # 如果连上板子可以进行性能测试
    if eval_perf_memory:
        if device_id == True:
            device_id = None
        ret = rknn.init_runtime(
            target_platform, perf_debug=False, eval_mem=False, device_id=device_id)
        if ret != 0:
            print('Init runtime failed.')
            exit(ret)
        rknn.eval_perf()
        ret = rknn.init_runtime(
            target_platform, perf_debug=False, eval_mem=True, device_id=device_id)
        if ret != 0:
            print('Init runtime failed.')
            exit(ret)
        rknn.eval_memory()
    # 精度测试
    if accuracy_analysis != None :
            if device_id != None :
                if device_id == True:
                    device_id = None
                ret = rknn.accuracy_analysis(inputs=[accuracy_analysis], target=target_platform, device_id=device_id)
            else:
                ret = rknn.accuracy_analysis(inputs=[accuracy_analysis])
            if ret != 0:
                print('accuracy_analysis failed.')
                exit(ret)
    elif accuracy_analysis != None and dataset_path == None:
        print("error: If accuracy_analysis is turned on, the dataset parameters and content in the yml file must be filled in, otherwise accuracy_analysis will not take effect.")

    results = None

    return results







    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ONNX2RKNN tool")

    parser.add_argument('-i', '--input_yml', 
                        help="yml config file path", 
                        # required=True,
                        default='mobilenetv2.yml')
    
    parser.add_argument('-o', '--output_dir', 
                        help="output dir", 
                        # required=True,
                        default='./')
    
    parser.add_argument('-t', '--target_platform',
                        help="target_platform, support rk3568/rk3566/rk3562/rk3588", 
                        # choices=['rk3568','rk3566','rk3562','rk3588'],
                        type=str, 
                        # required=True
                        default='rk3588')
     
    parser.add_argument('-a', '--accuracy_analysis',
                        help="Usage: -a \"xx1.jpg xx2.jpg\" Simulator accuracy_analysis, if want to turn on board accuracy_analysis, please use -d", 
                        type=str, 
                        default=None)
    
    parser.add_argument('-e', '--eval_perf_memory',
                        help="eval model perf and memory, board debugging is required, multi adb device use -d, default=false", 
                        action='store_true')
    
    parser.add_argument('-v', '--verbose', 
                        help="whether to print detailed log information on the screen", 
                        action='store_true')

    parser.add_argument('-d', '--device_id', 
                        help="Single adb device usage: -d. Multi adb device usage: -d device_id",
                        type=str, default=None)
    
    args = parser.parse_args()

    input_path = args.input_yml
    output_dir = args.output_dir
    target_platform = args.target_platform
    accuracy_analysis = args.accuracy_analysis
    eval_perf_memory = args.eval_perf_memory
    verbose = args.verbose
    device_id = args.device_id

    convert_model(
        config_path = input_path,
        output_dir = output_dir,
        target_platform = target_platform,
        accuracy_analysis = accuracy_analysis,
        eval_perf_memory=eval_perf_memory,
        verbose=verbose,
        device_id=device_id
    )
