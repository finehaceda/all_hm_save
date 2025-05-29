import json

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

from .forms import MyForm, EncodeForm, EncodeHiddenForm


def index1(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def detail(request, question_id):
    return HttpResponse("You're looking at question %s." % question_id)


def results(request, question_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % question_id)


def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)

def index(request):
    # latest_question_list = Question.objects.order_by("-pub_date")[:5]
    template = loader.get_template("index.html")
    context = {
        "latest_question_list": [{"id": 1, "question_text": "who?"},{"id": 2, "question_text": "where?"}],
    }
    return HttpResponse(template.render(context, request))
    # return HttpResponse("Hello, world. You're at the polls index.")


def dna_encoding_view(request):
    # 假设的初始参数值，这些值可以从表单中获取
    initial_params = {
        'sequence_length': 120,
        'mingc': 40,
        'maxgc': 60,
        'ecc': '',  # ECC可能需要更复杂的输入或选择
        'encoding_method': '喷泉码',  # 默认编码方法
        'single_dna_length': 120,
        'index_length': 12,
        'encoding_time': 0.02,
        'dna_sequence_number': 135,
        'information_density': 1.817,
    }

    # 如果这是POST请求，我们可以处理表单数据（这里省略）
    if request.method == 'POST':
        # 从表单中获取参数并更新initial_params（这里省略）
        pass

    # 将参数传递给模板进行渲染
    context = {
        'params': initial_params,
        'methods': ['喷泉码', 'YYC'],  # 编码方法列表
        'user_input': {
            "encoding_method": '喷泉码',
            "sequence_length": '120',
            "index_length": 12,
            "encoding_time": '0.02s',
        },
        # 'encoded_data': encoded_data,
        # 'info_density': info_density,
    }
    return render(request, 'encode.html', context)



# testform
def my_view(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            # 获取表单数据
            data = form.cleaned_data
            # 打印数据到控制台或进行其他处理
            for key, value in data.items():
                print(f'{key}: {value}')
            return HttpResponse('Form submitted successfully!')

    else:
        form = MyForm()

    return render(request, 'my_template.html', {'form': form})
# testform with encode
def dna_encoding_innerform(request):
    # 假设的初始参数值，这些值可以从表单中获取
    initial_params = {
        'sequence_length': 120,
        'mingc': 40,
        'maxgc': 60,
        'ecc': 1,  # ECC可能需要更复杂的输入或选择
        'encoding_method': '喷泉码',  # 默认编码方法
        'single_dna_length': 120,
        'index_length': 12,
        # 'encoding_time': 0.02,
        # 'dna_sequence_number': 135,
        # 'information_density': 1.817,
    }

    # 如果这是POST请求，我们可以处理表单数据（这里省略）
    if request.method == 'POST':
        # 从表单中获取参数并更新initial_params（这里省略）
        # pass
        form = EncodeForm(request.POST)
        print(f"?????????????{form.is_valid()}??????????")
        if form.is_valid():
            # 获取表单数据
            data = form.cleaned_data
            # 打印数据到控制台或进行其他处理
            for key, value in data.items():
                print(f'{key}: {value}')
            # return HttpResponse('Form submitted successfully!')

            # 获取用户输入的数据
            user_input = form.cleaned_data

            # 这里我们模拟调用一个外部接口进行编码
            # 实际上，你应该替换为真实的编码接口
            # response = requests.post('https://your-encoding-api-endpoint', json={'data': user_input})
            # encoded_data = response.json()  # 假设返回的是JSON格式的数据
            encoded_data1 = {
                "info_density": 1.11,
                "encode_time": '10s',
                "sequence_number": 85,
                "index_length": 12
            }
            encoded_data2 = {
                "info_density": 0.99,
                "encode_time": '17s',
                "sequence_number": 124,
                "index_length": 12
            }
            # 提取编码信息，如信息密度等（这里需要根据实际API返回的数据结构来调整）
            # info_density = encoded_data.get('info_density', 'N/A')

            # 将编码信息和原始数据传递给模板
            context = {
                'base': {"method1":user_input['method1'],"method2":user_input['method2'],},
                'user_input': user_input,
                'encoded_data1': encoded_data1,
                'encoded_data2': encoded_data2,
            }
            return render(request, 'encode.html', context)

        else:
            print(form.errors)  # 打印错误信息

    # 将参数传递给模板进行渲染
    # context = {
    #     'params': initial_params,
    #     'methods': ['喷泉码', 'YYC','derrick','极化码','YYC'],  # 编码方法列表
    # }

    context = {
        'base': {"method1":"fountain","method2":"yyc",},
        # 'encoded_data1': encoded_data1,
        # 'encoded_data2': encoded_data2,
    }
    return render(request, 'encode.html', context)

# def reset_data(request):
#     # 处理重置逻辑，这里我们简单地修改user_data作为示例
#     global user_data  # 注意：在实际应用中，避免使用全局变量，应使用数据库或会话等持久化存储
#     user_data["user_input"] = "Reset Value"  # 或者从数据库获取新的值
#     return json_response(user_data)  # 返回更新后的数据
@csrf_exempt #尝试不使用表单处理编码请求的情况
def dna_encoding(request):
    # 假设的初始参数值，这些值可以从表单中获取

    # 如果这是POST请求，我们可以处理表单数据（这里省略）
    if request.method == 'POST':
        user_input = json.loads(request.body.decode('utf-8'))
        print(f"user_input:{user_input}")
        # # 在这里执行你的操作，并获取结果
        # input_data = user_input['encodingMethod1']
        # result = your_function_to_execute(input_data)
        encoded_data1 = {
            "info_density": 1.11,
            "encode_time": '10s',
            "sequence_number": 85,
            "index_length": 12
        }
        encoded_data2 = {
            "info_density": 0.99,
            "encode_time": '17s',
            "sequence_number": 124,
            "index_length": 12
        }
        # 提取编码信息，如信息密度等（这里需要根据实际API返回的数据结构来调整）
        # info_density = encoded_data.get('info_density', 'N/A')

        # 将编码信息和原始数据传递给模板
        context = {
            'base': {"method1": user_input['method1'], "method2": user_input['method2'], "sequence_length": user_input['sequence_length'],
                     "mingc": user_input['mingc'], "maxgc": user_input['maxgc'],},
            'user_input': user_input,
            'encoded_data1': encoded_data1,
            'encoded_data2': encoded_data2,
        }
        return render(request, 'encode.html', context)

    # 将参数传递给模板进行渲染
    # context = {
    #     'params': initial_params,
    #     'methods': ['喷泉码', 'YYC','derrick','极化码','YYC'],  # 编码方法列表
    # }
    context = {
        # 'base': {"method1":"fountain","method2":"yyc",},
        # 'encoded_data1': encoded_data1,
        # 'encoded_data2': encoded_data2,
        'base': {
            "method1":"fountain",
            "method2":"yyc",
            'sequence_length': 120,
            'mingc': 40,
            'maxgc': 60,
            'ecc': 1,
            'index_length': 12,
        }
    }
    return render(request, 'encode.html', context)


# @csrf_exempt 使用两个表单的情况
def dna_encoding_1(request):
    # 假设的初始参数值，这些值可以从表单中获取
    initial_params = {
        "method1": "fountain",
        "method2": "yyc",
        'sequence_length': 120,
        'mingc': 40,
        'maxgc': 60,
        'ecc': 1,
        'index_length': 12,
    }
    # 如果这是POST请求，我们可以处理表单数据（这里省略）
    if request.method == 'POST':
        form = EncodeForm(request.POST)
        print(f"?????????????{form.is_valid()}??????????")
        if form.is_valid():
            # 获取表单数据
            data = form.cleaned_data
            # 打印数据到控制台或进行其他处理
            # for key, value in data.items():
            #     print(f'{key}: {value}')
            # 获取用户输入的数据
            user_input = form.cleaned_data

            encoded_data1 = {
                "info_density": 1.11,
                "encode_time": '10s',
                "sequence_number": 85,
                "index_length": 12
            }
            encoded_data2 = {
                "info_density": 0.99,
                "encode_time": '17s',
                "sequence_number": 124,
                "index_length": 12
            }
            # 提取编码信息，如信息密度等（这里需要根据实际API返回的数据结构来调整）
            # info_density = encoded_data.get('info_density', 'N/A')

            form = EncodeHiddenForm(initial={
                'hidden_method1': user_input['method1'],
                'hidden_method2': user_input['method2'],
                'hidden_mingc': user_input['mingc'],
                'hidden_maxgc': user_input['maxgc'],
                'hidden_sequence_length': user_input['sequence_length'],
                "info_density1": 1.11,
                "encode_time1": '10s',
                "sequence_number1": 85,
                "index_length1": 12,
                "info_density2": 0.99,
                "encode_time2": '17s',
                "sequence_number2": 124,
                "index_length2": 12
            })
            # 将编码信息和原始数据传递给模板
            context = {
                'initial_params': initial_params,
                'base': {"method1": user_input['method1'], "method2": user_input['method2'],
                         "sequence_length": user_input['sequence_length'],
                         "mingc": user_input['mingc'], "maxgc": user_input['maxgc'], },
                # 'user_input': user_input,
                'form': form,
                'encoded_data1': encoded_data1,
                'encoded_data2': encoded_data2,
            }
            return render(request, 'encode.html', context)

        else:
            print(form.errors)  # 打印错误信息

    context = {
        'initial_params': initial_params,
        'base': initial_params,
    }
    return render(request, 'encode.html', context)

# 从编码到合成页面，封装编码后的数据

def encode_to_synseq(request):
    if request.method == 'POST':
        # 从表单中获取参数并更新initial_params（这里省略）
        # pass
        form = EncodeHiddenForm(request.POST)
        print(f"?????????????{form.is_valid()}??????????")
        if form.is_valid():
            # 获取表单数据
            data = form.cleaned_data
            # 打印数据到控制台或进行其他处理
            for key, value in data.items():
                print(f'{key}: {value}')
        #     # 获取用户输入的数据
        #     user_input = form.cleaned_data
        #
        #
        #     encoded_data1 = {
        #         "info_density": 1.11,
        #         "encode_time": '10s',
        #         "sequence_number": 85,
        #         "index_length": 12
        #     }
        #     encoded_data2 = {
        #         "info_density": 0.99,
        #         "encode_time": '17s',
        #         "sequence_number": 124,
        #         "index_length": 12
        #     }
        #     # 提取编码信息，如信息密度等（这里需要根据实际API返回的数据结构来调整）
        #     # info_density = encoded_data.get('info_density', 'N/A')
        #
        #     form = EncodeHiddenForm(initial={
        #         'hidden_method1': user_input['method1'],
        #         'hidden_method2': user_input['method2'],
        #         'hidden_mingc': user_input['mingc'],
        #         'hidden_maxgc': user_input['maxgc'],
        #         'hidden_sequence_length': user_input['sequence_length'],
        #     })
        #     # 将编码信息和原始数据传递给模板
            context = {}
            return render(request, 'synseq.html', context)
        #
        # else:
        #     print(form.errors)  # 打印错误信息

    context = {}
    # return render(request, 'synseq.html', context)

    # context = {
    #     # 'clustering_methods': ['基于index聚类', '基于全序列聚类'],
    # }
    # return render(request, 'synseq.html', context)


def synseq_view(request):
    context = {
        'clustering_methods': ['基于index聚类', '基于全序列聚类'],
    }
    return render(request, 'synseq.html', context)



def cluster_view(request):
    context = {
        'clustering_methods': ['基于index聚类', '基于全序列聚类'],
    }
    return render(request, 'cluster.html', context)


def decode_view(request):
    context = {}
    return render(request, 'decode.html', context)

@csrf_exempt
def execute_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            input_data = data['encodingMethod1']

            # 在这里执行你的操作，并获取结果
            result = your_function_to_execute(input_data)
            # print(f"input_data:{input_data}")
            # print(f"result:{result}")
            return JsonResponse({'result': result})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


def your_function_to_execute(input_data):
    # 这里是你的执行逻辑
    # 返回执行结果
    return f"The result of your function with input '{input_data}' is ..."


