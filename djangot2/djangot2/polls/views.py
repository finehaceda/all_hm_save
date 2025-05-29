import os
# print("Current working directory:", os.getcwd())
# print("Sys.path:", sys.path)
import sys
# sys.path.append('/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform')
import traceback

from django.core.files.storage import default_storage
from django.http import HttpResponse, Http404
from django.shortcuts import render

# from djangot2.polls.forms import MyForm, EncodeForm, EncodeHiddenForm
# from .forms import MyForm, EncodeForm, EncodeHiddenForm
# from ...commain.encode import test222
# from django.views.decorators.csrf import csrf_exempt
# from Evaluation_platform.djangot2.polls.forms import MyForm, EncodeForm, EncodeHiddenForm
# from Evaluation_platform.commain.encode import test222
# from djangot2 import settings
# from .forms import MyForm, EncodeForm, EncodeHiddenForm
# from ...main0102 import test_dnafountain
from djangot2 import settings
from polls.Code.cluster import cluster_by_ref, cluster_by_index
from polls.Code.encode_all import test11, getDnaFountainEncodeInfo, getYYCEncodeInfo, getDerrickEncodeInfo, \
    getHedgesEncodeInfo, getPolarEncodeInfo, getDnaFountainDecodeInfo, getYYCDecodeInfo, getDerrickDecodeInfo, \
    getHedgesDecodeInfo, getPolarDecodeInfo
from polls.Code.reconstruct_all import reconstruct_seq
from polls.Code.simulate import adddt4simu_advanced
from polls.Code.utils import SimuInfo, getoriandallseqs_nophred, read_unknown_andsave, write_dict_to_csv, \
    initial_params, read_unknown_andsave_hedges, readandsave_noline0, savelistfasta, readandsave, readandsavefastq
from polls.forms import EncodeHiddenForm


# 添加项目根目录到sys.path
# from .forms import EncodeHiddenForm
# from polls.forms import EncodeHiddenForm


def getform(user_input:dict):
    form = EncodeHiddenForm(initial={
        'hidden_method1': user_input['hidden_method1'],
        'hidden_method2': user_input['hidden_method2'],
        'hidden_mingc': user_input['hidden_mingc'],
        'hidden_maxgc': user_input['hidden_maxgc'],
        'hidden_sequence_length': user_input['hidden_sequence_length'],
        "info_density1": user_input['info_density1'],
        "encode_time1": user_input['encode_time1'],
        "sequence_number1": user_input['sequence_number1'],
        "index_length1": user_input['index_length1'],
        "info_density2": user_input['info_density2'],
        "encode_time2": user_input['encode_time2'],
        "sequence_number2": user_input['sequence_number2'],
        "index_length2": user_input['index_length2']
    })
    return form


forsimulatefile = 'index_DNAFountain.fasta'
forclusterrecfile = 'simulated_seqsr1r2.fasta'

encodeinfo = {

    'fountain':{
        "info_density": 1.11,
        "encode_time": 10,
        "sequence_number": 85,
        "index_length": 12
    },
    'YYC':{
        "info_density": 0.99,
        "encode_time": 17,
        "sequence_number": 124,
        "index_length": 12
    },
    'derrick':{
        "info_density": 1.02,
        "encode_time": 30,
        "sequence_number": 57,
        "index_length": 12
    },
    'PolarCode':{
        "info_density": 1.0,
        "encode_time": 68,
        "sequence_number": 98,
        "index_length": 15
    },
    'hedges':{
        "info_density": 0.66,
        "encode_time": 50,
        "sequence_number": 33,
        "index_length": 13
    },
}


def handle_file(file):
    file_name = file.name
    file_path = os.path.join('uploads', file_name)
    # 确保上传目录存在
    os.makedirs('uploads', exist_ok=True)
    # 保存文件
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return file_path

UPLOAD_FOLDER = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/media'
FILE_FOLDER = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/'
def upload_file(request,name='file'):

    # print(f"-----files--------???:{request}")
    if request.method == 'POST' and name in request.FILES:
        uploaded_file = request.FILES[name]
        print(f"-----files--------???:{uploaded_file}")
        # 生成文件名（这里简单使用原始文件名，实际应用中应该做更安全的处理）
        filename = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

        # 保存文件到指定目录
        with default_storage.open(filename, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        return HttpResponse('File uploaded successfully: ' + filename,status=200)
    else:
        return HttpResponse('Invalid request or no file uploaded', status=400)

def download_file(request,mode='encode'):
    # 指定文件存储的根目录
    # file_root = os.path.join(settings.MEDIA_ROOT, 'uploads')
    defaultpath = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/media/index_DNAFountain.fasta'
    if mode=='encode':
        filename=request.session.get('encode_outfile',defaultpath)
    elif mode=='simulate':
        filename=request.session.get('simulate_outfile',defaultpath)
    elif mode == 'fordecode_outfile':
        filename=request.session.get('cluster_or_rec_file',defaultpath)
    else:
        filename=request.session.get('decodefile',defaultpath)
    # 构建文件的完整路径
    # file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_path = filename
    print(f"ppppppppppppppppppppppppppppppppfajeiof:{file_path}")
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise Http404("File not found")

    # 打开文件并读取其内容
    with open(file_path, 'rb') as fh:
        response = HttpResponse(fh.read(), content_type="application/octet-stream")

    # 设置响应头，指定下载的文件名
    response['Content-Disposition'] = 'attachment; filename="{}"'.format(os.path.basename(file_path))
    #
    return response

# def getencodeinfo(user_input,filename):
#     encode_worker = DNAFountainEncode(input_file_path=filename, output_dir='./',
#                                       sequence_length=int(user_input.get('seq_length', 120)),
#                                       max_homopolymer=int(user_input.get('homopolymer', 4)),
#                                       rs_num=int(user_input.get('rs_num', 0)), add_redundancy=False, add_primer=False,
#                                       primer_length=20, redundancy=float(user_input.get('redundancy_rate', 0)))
#
#     print(f"-----------编码中-----------:{encode_worker.input_file_path}")
#     encode_worker.common_encode()
#     print(f"编码后的文件位置为:{encode_worker.output_file_path}")
#     info={
#         'total_bit': encode_worker.total_bit,
#         'total_base': encode_worker.total_base,
#         'encode_time':  encode_worker.encode_time,
#         'density':  encode_worker.density,
#         'gc':encode_worker.gc,
#         'seq_num':encode_worker.seq_num,
#     }
#     output_file_path=encode_worker.output_file_path
#     return info,output_file_path
writefilepath = './encodedecode_infos_0519.txt'
# @csrf_exempt
def dna_encoding(request):
    # 假设的初始参数值，这些值可以从表单中获取
    # 如果这是POST请求，我们可以处理表单数据（这里省略）

    filename = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/1.jpg'
    # if request.method == 'POST':
    if request.method == 'POST' and 'encodeBtn' in request.POST:
        user_input = request.POST
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/media/1.jpg'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/media/33.jpg'

        # filename = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/1.py'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/2.jpg'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/3.png'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/4.jpg'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/5.jpg'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/6.jpg'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/7.txt'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/8.txt'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/9.jpg'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/10.jpg'
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/11.pdf'
        # if request.FILES.get('file','None') != 'None':
        #     context = {
        #         'errors': 'please upload file first！',
        #         'base': initial_params,
        #     }
        #     print(f"请先进行编码!")
        #     return render(request, 'encode.html', context)

        # print(f"encode 开始编码-----------:{user_input}")

        # # 获取上传到服务器的图片地址
        fileres = upload_file(request)
        if fileres.status_code==200:
            filename = os.path.join(UPLOAD_FOLDER, request.FILES['file'].name)
            print(f"the file path is :{filename}")
        # filename='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/media/consus.fa'
        method = user_input.get('method','fountain')
        try:
            if method == 'fountain':
                print(f"encode 开始编码 fountain-----------:{user_input}")
                info,output_file_path = getDnaFountainEncodeInfo(user_input,filename)
            elif method == 'YYC':
                print(f"encode 开始编码 YYC-----------:{user_input}")
                info,output_file_path = getYYCEncodeInfo(user_input,filename)
            elif method == 'derrick':
                print(f"encode 开始编码 derrick-----------:{user_input}")
                info,output_file_path = getDerrickEncodeInfo(user_input,filename)
            elif method == 'PolarCode':
                print(f"encode 开始编码 PolarCode-----------:{user_input}")
                info,output_file_path = getPolarEncodeInfo(user_input,filename)
                request.session['matrices'] = info
                # print(f"??????????????????request.session['matrices']:{request.session.get('matrices').keys()}")
            else:
                print(f"encode 开始编码 hedges-----------:{user_input}")
                info,output_file_path = getHedgesEncodeInfo(user_input,filename)
        except Exception as e:
            context = {'base': initial_params,}
            print("捕获到异常:")
            print(f"异常类型: {type(e)}")
            print(f"异常信息: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"完整的堆栈跟踪信息:\n{''.join(traceback.format_exception(exc_type, exc_obj, exc_tb))}")

            return render(request, 'encode.html', context)
        # finally:
            # print(f"***编码介绍，查看info信息:{info}***")
        # 保存用户输入的编码信息+编码后的信息 到session
        # write_dict_to_csv({'':f'\n\n','method':f'--------------------{method}--------------------'},writefilepath)
        # info['method']=method
        infos = {
            'method': method,
            'density':info['density'],
            'total_bit':info['total_bit'],
            'total_base':info['total_base'],
            'seq_num':info['seq_num'],
            'max_homopolymer':info['max_homopolymer'],
            'gc':info['gc'],
            'encode_time':info['encode_time']
        }
        write_dict_to_csv({'':f'\n'},writefilepath)
        write_dict_to_csv(infos,writefilepath)
        # print(f"???????????????jafoeigj:{request.session.get('matrices','cjfoaeaj')}")
        print(f"编码时间：{info['encode_time']},max_homopolymer:{info['max_homopolymer']}")
        request.session['encode_input'] = user_input
        request.session['encode_outfile'] =FILE_FOLDER + os.path.basename(output_file_path)
        # info = encodeinfo.get(user_input['method'])
        request.session['encoded_data'] = info
        # user_input['filename'] = os.path.basename(filename)
        request.session['filename'] = filename

        context = {
            'base': user_input,
            'encoded_data':info,
            'filename':os.path.basename(filename)
        }


        return render(request, 'encode.html', context)
    elif request.method == 'POST' and 'next' in request.POST:
        user_input = request.POST
        # if request.session.get('encoded_data','None') == 'None':
        #     context = {
        #         'errors': '请先进行编码！',
        #         'base': request.session.get('encode_input',initial_params),
        #     }
        #     print(f"请先进行编码!")
        #     return render(request, 'encode.html', context)

        context = {
            'base': initial_params,
        }
        print(f"encode_to_synseq 编码结束，进入下一流程-----------:{user_input}")
        return render(request, 'simulate.html',context)
    context = {
        'base': {**initial_params},'filename':os.path.basename(filename),
    }
    request.session.flush()

    # print(f"1111filename:{filename}")
    print(f"开始编码filename-----------:{context['filename']}")
    return render(request, 'encode.html', context)



sequencing_data = {
    "time": '192s',
    "size": '1.34MB',
    "file_path": './1.fastq',
    'consensus_file_path':'consus_file1.fasta'
}

def getseques(filename):
    dnasequences = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i in range(1,len(lines[1:])):
        dnasequences.append(lines[i].strip('\n'))
    return dnasequences


simu_path = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/simu/'
def simulate_view(request):
    context = {
        'simulate': True,
        'base': {**initial_params,},
    }
    if request.method == 'POST' and 'stepsok' in request.POST:
        #TODO 若希望合成步骤默认有上一步编码后的文件，可以把编码后的文件名称保存到前端的一个隐藏字段，然后在这里获取。
        #或者，在flush session之前，把编码后的文件保存下来在这里使用。解码时也可这样处理
        user_input = request.POST
        # print(f"user_input:{user_input}")
        request.session['channel']=user_input.getlist('channel',['synthesis'])
        # request.session['channel']=user_input.getlist('channel',[])
        encode_outfile_front = os.path.basename(request.session.get('encode_outfile', ''))

        print(f"user_input.getlist('channel'):{user_input.getlist('channel')}")
        context['base'] = {**initial_params, **request.session}
        context['base']['filename'] = encode_outfile_front
        # print(f"steps ok context:{context}")
        return render(request, 'simulate.html', context)
    elif request.method == 'POST' and 'synseqBtn' in request.POST:
        user_input = request.POST
        # hidden_method1 = request.session.get('hidden_method1', 'not exist')  # 使用get方法，如果键不存在则返回默认值
        # print(f"user_input-----------:{user_input}")
        filename = request.session.get('encode_outfile', '')
        encode_outfile_front = os.path.basename(filename)

        # 获取上传到服务器的图片地址
        fileres = upload_file(request)
        if fileres.status_code==200:
            filename = os.path.join(UPLOAD_FOLDER, request.FILES['file'].name)
            encode_outfile_front = request.FILES['file'].name
            print(f"the upload file path is :{filename}")

        #1.若有文件未上传，则提示
        if filename == '':
            request.session.flush()
            context['base'] = {**initial_params, **request.session}
            context['errors'] = 'please upload file first！'

            return render(request, 'decode.html', context)
        print(f'filename:{filename}')
        # filename = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/index_PolarCode.fasta'

        # filename = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/media/index_DNAFountain (3).fasta'
        # def __init__(self,channel:list,syncycles=1,synyield:float=0.99,
        #             pcrcycle=2, pcrpro=0.8, decay_year=2, decaylossrate=0.3,sample_ratio=0.005,syn_method='illuminas',depth=10,):
        simuInfo = SimuInfo(
            inputfile_path=filename,synthesis_method = user_input.get('synthesis_method','electrochemical'),
            channel=request.session.get('channel',[]),
            oligo_scale = user_input.get('oligo_scale',1), sample_multiple = user_input.get('sample_multiple',100),
            pcrcycle = user_input.get('pcrcycle',2), pcrpro = user_input.get('pcrpro',0.8),
            decay_year =  user_input.get('decay_year',2),decaylossrate =  user_input.get('decaylossrate',0.3),
            sample_ratio =  user_input.get('sample_ratio',0.005), sequencing_method = user_input.get('sequencing_method','paired-end'),
            depth =  user_input.get('depth',10),badparams = user_input.get('badparams',''), thread = initial_params.get('thread_num'),
        )

        # dnasequences = getseques(filename)
        print(f"synseq_view 开始模拟合成测序-----------:{user_input}")

        if user_input.get('sequencing_method','paired-end') == "Nanopone" or simuInfo.sequencing_method == "Pacbio":
            filessave = simu_path+'simulated_seqsr1r2.fastq'
        else:
            filessave = simu_path+'simulated_seqsr1r2.fasta'
        infos = adddt4simu_advanced(simuInfo,filessave)
        request.session['simulate_outfile'] = infos.get('path')
        # request.session['sequencinsimulated_seqsr1r2g_method'] = user_input.get('sequencing_method','paired-end')
        write_dict_to_csv({'simulate_time':infos.get('time','')},writefilepath)
        # write_dict_to_csv(infos,writefilepath)
        print(f"已模拟完成,共耗时{infos.get('time')},path为：-----------:{infos.get('path') }")
        context['base'] = {**initial_params, **request.session}
        context['base']['filename'] = encode_outfile_front
        context['sequencing_data']= {'time':infos.get('time')}
        return render(request, 'simulate.html', context)
    matrices = request.session.get('matrices', dict())
    encode_outfile = request.session.get('encode_outfile','')
    filename = request.session.get('filename','')
    request.session.flush()
    request.session['encode_outfile']=encode_outfile
    request.session['filename']=filename
    request.session['matrices']=matrices
    context = {
        'base': {**initial_params,},
    }
    return render(request, 'simulate.html', context)


decoded_data = {
    "success": True,
    "time": '30s',
    "bit_rev": 0.9971,
    "seq_rev": 0.9878
}

output_dir = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode/'
def decode_view(request):
    context = {
        'cluster': True,
        'reconstruct': True,
        'base': {**initial_params},
    }
    matrices = request.session.get('matrices',dict())
    simulate_outfile = request.session.get('simulate_outfile','')
    encode_outfile = request.session.get('encode_outfile','')
    filename = request.session.get('filename','')
    forclusterfile = simulate_outfile
    if forclusterfile == '':
        forclusterfile = encode_outfile

    context['filename'] = os.path.basename(forclusterfile)
    context['filename1'] = os.path.basename(encode_outfile)
    # file_seqs = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/simu/'+ forclusterrecfile
    # file_ref = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/'+forsimulatefile
    # file_ref = ''

    if request.method == 'POST' and 'startButton' in request.POST:
        fileres2 = request.session.get('encode_outfile', '')
        # print(f"request.FILES:{request.FILES}")
        # if request.FILES.get('file','None') == 'None':
        #     context = {
        #         'errors': 'please upload file first！',
        #         'base': initial_params,
        #     }
        #     return render(request, 'decode.html', context)
        # filename,filename1 = 'simulated_seqsr1r2_.fasta','index_DNAFountain_.fasta'
        #1.获取文件
        try:
            file_seqs,filename = forclusterfile,os.path.basename(forclusterfile)
            # file_ref,filename1 = '',''
            file_ref = request.session.get('encode_outfile', '')
            filename1 = os.path.basename(file_ref)
            fileres1 = upload_file(request)
            fileres2 = upload_file(request,'ref_or_index')
            # file_seqs = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/simu/simulated_seqsr1r2.fastq'
            # file_ref = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/6_YinYangCode.fasta'
            # file_seqs = '/home2/hm/badreads/badreads_simutest_0.5.fastq'
            # file_ref = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/5_PolarCode.fasta'
            # file_seqs = '/home2/hm/simufiles/9_yinyang.simulated_seqsr1r2.fastq'
            # file_ref = '/home2/hm/simufiles/9_YinYangCode.fasta'

            if fileres1.status_code == 200 and fileres2.status_code == 200 :
                file_seqs = os.path.join(UPLOAD_FOLDER, request.FILES['file'].name)
                file_ref = os.path.join(UPLOAD_FOLDER, request.FILES['ref_or_index'].name)
                filename,filename1 = request.FILES['file'].name,request.FILES['ref_or_index'].name
            elif fileres1.status_code == 200:
                file_seqs = os.path.join(UPLOAD_FOLDER, request.FILES['file'].name)
                filename = request.FILES['file'].name
            elif fileres2.status_code == 200 :
                file_ref = os.path.join(UPLOAD_FOLDER, request.FILES['ref_or_index'].name)
                filename1 = request.FILES['ref_or_index'].name
            # print(f'file_seqs:{file_seqs}\nfile_ref:{file_ref}')
            context['filename3'] = os.path.basename(request.session.get('filename',''))
            # print(f"filename2:{request.session.get('cluster_or_rec_file','')}\nfilename3:{request.session.get('filename','')}")

            #2.不需要聚类的情况
            user_input = request.POST
            if user_input.get('cluster_method','allbase') == 'no':
                request.session['reconstruct']=user_input.get('reconstruct','no')
                request.session['cluster_method']='no'
                context['base'] = {**initial_params,**request.session}
                context['reconstruct'] = False
                request.session['cluster_or_rec_file'] = file_seqs

                filename = request.session.get('cluster_or_rec_file', '')
                if filename == '':
                    filename = forclusterfile
                context['filename2'] = os.path.basename(filename)

                # print(f"context111:{context}")
                return render(request, 'decode.html', context)
            print(f'file_ref:{file_ref}\nfile_seqs:{file_seqs}')
            #1.若有文件未上传，则提示
            if file_ref == '':
                request.session.flush()
                context = {'base': initial_params,}
                context['errors'] = 'please upload file first！'
                context['filename'] = os.path.basename(forclusterfile)

                return render(request, 'decode.html', context)
            request.session['file_ref_name']=filename1
            context['filename'] = filename
            context['filename1'] = filename1

            print(f"the upload file1(file_seqs) path is :{file_seqs}")
            print(f"the upload file2(file_ref) path is :{file_ref}")

            #3.需要聚类的情况，聚类后的序列不保留param
            request.session['reconstruct']=user_input.get('reconstruct','no')
            request.session['confidence'] = user_input.get('confidence', 'no')
            request.session['cluster_method']=user_input.get('cluster_method','index')
            request.session['copy_number']=int(user_input.get('copy_number',10))
            request.session['cluster_or_rec_file'] = file_seqs
            context['base'] = {**initial_params, **request.session}
            context['reconstruct']=False
            # print(f"context:{context}")
            save_infos={}

            #不要聚类，只重建


            # # TODO 这里要记得去掉，需要后面的聚类方法
            # with open(file_ref, 'r') as f:
            #     lines = f.readlines()
            # param = lines[0]
            # request.session['param'] = param
            # request.session['encode_outfile']=file_ref
            cluster_with_filename = output_dir + f"cluster.{os.path.basename(request.session.get('encode_outfile', 'fasta'))}"
            # 3.1 使用ref聚类 聚类不考虑copynum
            if user_input.get('cluster_method','allbase') == 'allbase':
                lens = -1
                cluster_file_path = cluster_by_ref(file_ref,file_seqs,int(lens),
                                                   cluster_with_filename,
                                                   # request.session['copy_number'],
                                                   output_dir,
                                                   )
                # ori_dna_sequences, all_seqs = getoriandallseqs_nophred(cluster_file_path.get('cluster_seqs_path',output_dir+'cluster.fasta'))
                # dna_sequences = getRadomSeqsNoQua(all_seqs, int(user_input.get('copy_number',10)))
                # saveclusterfile(output_dir + 'cluster.fasta', ori_dna_sequences, dna_sequences)
            # 3.2使用index聚类
            else:
                # TODO 待修改成与cluster_by_ref一样
                lens = user_input.get('index_length',12)
                cluster_file_path = cluster_by_index(file_ref,file_seqs,int(lens),cluster_with_filename)
            # request.session['cluster_or_rec_file'] = cluster_file_path.get('cluster_seqs_path',
            #                         output_dir+f"cluster.fasta")
            request.session['cluster_or_rec_file'] = cluster_file_path.get('cluster_seqs_path',cluster_with_filename)

            # 聚类后的序列不保留param，故聚类后保存到session，方便解码时获取
            request.session['param'] = cluster_file_path.get('param','')
            save_infos['cluster_time']=cluster_file_path.get('cluster_time', '')
            save_infos['cluster_phred_path']=cluster_file_path.get('cluster_phred_path', '')


            #聚类后检查是否有丢失
            ori_dna_sequences, all_seqs,_ = getoriandallseqs_nophred(request.session['cluster_or_rec_file'],1000)
            lasti = [i for i in range(len(all_seqs)) if len(all_seqs[i]) == 0]
            last_oriseqs = [ori_dna_sequences[i] for i in lasti]
            print(f"注意，这里有{len(last_oriseqs)}条序列丢失！")

            # write_dict_to_csv({'cluster_time': cluster_file_path.get('cluster_time', '')},writefilepath)
            # 4.需要重建序列，重建后的序列也不保留param 重建序列考虑 copy_number
            if user_input.get('reconstruct','no') == 'yes':
                reconstruct_file_path,reconstruct_time,editerrorrate,seqerrorrate = reconstruct_seq(user_input.get('rebuild_method','SeqFormer'),
                                                        user_input.get('confidence','no'),'',
                                                        cluster_file_path,
                                                        int(user_input.get('copy_number',10)),
                                                        output_dir)
                request.session['cluster_or_rec_file'] = reconstruct_file_path
                save_infos['reconstruct_time']=reconstruct_time
                save_infos['edit_errorrate']=editerrorrate
                save_infos['seq_errorrate']=seqerrorrate
            # decode file os.path.basename(request.session.get('cluster_or_rec_file',''))
            context['filename2'] = os.path.basename(request.session.get('cluster_or_rec_file',''))
            write_dict_to_csv(save_infos,writefilepath)
            # context['base']['filename2'] = request.session.get('cluster_or_rec_file','')
            return render(request, 'decode.html', context)
        except Exception as e:
            request.session.flush()
            request.session['simulate_outfile'] = simulate_outfile
            request.session['encode_outfile'] = encode_outfile
            context = {'base': initial_params, }
            context['filename'] = os.path.basename(forclusterfile)


            print("捕获到异常:")
            print(f"异常类型: {type(e)}")
            print(f"异常信息: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"完整的堆栈跟踪信息:\n{''.join(traceback.format_exception(exc_type, exc_obj, exc_tb))}")
            return render(request, 'decode.html', context)

    elif request.method == 'POST' and 'decodeBtn' in request.POST:
        #1.默认 通过ref/index聚类，不用重建，得到文件cluster.fasta；
        #2.通过ref/index聚类,需要重建，得到文件reconstruct.fasta/reconstruct.fastq;
        #3.无需聚类，无需重建，使用测序文件simulated_seqsr1r2_.fasta
        filename = request.session.get('cluster_or_rec_file', '')
        # request.session['encode_outfile']='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/7_hedges.fasta'
        #hedges TODO
        print(f"encode_outfile:{request.session.get('encode_outfile', '')}")
        with open(request.session.get('encode_outfile', ''), 'r') as f:
            lines = f.readlines()
        param = lines[0]
        request.session['param'] = param
        if filename == '':
            filename = forclusterfile
        #
        # UPLOAD_FOLDER='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/media'
        # file_ori = UPLOAD_FOLDER + '1.jpg'
        # file_ori_front = '1.jpg'
        # filename = ''
        # file_ori = ''
        # file_ori_front = ''
        # request.session['filename']='/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/upload/5.jpg'


        # 获取上传到服务器的图片地址 file2为decodefile file3为originfile
        fileres = upload_file(request,'file2')
        fileres1 = upload_file(request,'file3')
        if fileres.status_code==200:
            filename = os.path.join(UPLOAD_FOLDER, request.FILES['file2'].name)
            # file_decode_front = request.FILES['file2'].name
            print(f"the upload decode file path is :{filename}")
        if fileres1.status_code==200:
            file_ori = os.path.join(UPLOAD_FOLDER, request.FILES['file3'].name)
            file_ori_front = request.FILES['file3'].name
            print(f"the upload originfile path is :{file_ori}")
        else:
            file_ori = request.session.get('filename', '')
            print(f"the upload originfile path is :{file_ori}")
        #若有文件未上传，则提示
        # file_ori = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/media/1.jpg'
        if filename == '' or file_ori == '':
            context['base'] = {**initial_params, **request.session}
            context['errors'] = 'please upload file first！'
            context['base']['filename2'] = os.path.basename(request.session.get('cluster_or_rec_file', ''))
            return render(request, 'decode.html', context)

        context['filename1'] = request.session.get('file_ref_name','')
        context['filename2'] = os.path.basename(filename)
        context['filename3'] = os.path.basename(file_ori)

        print(f"fordecodefile:{filename}\noriginfile:{file_ori}")
        decode_outfile = filename
        # decode_outfile = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode/cluster.9_DNAFountain.fasta'
        # decode_outfile = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode/cluster.6_hedges.fasta'
        # decode_outfile = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode/cluster.'
        # decode_outfile = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/decode/reconstruct.fastq'


        # 处理文件，构造成这样
        # >seq0
        # GGAAGGACGAGTGAGGCCGTAAAGCAGCAACGACGGACGGCCGTCCGGCCAGCCTTAATTAGATAACTCCAGACTTCCGACCGTCCGACTTATCTATGTCCCTACTCCGATAAGCAACC
        # >seq1
        # GGAAGGACGAGTGAGGCCGTAAAGCAGCAACGACGGACGGCCGTCCGCCAGCCTTAATTAGATAACTCCAGACTTCCGACCGTCCGTACTTATCTATGTCCCTACTCCGATAAGCAACC
        #1.不用聚类
        # decode_outfile = request.session.get('cluster_or_rec_file',file_seqs)
        # decode_outfile = '/home1/hongmei/00work_files/0000/0isecondwork/Evaluation_platform/djangot2/files/index_DNAFountain_1.fasta'
        allseqs,allconfidence=[],[]
        user_input = request.POST
        method = user_input.get('method','fountain')
        copy_number = int(user_input.get('copy_number_decode',10))

        if str(decode_outfile).find('simulated_seqsr1r2') != -1:
            #直接使用模拟后的序列进行解码则无法限制copy_number解码，如果要限制copy_number，需要先聚类
            print(f"**************************************1*********************************************")
            # 处理 simulated_seqsr1r2_.fasta/simulated_seqsr1r2_.fastq
            allseqs,param = readandsave_noline0(decode_outfile,decode_outfile + '.forreconstruct')
            request.session['param'] = param.rstrip()
        # 2.需要聚类，不用重建
        elif str(decode_outfile).find('cluster') != -1:
            print(f"**************************************2*********************************************")
            # 处理 cluster.fasta+cluster.phred
            ori_dna_sequences, allclusterseqs,param = getoriandallseqs_nophred(decode_outfile,copy_number)
            allseqs = savelistfasta(decode_outfile + '.forreconstruct',allclusterseqs)
        # 3.需要聚类，需要重建
        elif str(decode_outfile).find('reconstruct') != -1:
            print(f"**************************************3*********************************************")
            # 处理 reconstruct.fasta/reconstruct.fastq
            if str(decode_outfile).endswith('.fastq'):
                allseqs,allconfidence,param = readandsavefastq(decode_outfile,decode_outfile + '.forreconstruct')
            else:
                allseqs,param = readandsave(decode_outfile,decode_outfile + '.forreconstruct')
        else:
            print(f"**************************************4*********************************************")
            # 处理 any
            allseqs,param = read_unknown_andsave(decode_outfile,decode_outfile + '.forreconstruct')
        #处理hedges
        # allseqs,param = read_unknown_andsave_hedges(decode_outfile,decode_outfile + '.forreconstruct',10)
        if param!='':
            request.session['param'] = param.rstrip()
        # TODO 后面需要检查 request.session['param'] 是否存在
        # print(f"allseqs:{len(allseqs)}")
        # if str(decode_outfile).startswith('cluster'):
        #     # 需要聚类，不用重建 处理 cluster.fasta+cluster.phred
        #     ori_dna_sequences, all_seqs = getoriandallseqs_nophred(decode_outfile)
        #     savefasta(decode_outfile + '.forreconstruct',all_seqs)
        # else:
        #     #    不用聚类 处理 simulated_seqsr1r2_.fasta/simulated_seqsr1r2_.fastq
        #     # 或 需要聚类，需要重建 处理 reconstruct.fasta/reconstruct.fastq
        #     readandsave_noline0(decode_outfile,decode_outfile + '.forreconstruct')
        # print(f"decode_outfile-----------:{decode_outfile}")
        soft_data = {
            'copy_number' : user_input.get('copy_number_decode', 10),
            'allconfidence':allconfidence,
            'decision':user_input.get('decision','hard')
        }
        param = request.session.get('param','')
        try:
            if method == 'fountain':
                print(f"------开始解码------,fountain param:{param}")
                decoded_data = getDnaFountainDecodeInfo(decode_outfile + '.forreconstruct',file_ori,allseqs,soft_data,param)
            elif method == 'YYC':
                print(f"------开始解码------,YYC param----------------:{param}")
                decoded_data = getYYCDecodeInfo(decode_outfile + '.forreconstruct',file_ori,allseqs,soft_data,param)
            elif method == 'derrick':
                print(f"------开始解码------,derrick param----------------:{param}")
                decoded_data = getDerrickDecodeInfo(decode_outfile + '.forreconstruct',file_ori,allseqs,param)
            elif method == 'hedges':
                print(f"------开始解码------,hedges param----------------:{param}")
                decoded_data = getHedgesDecodeInfo(decode_outfile + '.forreconstruct',file_ori,allseqs,param)
            elif method == 'PolarCode':
                if request.session.get('matrices', '') != '':
                    matrices = request.session['matrices']
                    print(f'matrices no null:{matrices.keys()}')
                print(f"------开始解码------,Polar param----------------:{param}")
                decoded_data = getPolarDecodeInfo(decode_outfile + '.forreconstruct',file_ori,allseqs,allconfidence,param,matrices)
                print(f'decoded_data:{decoded_data}')
        except Exception as e:
            request.session.flush()
            request.session['simulate_outfile'] = simulate_outfile
            request.session['encode_outfile'] = encode_outfile
            context = {'base': initial_params, }
            context['filename'] = os.path.basename(forclusterfile)

            print("捕获到异常:")
            print(f"异常类型: {type(e)}")
            print(f"异常信息: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"完整的堆栈跟踪信息:\n{''.join(traceback.format_exception(exc_type, exc_obj, exc_tb))}")
            return render(request, 'decode.html', context)

        # elif method == 'YYC':
        #     print(f"encode 开始解码 YYC-----------:{user_input}")
        #     info,output_file_path = getYYCEncodeInfo(user_input,filename)
        # elif method == 'derrick':
        #     print(f"encode 开始解码 derrick-----------:{user_input}")
        #     info,output_file_path = getDerrickEncodeInfo(user_input,filename)
        # elif method == 'PolarCode':
        #     print(f"encode 开始解码 PolarCode-----------:{user_input}")
        #     info,output_file_path = getPolarEncodeInfo(user_input,filename)
        # else:
        #     print(f"encode 开始解码 hedges-----------:{user_input}")
        #     info,output_file_path = getHedgesEncodeInfo(user_input,filename)
        decoded_data_infos = {'badbits': decoded_data.get('badbits', 0),
                              'allbits': decoded_data.get('allbits', 0),
                              'bits_recov': decoded_data.get('bits_recov', 0),
                              'decode_time':decoded_data.get('decode_time',0),
                              }
        write_dict_to_csv(decoded_data_infos,writefilepath)
        request.session['decodefile'] = decoded_data.get('decodefile')
        request.session['copy_number']=user_input.get('copy_number',10)
        request.session['decision']=user_input.get('decision','hard')
        context['decode']=True
        context['decoded_data']=decoded_data
        context['base'] = {**initial_params, **request.session}
        context['base']['method'] = method

        # context['base']['confidence']=user_input.get('confidence')
        # context['base']['copy_number']=user_input.get('copy_number')
        # context['base']['decision']=user_input['decision']
        # print(f"synseq_view 开始解码-----------:{context}")
        return render(request, 'decode.html', context)
    elif request.method == 'POST' and 'restart' in request.POST:
        context = {
            'base': initial_params,
        }
        request.session.flush()
        print(f"重新开始-----------")
        return render(request, 'encode.html', context)

    # simulate_outfile = request.session.get('simulate_outfile','')
    # encode_outfile = request.session.get('encode_outfile','')
    print(f"simulate_outfile:{simulate_outfile},encode_outfile:{encode_outfile}")
    request.session.flush()
    request.session['simulate_outfile']=simulate_outfile
    request.session['encode_outfile']=encode_outfile
    request.session['matrices']=matrices
    request.session['filename']=filename
    # forclusterfile = simulate_outfile
    # if forclusterfile =='':
    #     forclusterfile=encode_outfile
    # request.session.flush()
    context = {'base': initial_params,}
    context['filename'] = os.path.basename(forclusterfile)
    context['filename1'] = os.path.basename(encode_outfile)
    # filename= forclusterrecfile
    # context['base']['filename'] = filename
    return render(request, 'decode.html', context)

def reconstruct_view(request):
    # DNAFountainEncode(input_file_path=file_input, output_dir=output_dir, sequence_length=sequence_length, max_homopolymer=max_homopolymer,
    #                           rs_num=rs_num,add_redundancy=add_redundancy, add_primer=add_primer, primer_length=primer_length, redundancy=dnafountain_redundancy)
    return render(request, 'test.html',{'test':test11()})

def evaluate_view(request):
    context = {
        'base': {**initial_params},
        'MEDIA_URL':settings.MEDIA_URL
    }
    return render(request, 'evaluate.html', context)