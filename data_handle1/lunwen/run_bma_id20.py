import csv
import subprocess

from joblib import Parallel, delayed


def process_file(input_file, output_file, label, num=10000):
    search_string = ['Substitution rate','Deletion rate','Insertion rate','Error rate','Success rate','Time taken']
    # 1. 读取文件尾部
    with open(input_file, 'r', encoding='utf-8') as file:
        last_10_lines = file.readlines()[-10:]
    # 2. 查找包含特定字符串的行
    fi = 0
    save = ['']*6
    for i in range(len(last_10_lines)-1):
        if last_10_lines[i].find(search_string[fi]) != -1:
            save[fi] = last_10_lines[i].strip('\n').split(':\t')[-1]
            fi+=1
    save[-1]=last_10_lines[-1].strip('\n').split(': ')[-1]
    # 3. 将符合条件的行写入输出文件
    # data = [
    #     (label,'',save[3], '', save[4], save[0], save[1], save[2],save[5]),
    # ]
    editerr = float(save[3])*1000/num
    editsuc = 1-editerr
    indels = [float(save[0])*1000/num,float(save[1])*1000/num,float(save[2])*1000/num]
    data = [
        (label,editsuc, editerr, 1-float(save[4]), float(save[4]), indels[0], indels[1], indels[2],save[5]),
    ]
    with open(output_file, 'a', encoding='utf8', newline='') as f:
        writer = csv.writer(f)  # csv.writer()中可以传一个文件对象
        for line in data:  # 该data既可以是列表嵌套列表的数据类型也可以是列表嵌套元组的数据类型
            writer.writerow(line)

def run_methods(i,m,basedir='/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/Reconstructionmain'):
    print(f"{m}-{i} start...")
    shell = f'{basedir}/{m}/DNA /home2/hm/datasets/Randomaccess/no_fix_alldata/data{i}.txt out5 > {basedir}/{m}/id200210/result_{i}_.txt'
    result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"result:{result.stdout} , error info:{result.stderr}\n")
    process_file(f'{basedir}/{m}/id200210/result_{i}_.txt', f'{basedir}/lunwen/id200210_result_1.csv', f"{m}_{i}", 4819)

def run_id20(i):
    basedir = '/home1/hongmei/00work_files/0000/0ifirstCompare/CompareTest/Reconstructionmain'
    dataset = 'LDPC_Chandak0210'
    methods = ['Iterative']
    # methods = ['BMALA','DivBMA']
    # methods = ['Hybrid']
    # results = (Parallel(n_jobs=4)(delayed(run_methods)(i,m) for m in methods))
    # for i in range(5,21):
    for m in methods:
        print(f"{m}-{i} start...")
        # shell = f'{basedir}/{m}/DNA /home2/hm/datasets/Randomaccess/no_fix_alldata/data{i}_.txt out > {basedir}/{m}/{dataset}/result_{i}_1.txt'
        shell = f'{basedir}/{m}/DNA /home2/hm/datasets/LDPC_Chandak/data{i}.txt out > {basedir}/{m}/{dataset}/result_{i}.txt'
        print(f"shell:{shell}")
        # shell = f'{basedir}/{m}/DNA /home2/hm/datasets/Randomaccess/no_fix_alldata/11.txt out > {basedir}/{m}/id200210/result_{i}_.txt'
        result = subprocess.run(shell, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"result:{result.stdout} , error info:{result.stderr}\n")
        process_file(f'{basedir}/{m}/{dataset}/result_{i}.txt', f'{basedir}/lunwen/{dataset}_result.csv', f"{m}_{i}",4819)
run_id20(25)
# results = (Parallel(n_jobs=1)(delayed(run_id20)(i) for i in range(25, 26)))

