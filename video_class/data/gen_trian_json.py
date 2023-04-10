'''
@Author: your name
@Date: 2022-01-06 20:15:26
@LastEditTime: 2022-01-14 14:36:14
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /gaoqingdong/图短小融合/model/UVAT/test2/UVAT/data/gen_trian_json.py
'''
import os
import json
audio_path = '/home/bpfsrw_8/gaoqingdong/抽帧/audio/'
video_path = '/home/bpfsrw_8/gaoqingdong/抽帧/image/'
w = open('./UVAT_train_categroy_only_has_title.json','w')
w1 = open("train.txt",'w')
dic = {}
root='./'
v_num = os.listdir(video_path)
a_num = os.listdir(audio_path)
for name in ["动物.txt",'动漫.txt']:
    with open(name)as ff:
        for ll in ff:
            data = ll.strip().split('\t')
            #生成全部数据集
            # if str(data[0])+'.mp4'+'.wav' in a_num and str(data[0])+'.mp4' in v_num and len(os.listdir(video_path+str(data[0])+'.mp4'))>0: 
            #     dic[str(data[0])] = {"title":data[2],"video_image_path":video_path+str(data[0])+'.mp4',\
            #          "video_audio_path": audio_path+str(data[0])+'.mp4'+'.wav',"label":data[-1] }
            #生成只有标题的数据集
            #dic[str(data[0])] = {"title":data[2], "label":data[-1] } 
            # if str(data[0])+'.mp4' in v_num and len(os.listdir(video_path+str(data[0])+'.mp4'))>0:
            #     dic[str(data[0])] = {"video_image_path":video_path+str(data[0])+'.mp4',"title":data[2], "label":data[-1] }
            #生成一级分类数据
            if name == '动物.txt':
                label = '动物'
                x= 0
            if name == '动漫.txt':
                label = "动漫"
                x=1
            
            w1.write('{}\t{}\n'.format(data[2], x))
            dic[str(data[0])] = {"title":data[2], "label":label} 
            # if str(data[0])+'.mp4'+'.wav' in a_num and str(data[0])+'.mp4' in v_num and len(os.listdir(video_path+str(data[0])+'.mp4'))>0: 
            #     dic[str(data[0])] = {"title":data[2],"video_image_path":video_path+str(data[0])+'.mp4',\
            #         "video_audio_path": audio_path+str(data[0])+'.mp4'+'.wav',"label":label }
w.write(json.dumps(dic,ensure_ascii=False))
w.close()
w1.close()
