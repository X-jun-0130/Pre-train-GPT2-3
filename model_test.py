import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from transformers import BertTokenizer, GPT2LMHeadModel
import time


model_name = './results/checkpoint-26000/'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()


# '''
# inference
# '''
def infer(model, payload):
    input_token = tokenizer(payload)
    input_id = torch.LongTensor([input_token['input_ids']]).cuda()
    atten_mask = torch.LongTensor([input_token['attention_mask']]).cuda()
    logits = model.generate(input_id,attention_mask=atten_mask, num_beams=1, top_k=3, max_length=len(payload)+100, early_stop=True)
    out = tokenizer.decode(logits[0].tolist())
    out = out.replace('[PAD]', '')
    out = out.replace(' ', '')
    return out


infer_list = [
            '头孢替安禁忌症有哪些抽取句中药品：', 
            '头孢菌素类药过敏可以吃头孢匹胺吗？抽取句中药品：', 
            '头孢呋辛可以用来治疗尿路感染吗？抽取句中疾病：', 
            '氟达拉滨说明书，写一下抽取句中药品：', 
            '怎么办，胃炎胃胀难受，但我还有膀胱炎，能不能吃兰索拉唑？抽取句中疾病：',
            '什么样的人不能吃氟尿嘧啶？抽取句中药品：',
            "有黄疸病史，可以吃头孢他啶治疗蜂窝织炎吗？抽取句中疾病：",
            "可以吃阿罗洛尔治疗高血压吗？抽取句中疾病：",
            "慢性胃炎可以吃奥美拉唑胶囊吗，一天吃几次？抽取句中疾病：",
            "感冒发烧应该吃什么药？抽取句中疾病：",
            "使用奥司他韦时药每天吃多少抽取句中药品：",
            "萆薢分清丸每次吃多少抽取句中药品：",
            "肾石通这药有什么作用？抽取句中药品：",
            "育婴丸作用是什么抽取句中药品：",
            "孕妇能不能吃缓血酸胺噻洛芬酸治关节疼痛？抽取句中药品：",
            "有青光眼，能不能用科博肽抽取句中疾病：",
            "科博肽可以用于哪些病？抽取句中药品：",
            "失眠多梦可以吃益心宁神片吗抽取句中药品：",
            "百乃定的介绍抽取句中药品：",
            "白药子这个药怎么吃抽取句中药品：",
            "常遗留下不可逆肺损伤，后遗症有肺不张、肺纤维化、支气管扩张、反复发作性肺炎等。抽取句中疾病：",
            "抗真菌药物治疗有效，如氟康唑、两性霉素B、伏立康唑等。抽取句中药品："]


# data_set = [[k,j] for j,k in enumerate(infer_list[:3])]
# print(the_collate_fn(data_set))

s = time.time()
for payload in infer_list:
    model.eval()

    input_text = payload 
    out = infer(model, input_text)
    out = out.replace(input_text, '')

    print("="*70+" 模型输入输出 "+"="*70)
    print(f"模型输入: {payload}")
    print(f"模型输出: {out}")
e = time.time()
print('推理耗时：' , str(e-s)+ 's' )
