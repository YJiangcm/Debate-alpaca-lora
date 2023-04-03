# Debate-alpaca-lora
An English debate model by instruct-tuning LLaMA on [Kialo](https://www.kialo.com/) data. 

We may ask the model to **support** or **oppose** a claim by the desconding order of impact. <a href="https://colab.research.google.com/github/LC1332/Chinese-alpaca-lora/blob/main/notebook/TuoLingC_evaluation_code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

For example,

```
Instruction: Support the following claim in descending order of impact.
Input: Buying and selling sex should be legal for sex workers and their clients.
Response: 1. Prohibiting prostitution makes sex workers more vulnerable to violence and abuse.
2. legalizing prostitution will make it easier to regulate and reduce the harm done to sex workers.
```

```
Instruction: Oppose the following claim in descending order of impact.
Input: Buying and selling sex should be legal for sex workers and their clients.
Response: 1. Sex work is an industry that can be detrimental to society, as it can [undermine the social norms](http://www.apa.org/pi/ses/resources/sex-work.pdf) that hold it together.
2. Sex work is [degrading] (https://www.psychologytoday.com/blog/in-the-name-love/201106/prostitution-degrading-and-dehumanizing-women) to women and has a [damaging effect] (http://www.apa.org/pi/ses/resources/sex-work.pdf) on society.
```



## Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{alpaca,
  author={Yuxin Jiang},
  title = {An Instruction-following English debate model, LoRA tuning on LLaMA},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YJiangcm/Debate-alpaca-lora}},
}
```
