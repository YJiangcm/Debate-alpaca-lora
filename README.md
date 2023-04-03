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

## Data Collection
We collected 1,560 discussion threads before January 2020 of Kialo. Each discussion forms an argument tree as the following figure shows. Except the thesis, every claim in the argument tree either opposes or supports its parent claim. Moreover, each claim has impact votes assigned by the users of the platform. The impact votes evaluate how impactful a claim is.

<img src="https://github.com/YJiangcm/Debate-alpaca-lora/blob/master/pics/kialo_example.png" width="600" height="300">

We extract each argument with its supported children claims or opposed children claims to form a training example. The children claims are ranked by the descending order of impact. We delete children claims whose votes are less than 5 and the average impact score less than 2. Finally, we obtain 20,998 training data formuated as the following examples:
```
{
    "instruction": "Support the following claim in descending order of impact.",
    "input": "Even if racial profiling is wrong, a ban on racial profiling is likely to be counterproductive.",
    "output": "1. Enforcing a ban on racial profiling is very difficult.\n2. Banning racial profiling may only further increase racial tensions"
},
{
    "instruction": "Oppose the following claim in descending order of impact.",
    "input": "Even if racial profiling is wrong, a ban on racial profiling is likely to be counterproductive.",
    "output": "1. Considering the deaths associated around race in relation to law enforcement, it would give much hope to communities where the attitude is that 'police are not there to help non-white people', declaring no race is different, and that all races deserve to be safe in public is not counterproductive by any means.\n2. Since racial profiling is morally unacceptable, it ought to be illegal regardless of the consequences of banning it"
},
```

## Training
We train our model based on [Alpaca LoRA](https://github.com/tloen/alpaca-lora). It costs about 1.5 hours on 2 RTX 3090Ti.
```
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 finetune.py \
    --base_model='decapoda-research/llama-7b-hf' \
    --resume_from_checkpoint 'alpaca-lora-7b' \
    --num_epochs=3 \
    --cutoff_len=256 \
    --group_by_length \
    --data_path 'kialo_debate.json' \
    --output_dir './DebateGPT_7b_001' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --micro_batch_size=16
```


## Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{debate-alpaca-lora,
  author={Yuxin Jiang},
  title = {An Instruction-following English debate model, LoRA tuning on LLaMA},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YJiangcm/Debate-alpaca-lora}},
}
```
