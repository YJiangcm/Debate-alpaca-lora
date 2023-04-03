# Debate-alpaca-lora
An English debate model by instruct-tuning LLaMA on [Kialo](https://www.kialo.com/) data. 

We may ask the model to **support** or **oppose** a claim by the desconding order of impact.

**A quick start for inference**: <a href="https://colab.research.google.com/drive/1emuN4LcO4X2-BCoATbhawIZeGBr-d3Y-?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> Some results are shown below:

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

## Well-trained Model

| Model Name | Hugging Face | Training Data and Setting |
| --- | --- | :--- |
| debate-alpaca-lora_7b_001 | [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97-huggingface-yellow)](https://huggingface.co/YuxinJiang/debate-alpaca-lora_7b_001) | Trained on 21k kialo data, 3 epoch |
| debate-alpaca-lora_7b_002 | [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97-huggingface-yellow)](https://huggingface.co/YuxinJiang/debate-alpaca-lora_7b_002) | Trained on 52k alpaca data + 21k kialo data, 3 epoch |

## Data Collection
We collected 1,560 discussion threads before January 2020 of Kialo. Each discussion forms an argument tree as the following figure shows. Except the thesis, every claim in the argument tree either opposes or supports its parent claim. Moreover, each claim has impact votes assigned by the users of the platform. The impact votes evaluate how impactful a claim is.

<img src="https://github.com/YJiangcm/Debate-alpaca-lora/blob/master/pics/kialo_example.png" width="600" height="300">

We extract each argument with its supported children claims or opposed children claims to form a training example. The children claims are ranked by the descending order of impact. We delete children claims whose votes are less than 5 and whose average impact score less than 2. Finally, we obtain 20,998 training data formuated as the following examples:
```
{
    "instruction": "Support the following claim in descending order of impact.",
    "input": "Pineapple belongs on pizza.",
    "output": "1. From a culinary point of view, it's always a good thing to have another recipe to make.\n2. Even if pizza does not remain authentically Italian, that is fine, as many foods have been reinvented in foreign cultures and remained delicious e.g. sushi.\n3. Pineapple is a cost effective topping, adding sweetness to the recipe that few other ingredients could at that price point.\n4. Pineapples make pizza aesthetically pleasing.\n5. It's no coincidence that pineapple and pizza both start with pi.\n6. The taste is unusual, which is a good way of changing a routine.\n7. Pineapple enhances the pizza-indulging experience, making pizza bring a greater joy to and satisfaction from eating than ever before.\n8. Sweet pineapple and salty Canadian bacon or ham brings an [umami](https://www.realsimple.com/food-recipes/recipe-collections-favorites/what-umami) that is acceptable the world over.\n9. The pineapple cleanses the palate from having an overly fatty and savory taste.\n10. Assuming meat is also involved, the idea of combining meat and fruit is in [no way new](http://www.dartagnan.com/meat-and-fruit-recipes-and-combinations.html).\n11. If [pizza wurstel](http://pizzawuerstel.de/uber/) is ok, pineapple on pizza must be ok, too.\n12. It is a way for students to have a main course and desert in one meal, enabling more time for exstensive learning.\n13. Pineapple adds a texture and mouthfeel \\(in between crisp and juicy\\) that is missing from most pizzas. Textural variety makes for a more pleasing meal"
},
{
    "instruction": "Oppose the following claim in descending order of impact.",
    "input": "Pineapple belongs on pizza.",
    "output": "1. Pineapple on pizza is an insult to the Italian origins of pizza.\n2. While pineapple is not an ideal pizza topping, it may be helped by being applied directly to the pizza surface by a qualified professional with the appropriate certifications in moisture removal procedures.\n3. You cannot eat Spongebob's house and be considered a good person at the same time.\n4. The sweetness of the pineapple is a bad fit for the general taste of the pizza.\n5. It's wrong in class terms. Pizza is proletarian food: good, cheap, and filling. Pineapples were a complete luxury when they first came to Europe, being displayed for [social status](https://www.foodbeast.com/news/til-people-used-to-rent-pineapples/) instead of eaten. This caused a divide, as the poor were deprived from eating it on pizza, while knowing the wealthy frivolously wasted it on decor. It's a class war on a plate, and that's exactly what it tastes like.\n6. Pineapple agriculture is [heavily polluting](http://www.ticotimes.net/2011/05/26/costa-rica-s-pineapple-boom-unhealthy-warn-experts), It destroys the lives of people in the tropics. Pizza is a large part of the demand for these pineapples.\n7. Torture is wrong. In today's day and age, we should have moved well beyond this kind of barbarism. It's cruel to a tropical fruit to be stuck on top of a pizza and be shoved into an oven.\n8. According to the [Oxford dictionary](https://en.oxforddictionaries.com/definition/pizza), pizza is \"a dish of Italian origin, consisting of a flat round base of dough baked with a topping of tomatoes and cheese, typically with added meat, fish, or vegetables\". Pineapple is a fruit.\n9. Eating pizza first and pineapple as dessert would make the whole meal experience better than together.\n10. Many people have spoken out publicly against pineapple pizza.\n11. Pineapple agriculture is bad for the environment.\n12. [Hawaiian pizza](https://en.wikipedia.org/wiki/Hawaiian_pizza) is a Canadian invention.\n13. Because of the incredible passion people have against putting pineapples on pizza, we ought not to combine the two, thus ending existing conflict and reducing the chance of future conflict, altogether leading towards world peace"
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
