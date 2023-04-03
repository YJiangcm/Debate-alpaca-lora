# Debate-alpaca-lora
An English debate model by instruct-tuning LLaMA on [Kialo](https://www.kialo.com/) data. 

We may ask the model to **support** or **oppose** a claim by the desconding order of impact.

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
