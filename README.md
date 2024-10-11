# CARF

### 运行
在alfworld_runs目录下执行`./run_reflexion.sh`以运行程序

执行之前需要在gpt_policy.py中添加自己的api-key
```
export OPENAI_API_KEY=<your key>
```

### 相关代码目录
```
run_reflexion.sh    存储一些控制参数
generate_reflections*.py   跟生成反思相关的文件
main.py   主流程函数
utils_router.py  跟GPT相关的交互接口文件
/reflexion_run_logs 存储运行结果的log文件


```
