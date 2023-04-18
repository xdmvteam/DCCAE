# 数据处理方法
```bash
./download_data.sh     # 下载数据
python3 preprocess.py  # 转换数据格式为本实验形式
```

最后获得三个文件：`train.pkl`,`valid.pkl`,`test.pkl`可供实验使用。

以`train.pkl`示例数据格式：
```python
[
    # dict key 表示视角编号；array(...)代表该视角的样本
    {0:[array(...), array(...), ...], 1:[array(...), array(...), ...], ...},
    [int, int, ...]  # 每个样本的标签
]
```
请使用以下语句读取：
```bash
 pickle.load(open('./train.pkl', 'rb'))
```