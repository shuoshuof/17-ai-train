# 17届全国大学生智能汽车竞赛 中国石油大学（华东）智能视觉组 --模型训练篇
+ [环境配置](./%E6%96%87%E6%A1%A3/%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.md)
+ [训练](./文档/%E8%AE%AD%E7%BB%83.md)
+ 量化
+ 超参搜索
+ 数据增强
+ 目标检测

# 概述
模型的精度在比赛中非常关键，毕竟分类错误的罚时是非常狠的。  
在这里分享下我们的经验：
## 摄像头放置
摄像头的高度跟你的镜头有关，也跟你模型输入的尺寸有关。一个简单的标准是，让实际输入的图像的尺寸大于等于模型输入的尺寸。  
在art上，我们可以打印find_rect得到的矩形的roi值，根据这个矩形的长宽来确定你的摄像头高度是否满足要求。
## 模型选择
不建议自行搭建模型，因为自己搭建的模型通常都是答辩，又大又臭，或者是太过简单，深度不够。  
我们很难超过那些优秀论文所提出的模型，比如mobilenet系列，它是由Google提出的;shufflenet系列，由旷视提出。这些大公司的模型的优越性经过了大量的验证，我们完全可以借鉴。   
所以，更应该做的是借鉴别人的模型，用别人的模型进行迁移学习，把更多的精力放在数据集的制作上面。  
## 数据集制作
一个好的数据集对模型的精度提升特别大。我们需要使用适合的增强手段。  
增强不是增得越多越好，比如对一张图片进行360度的增强得到360张图片，这样你虽然能得到大量的数据，但这样的数据过于臃肿，意义不大。  
我们希望增强后的图片可以接近真实情况下得到的图片（这样可以省下我们拍摄数据集的时间），但比真实图片的情况更恶劣些，但又不至于恶劣使得我们的网络学习不到特征。    
一个好的增强效果如下图：    
![增强图片](./%E6%96%87%E6%A1%A3/source_48.jpg)     
当然，如果有条件，还是最好手拍图片，然后配合增强可以达到非常好的效果。
## 模型评价
我们通常通过验证集或者说是测试集上的准确率来判断模型的好坏。但有时候你会发现你在验证集的正确率都接近百分百了，实际效果还是很不好。这样的原因是你的验证集过于简单了。所以我们需要一个符合要求的验证集。  
我们可以通过将车实际运行时art识别的图片保存到sd卡上作为验证集（这些图片也可以作为数据增强的根据和目标），或者用art进行拍摄。总而言之，需要与真实情况相符，这样验证集的准确率才会具有说服力。    
另一方面，我们可以通过混淆矩阵来判断每一小类的分类情况。混淆矩阵在eiq中是自带的，这里的例子也用plt简单进行了可视化。
## 好模型的标准
我们需要将模型训练到多高的精度才可以去摸鱼？    
假设我们有了一个符合要求的验证集，我们的精度可以达到0.95，那么这是否够了呢？    
我们可以大致计算下，17届线下决赛的图片数量为18张，那么一次发车全部识别正确的概率是多少呢？不到百分之40（$0.95^{18}$）。你的车可能在行进途中还会出其它问题，所以一次完美的发车（不罚时）出现的概率似乎不容乐观。事实上也确实如此，在线下决赛时许多队伍都没能完美发车，包括我们也是状况频出，当然这也有场地与灯光的原因。   
所以，模型的精度要尽可能高，要到0.97以上可能才是较为稳妥的。


