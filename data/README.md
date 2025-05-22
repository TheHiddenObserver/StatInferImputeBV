\# Dataset Documentation 

## Overview 

This repository contains three core data components: 

1. ***\*Main Dataset\**** (`full_data_final.csv`): Comprehensive records from Douyin
2. ***\*Pilot Dataset\**** (`pilot_final.csv`): Dataset of pilot data including label
3. ***\*Audio Features\**** (`W_pilot_final.npy`): Feature vector extracted from audio, used for inference

## Variable Description

full_data_final.csv

- id: the unique id of the live
- time: real time of the live (timezone: UTC8)
- time-index: time index of the audio, each index stands for 5 secs
- positive: predicted probability of *valance* of the live streamer's emotion, 1 for positive，0 otherwise (including neural and negative)
- strong: predicted probability of *arousal* of the live streamer's emotion, 1 for strong, 0 otherwise
- danmaku: the number of danmakus appeared in this period
- people: the number of people entered the live in this period
- gift: the number of gift in this period
- follow: the number of new follows in this period
- like: the number of new likes in this period
- emotion: the emotion score of the danmaku
- emoji_positive, emoji_negative: the number of positive/negative emojis in danmaku in this period
- zifu: the number of danmaku characters in this period
- word_positive, word_negative: the number of positive/negative words in danmaku in this period
- fans: the estimated number of fans, computed by the sum of the number of fans before the live and incremental fans during the live
- date: the date of the live
- brand: brand of the live, including '一汽丰田' (FAW-Toyota),'一汽大众'(FAW-VK), '保时捷'(Porsche), '奔驰'(Benz), '奥迪'(Audi) and '宝马' (BMW)
- start_time: the start time of the live
- fans_count: the number of fans before the live
- gender: the gender of the live streamer, including '女'(female), '混杂'(mixed) and '男'(male)
- level: the level of the live in Douyin
- duration: the time duration of the live
- '一汽丰田','一汽大众', '保时捷', '奔驰', '奥迪', '宝马': dummy variables of the variable 'brand'
- '女', '混杂', '男': dummy variables of the variable 'gender'
- like_cum: cumulative number of likes before this period
- start_hour: the hour when the live started
- 'morning', 'noon', 'afternoon', 'evening': dummy variables generated from 'hour', morning=1 if 'hour' between 4-10, noon=1, if 'hour' bewteen10-14, afternoon = 1 if 'hour' between 14-18, 'evening' = 1 if 'hour' between 18-24
- weekend: binary variable indicating whether the live is held in weekend, 1 if weekend and 0 otherwise 
- 'fans_log', 'like_cum_log', 'zifu_log', 'duration_log': the log(1+x) transformation of the variables "fans", "like_cum", "zifu" and "duration", respectively

pilot_final.csv: the pilot dataset including predicted labels
- positive: *valance* of the live streamer's emotion, 1 for positive，0 otherwise (including neural and negative (binary variable)
- strong: *arousal* of the live streamer's emotion, 1 for strong, 0 otherwise
- positive_pre1: predicted probability of *valance* of the live streamer's emotion
- strong_pre1: predicted probability of *arousal* of the live streamer's emotion

## Contact

If you have further questions, contact  Ziqian Lin ([linziqian@stu.pku.edu.cn](mailto:linziqian@stu.pku.edu.cn)) or Danyang Huang ([dyhuang89@126.com](mailto:dyhuang89@126.com))