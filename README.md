###### Askar Arslanov's school project. Republican Engineering Boarding Lyceum, Russia.

# ESPCN Super Resolution

![](https://i.imgur.com/TDZMhOx.png)

-

###### Make sure you have Python installed (preferably 3.9) and got all the packages from *requirements.txt*.
###### Before launching a script set up *config.json*.

### parse.py
Creates a folder *dataset* and loads `dataset_size` images from the website there.

Won't run if you already have the folder.


### learn.py
Creates a folder named as `model_type` and places here the model.

Won't run if you already have the folder or don't have a dataset.


### bot.py
Launches the Telegram-bot with token `bot_token` that upscales images.

Won't run if you don't have a model.
