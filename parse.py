import json
import os

import bs4
import cv2 as cv
import requests

os.mkdir('dataset')

with open('config.json') as config_file:
    dataset_size = json.load(config_file)['dataset_size']

NoneType = type(None)

num = 1
page = 0

while num <= dataset_size:
    for i in bs4.BeautifulSoup(
            requests.get(f'{"https://wallpaperscraft.ru/all/3840x2160"}/page{page}').text, 'lxml'
    ).find('ul', class_='wallpapers__list').find_all('li', class_='wallpapers__item'):
        with open(f'dataset/{num}.png', 'wb') as raw:
            bs = bs4.BeautifulSoup(
                requests.get('https://wallpaperscraft.ru' + i.find('a').get('href')).text,
                'lxml'
            ).find('div', class_='gui-toolbar__item gui-hidden-mobile')

            if isinstance(bs, NoneType):
                break

            raw.write(requests.get(
                bs.find('a').get('href')
            ).content)

        img = cv.imread(f'dataset/{num}.png')
        os.remove(f'dataset/{num}.png')

        if img.shape[0] < img.shape[1]:
            px = 1080 * img.shape[1] / img.shape[0]

            if int(px) % 2 == 1:
                px = (px + 1) // 2
            else:
                px = int(px)

            cv.imwrite(f'dataset/{str(num).rjust(5, "0")}.png',
                       cv.resize(img, (px, 1080), cv.INTER_AREA)[:, px // 2 - 540: px // 2 + 540])

        else:
            px = 1080 * img.shape[0] / img.shape[1]

            if int(px) % 2 == 1:
                px = (px + 1) // 2
            else:
                px = int(px)

            cv.imwrite(f'dataset/{str(num).rjust(5, "0")}.png',
                       cv.resize(img, (1080, px), cv.INTER_AREA)[px // 2 - 540: px // 2 + 540, :])

        print(f'{round(100 * num / dataset_size, 2)}%')
        num += 1

        if num > dataset_size:
            break

    page += 1
