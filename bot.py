import json
import os
import shutil

import aiogram
import cv2 as cv
import numpy as np
import tensorflow as tf

with open('config.json') as config_file:
    config = json.load(config_file)

if config['model_type'] == 'dpsnr':
    def dpsnr(y_true, y_pred):
        return 108.79928 - tf.image.psnr(y_true, y_pred, 1)


    with tf.keras.utils.custom_object_scope({'dpsnr': dpsnr}):
        model = tf.keras.models.load_model('dpsnr/model.h5')

elif config['model_type'] == 'dssim':
    def dssim(y_true, y_pred):
        return 1 - tf.image.ssim(y_true, y_pred, 1)


    with tf.keras.utils.custom_object_scope({'dssim': dssim}):
        model = tf.keras.models.load_model('dssim/model.h5')

elif config['model_type'] == 'mse':
    model = tf.keras.models.load_model('mse/model.h5')

dp = aiogram.Dispatcher(aiogram.Bot(config['bot_token']))

del config

user_requests = {}  # user_id: [file_name, opencv_image]
NoneType = type(None)


def predict(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    y = y.astype(np.float32) / 255
    y = np.expand_dims(y, (0, -1))

    y = model.predict(y)
    y = y[0] * 255
    y[y < 0] = 0
    y[y > 255] = 255
    y = y.astype(np.uint8)

    cr = cv.resize(cr, (y.shape[0], y.shape[1]), interpolation=cv.INTER_CUBIC)
    cr = np.expand_dims(cr, -1)

    cb = cv.resize(cb, (y.shape[0], y.shape[1]), interpolation=cv.INTER_CUBIC)
    cb = np.expand_dims(cb, -1)

    ycrcb = np.concatenate((y, cr, cb), 2)
    return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)


def upscale(image):
    if image.shape[0] < image.shape[1]:
        result = predict(image[:, :image.shape[0], :])

        for i in range(image.shape[0], image.shape[1] - image.shape[0], image.shape[0]):
            result = np.hstack((result, predict(image[:, i: i + image.shape[0], :])))

        try:
            return np.hstack((
                result,
                predict(image[:, -image.shape[0]:, :])[:, -4 * (image.shape[1] % image.shape[0]):, :]
            ))
        except:
            return result

    elif image.shape[0] > image.shape[1]:
        result = predict(image[:image.shape[1], :, :])

        for i in range(image.shape[1], image.shape[0] - image.shape[1], image.shape[1]):
            result = np.vstack((result, predict(image[i: i + image.shape[1], :, :])))

        try:
            return np.vstack((
                result,
                predict(image[-image.shape[1]:, :, :])[-4 * (image.shape[0] % image.shape[1]):, :, :]
            ))
        except:
            return result

    else:
        return predict(image)


@dp.message_handler(commands='start')
async def start(message: aiogram.types.Message):
    # print('+', end=' ')
    # if message.from_user.username is not None:
    #     print(f'{message.from_user.username} ({message.from_user.id})', end=' ')
    # else:
    #     print(message.from_user.id, end=' ')
    # print('+')

    if message.from_user.language_code in ('ru', 'uk', 'be'):
        await message.answer(
            'Привет!\n'
            'Я бот, который умеет увеличивать качество изображений.\n'
            'Отправь мне картинку и проверь!\n'
            '\n'
            '<i>Рекомендую загружать в виде документа.</i>',
            aiogram.types.ParseMode.HTML
        )
    else:
        await message.answer(
            'Hi!\n'
            "I'm a bot that can increase images quality.\n"
            'Send me a picture and check it!\n'
            '\n'
            '<i>I recommend to upload as a document.</i>',
            aiogram.types.ParseMode.HTML
        )


@dp.message_handler(content_types=aiogram.types.ContentTypes.ANY)
async def response(message: aiogram.types.Message):
    if message.from_user.id in user_requests:
        # print('*', end=' ')
        # if message.from_user.username is not None:
        #     print(f'{message.from_user.username} ({message.from_user.id})')
        # else:
        #     print(message.from_user.id)

        if message.from_user.language_code in ('ru', 'uk', 'be'):
            await message.answer(
                'Подожди!\n'
                'Можно отправить лишь один запрос за раз!'
            )
        else:
            await message.answer(
                'Wait!\n'
                'You can only make one request at a time!'
            )
        return

    elif message.content_type == 'document':
        # print('+', end=' ')
        # if message.from_user.username is not None:
        #     print(f'{message.from_user.username} ({message.from_user.id})')
        # else:
        #     print(message.from_user.id)

        user_requests[message.from_user.id] = [
            f'processing/{message.from_user.id}/'
            f'{os.path.splitext(message.document.file_name)[0]}.png',
            None
        ]

        if message.document.file_size > 15_728_640:
            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer(
                    'Файл слишком большой!\n'
                    '\n'
                    '<i>Вес должен быть не больше 15 МБ.</i>',
                    aiogram.types.ParseMode.HTML
                )
            else:
                await message.answer(
                    'The file is too large!\n'
                    '\n'
                    "<i>The weight mustn't be more than 15 MB.</i>",
                    aiogram.types.ParseMode.HTML
                )

            user_requests.pop(message.from_user.id)

            # print('-', end=' ')
            # if message.from_user.username is not None:
            #     print(f'{message.from_user.username} ({message.from_user.id})')
            # else:
            #     print(message.from_user.id)

            return

        await message.document.download(user_requests[message.from_user.id][0])
        user_requests[message.from_user.id][1] = cv.imread(user_requests[message.from_user.id][0])

        if isinstance(user_requests[message.from_user.id][1], NoneType):
            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer(
                    'Упс! Произошла ошибка...\n'
                    'Проверь, пожалуйста, файл и его имя.'
                )
            else:
                await message.answer(
                    'Oops! An error has occurred...\n'
                    'Please check the file and its name.'
                )

        elif user_requests[message.from_user.id][1].shape[0] > 1080 and \
                user_requests[message.from_user.id][1].shape[1] > 1080:
            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer(
                    'Фотография слишком большая!\n'
                    '\n'
                    '<i>Размер должен быть не больше 1080x1080 пикс.</i>',
                    aiogram.types.ParseMode.HTML
                )
            else:
                await message.answer(
                    'The photo is too large!\n'
                    '\n'
                    "<i>The size mustn't be more than 1080x1080 px.</i>",
                    aiogram.types.ParseMode.HTML
                )

        elif user_requests[message.from_user.id][1].shape[0] > 1080:
            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer(
                    'Фотография слишком длинная!\n'
                    '\n'
                    '<i>Высота должна быть не больше 1080 пикс.</i>',
                    aiogram.types.ParseMode.HTML
                )
            else:
                await message.answer(
                    'The photo is too long!\n'
                    '\n'
                    "<i>The height mustn't be more than 1080 px.</i>",
                    aiogram.types.ParseMode.HTML
                )

        elif user_requests[message.from_user.id][1].shape[1] > 1080:
            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer(
                    'Фотография слишком широкая!\n'
                    '\n'
                    '<i>Ширина должна быть не больше 1080 пикс.</i>',
                    aiogram.types.ParseMode.HTML
                )
            else:
                await message.answer(
                    'The photo is too wide!\n'
                    '\n'
                    "<i>The weight mustn't be more than 1080 px.</i>",
                    aiogram.types.ParseMode.HTML
                )

        else:
            cv.imwrite(user_requests[message.from_user.id][0],
                       upscale(user_requests[message.from_user.id][1]))

            await message.answer_document(open(user_requests[message.from_user.id][0], 'rb'))

        shutil.rmtree(f'processing/{message.from_user.id}')
        user_requests.pop(message.from_user.id)

        # print('-', end=' ')
        # if message.from_user.username is not None:
        #     print(f'{message.from_user.username} ({message.from_user.id})')
        # else:
        #     print(message.from_user.id)

    elif message.content_type == 'photo':
        # print('+', end=' ')
        # if message.from_user.username is not None:
        #     print(f'{message.from_user.username} ({message.from_user.id})')
        # else:
        #     print(message.from_user.id)

        user_requests[message.from_user.id] = [
            f'processing/{message.from_user.id}/Result.png',
            None
        ]

        await message.photo[-1].download(user_requests[message.from_user.id][0])
        user_requests[message.from_user.id][1] = cv.imread(user_requests[message.from_user.id][0])

        if user_requests[message.from_user.id][1].shape[0] > 1080 and \
                user_requests[message.from_user.id][1].shape[1] > 1080:
            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer(
                    'Фотография слишком большая!\n'
                    '\n'
                    '<i>Размер должен быть не больше 1080x1080 пикс.</i>',
                    aiogram.types.ParseMode.HTML
                )
            else:
                await message.answer(
                    'The photo is too large!\n'
                    '\n'
                    "<i>The size mustn't be more than 1080x1080 px.</i>",
                    aiogram.types.ParseMode.HTML
                )

        elif user_requests[message.from_user.id][1].shape[0] > 1080:
            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer(
                    'Фотография слишком длинная!\n'
                    '\n'
                    '<i>Высота должна быть не больше 1080 пикс.</i>',
                    aiogram.types.ParseMode.HTML
                )
            else:
                await message.answer(
                    'The photo is too long!\n'
                    '\n'
                    "<i>The height mustn't be more than 1080 px.</i>",
                    aiogram.types.ParseMode.HTML
                )

        elif user_requests[message.from_user.id][1].shape[1] > 1080:
            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer(
                    'Фотография слишком широкая!\n'
                    '\n'
                    '<i>Ширина должна быть не больше 1080 пикс.</i>',
                    aiogram.types.ParseMode.HTML
                )
            else:
                await message.answer(
                    'The photo is too wide!\n'
                    '\n'
                    "<i>The weight mustn't be more than 1080 px.</i>",
                    aiogram.types.ParseMode.HTML
                )

        else:
            cv.imwrite(user_requests[message.from_user.id][0],
                       upscale(user_requests[message.from_user.id][1]))

            if message.from_user.language_code in ('ru', 'uk', 'be'):
                await message.answer_document(
                    open(user_requests[message.from_user.id][0], 'rb'),
                    caption='Лучше в следующий раз отправь в виде документа.'
                )
            else:
                await message.answer_document(
                    open(user_requests[message.from_user.id][0], 'rb'),
                    caption='Better at the next time send as a document.'
                )

        shutil.rmtree(f'processing/{message.from_user.id}')
        user_requests.pop(message.from_user.id)

        # print('-', end=' ')
        # if message.from_user.username is not None:
        #     print(f'{message.from_user.username} ({message.from_user.id})')
        # else:
        #     print(message.from_user.id)

    else:
        # print('?', end=' ')
        # if message.from_user.username is not None:
        #     print(f'{message.from_user.username} ({message.from_user.id})')
        # else:
        #     print(message.from_user.id)

        if message.from_user.language_code in ('ru', 'uk', 'be'):
            await message.answer('Я умею работать только с фотографиями.')
        else:
            await message.answer('I can work only with photos.')


@dp.errors_handler(exception=aiogram.exceptions.BotBlocked)
async def blocked(update: aiogram.types.Update, exception: aiogram.exceptions.BotBlocked):
    shutil.rmtree(f'processing/{update.message.from_user.id}')
    user_requests.pop(update.message.from_user.id)

    # print('-', end=' ')
    # if update.message.from_user.username is not None:
    #     print(f'{update.message.from_user.username} ({update.message.from_user.id})', end=' ')
    # else:
    #     print(update.message.from_user.id, end=' ')
    # print('-')

    return True


aiogram.executor.start_polling(dp, skip_updates=True)
