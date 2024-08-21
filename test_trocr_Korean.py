from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import requests
from io import BytesIO
from PIL import Image
import time
import glob


def get_trocr_processor(model_nm):
    return TrOCRProcessor.from_pretrained(model_nm)


def get_trocr_model(model_nm):
    return VisionEncoderDecoderModel.from_pretrained(model_nm)


def get_tokenizer(model_nm):
    return AutoTokenizer.from_pretrained(model_nm)


def test_0():  # # https://github.com/daekeun-ml/sm-kornlp-usecases/tree/main/trocr
    test_for = "printed"  # "handwritten"
    processor = get_trocr_processor(f'microsoft/trocr-small-{test_for}')
    # processor = get_trocr_processor("microsoft/trocr-base-handwritten")

    model = get_trocr_model("daekeun-ml/ko-trocr-base-nsmc-news-chatbot")
    tokenizer = get_tokenizer("daekeun-ml/ko-trocr-base-nsmc-news-chatbot")

    url = "https://raw.githubusercontent.com/aws-samples/sm-kornlp/main/trocr/sample_imgs/news_1.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    pixel_values = processor(img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=64)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"[{generated_text}]")
    return generated_text


def test_1():
    test_for = "printed"  # "handwritten", "printed"
    # processor = get_trocr_processor(f'microsoft/trocr-small-{test_for}')
    processor = get_trocr_processor(f"microsoft/trocr-base-{test_for}")

    model = get_trocr_model("team-lucid/trocr-small-korean")
    tokenizer = get_tokenizer("team-lucid/trocr-small-korean")

    url = "https://raw.githubusercontent.com/aws-samples/sm-kornlp/main/trocr/sample_imgs/news_1.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    pixel_values = processor(img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=64)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"[{generated_text}]")
    return generated_text


def test_1_file(img):
    test_for = "printed"  # "handwritten", "printed"
    processor = get_trocr_processor(f'microsoft/trocr-small-{test_for}')
    # processor = get_trocr_processor(f"microsoft/trocr-base-{test_for}")

    model = get_trocr_model("team-lucid/trocr-small-korean")
    tokenizer = get_tokenizer("team-lucid/trocr-small-korean")

    pixel_values = processor(img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_length=32)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


if __name__ == '__main__':
    # st_time = time.time()
    # res = test_0()
    # res = test_1()
    img_path_list = glob.glob("./images/test_ocr/*.jpg")
    # img_path_list = glob.glob("./images/newspaper/*.png")
    for img_path in img_path_list:
        st_time = time.time()
        img = Image.open(img_path)
        # remove alpha channel
        img = img.convert("RGB")
        res = test_1_file(img)
        print(f"[{res}]")
        print("::::::::::: %.2f seconds ::::::::::::::" % (time.time() - st_time))
