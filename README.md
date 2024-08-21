# test_trocr
TrOCR – Getting Started with Transformer Based OCR

# References
* https://learnopencv.com/trocr-getting-started-with-transformer-based-ocr/
* https://learnopencv.com/fine-tuning-trocr-training-trocr-to-recognize-curved-text/

## 1. Setting up the Environment for silicon mac 
    conda create -n tr_ocr_pip python==3.9.18
    conda activate tr_ocr_pip
    
    pip install transformers==4.43.3
    pip install sentencepiece==0.2.0
    pip install jiwer==3.0.4
    pip install datasets==2.20.0
    pip install evaluate==0.4.2
    pip install -U accelerate==0.33.0  # torch-2.4.0

    pip install matplotlib==3.9.0
    pip install protobuf==3.20.1  # must v3.20.1
    pip install tensorboard==2.17.0
    pip install torchvision==0.19.0

## 2. Setting up the Environment for ubuntu 
    conda create -n tr_ocr_pip python==3.8.13
    conda activate tr_ocr_pip
    
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    # # Pytorch 설치 후, GPU 확인
        >>> import torch
        >>> torch.cuda.is_available()
        True

    pip install transformers==4.43.3
    pip install sentencepiece==0.2.0
    pip install jiwer==3.0.4
    pip install datasets==2.20.0
    pip install evaluate==0.4.2
    pip install -U accelerate==0.33.0

    pip install matplotlib==3.7.5
    pip install protobuf==3.20.1  # must v3.20.1
    pip install tensorboard==2.14.0
